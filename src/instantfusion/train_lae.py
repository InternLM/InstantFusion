import argparse
import math
import sys
from pathlib import Path
import torch
import torch.nn.functional as F
from diffsynth import ModelManager
from diffsynth.trainers.text_to_image import LightningModelForT2ILoRA, add_general_parsers, launch_training_task
from .model_utils import expand_qwen_image_component_path, resolve_torch_dtype, load_sd3_vae_encoder, load_sd3_dit
from .bridge import build_sd3_bridge
from .qwen import QwenPipeline
from .prompt_cache import load_prompt_embedding_cache, single_prompt
VELOCITY_DOMAINS = ('sd3', 'qwen')

class SD3VelocityLAE(LightningModelForT2ILoRA):

    def __init__(self, torch_dtype, qwen_path, sd3_path, prompt_embedding_dir, learning_rate=0.0001, sampling_steps=50, exponential_shift_mu=math.log(3.0), timesteps_per_batch=1, shared_channels=32, hidden_channels=64, bridge_num_res_blocks=3, bridge_sigma_embedding_dim=128, reconstruction_weight=1.0, cross_reconstruction_weight=1.0, alignment_weight=0.1, velocity_weight=0.05, shared_norm_weight=0.0001, sd3_t5_sequence_length=512):
        super().__init__(learning_rate=learning_rate, use_gradient_checkpointing=False)
        if sampling_steps < 2:
            raise ValueError('--sampling_steps must be at least 2')
        if timesteps_per_batch < 1:
            raise ValueError('--timesteps_per_batch must be at least 1')
        if not math.isfinite(velocity_weight) or velocity_weight <= 0:
            raise ValueError('LAE requires velocity_weight > 0')
        qwen_path = Path(qwen_path)
        self.prompt_embedding_dir = Path(prompt_embedding_dir)
        if not self.prompt_embedding_dir.is_dir():
            raise FileNotFoundError(f'Missing prompt embedding cache: {self.prompt_embedding_dir}')
        manager = ModelManager(torch_dtype=torch_dtype, device=self.device)
        manager.load_models([expand_qwen_image_component_path(qwen_path / 'transformer'), expand_qwen_image_component_path(qwen_path / 'vae')])
        self.qwen_pipe = QwenPipeline(device=self.device, torch_dtype=torch_dtype, qwen_tokenizer_path=str(qwen_path / 'tokenizer'), turbo_exponential_shift_mu=exponential_shift_mu)
        self.qwen_pipe.dit = manager.fetch_model('qwen_image_dit')
        self.qwen_pipe.vae = manager.fetch_model('qwen_image_vae')
        self.sd3_dit = load_sd3_dit(sd3_path, torch_dtype, self.device)
        self.sd3_vae = load_sd3_vae_encoder(sd3_path, torch_dtype, self.device)
        self.qwen_pipe.scheduler.set_timesteps(sampling_steps, exponential_shift_mu=exponential_shift_mu)
        self.bridge = build_sd3_bridge(bridge_type='resblock', native_channels=16, shared_channels=shared_channels, hidden_channels=hidden_channels, num_res_blocks=bridge_num_res_blocks, sigma_embedding_dim=bridge_sigma_embedding_dim)
        self.timesteps_per_batch = int(timesteps_per_batch)
        self.reconstruction_weight = float(reconstruction_weight)
        self.cross_reconstruction_weight = float(cross_reconstruction_weight)
        self.alignment_weight = float(alignment_weight)
        self.velocity_weight = float(velocity_weight)
        self.shared_norm_weight = float(shared_norm_weight)
        self.sd3_t5_sequence_length = int(sd3_t5_sequence_length)
        self.freeze_parameters()

    def freeze_parameters(self):
        self.qwen_pipe.requires_grad_(False).eval()
        self.sd3_dit.requires_grad_(False).eval()
        self.sd3_vae.requires_grad_(False).eval()
        self.bridge.requires_grad_(True).train()
        for parameter in self.bridge.parameters():
            parameter.data = parameter.data.float()

    def encode_prompts(self, text):
        prompt = single_prompt(text)
        dtype = self.qwen_pipe.torch_dtype
        return {'qwen': load_prompt_embedding_cache(self.prompt_embedding_dir, 'qwen', prompt, self.device, dtype, self.sd3_t5_sequence_length), 'sd3': load_prompt_embedding_cache(self.prompt_embedding_dir, 'sd3', prompt, self.device, dtype, self.sd3_t5_sequence_length)}

    def sample_timestep_ids(self):
        ids = torch.randint(0, len(self.qwen_pipe.scheduler.timesteps) - 1, (self.timesteps_per_batch,), dtype=torch.long)
        if torch.distributed.is_available() and torch.distributed.is_initialized():
            ids = ids.to(self.device)
            torch.distributed.broadcast(ids, src=0)
            ids = ids.cpu()
        return ids

    def forward_qwen(self, latents, timestep, prompt, height, width):
        self.qwen_pipe.device = self.device
        return self.qwen_pipe.forward_dit(latents, timestep=timestep, prompt_emb=prompt, height=height, width=width)

    def forward_sd3(self, latents, timestep, prompt):
        return self.sd3_dit(latents, timestep=timestep, **prompt, use_gradient_checkpointing=False)

    def losses_at_timestep(self, clean, noise, prompts, timestep_id, height, width):
        scheduler = self.qwen_pipe.scheduler
        timestep = scheduler.timesteps[timestep_id].reshape(1).to(self.device)
        sigma = scheduler.sigmas[timestep_id].to(device=self.device, dtype=torch.float32)
        next_sigma = scheduler.sigmas[timestep_id + 1].to(device=self.device, dtype=torch.float32)
        delta_sigma = next_sigma - sigma
        native = {domain: scheduler.add_noise(clean[domain], noise, timestep) for domain in VELOCITY_DOMAINS}
        with torch.no_grad():
            native_velocity = {'qwen': self.forward_qwen(native['qwen'], timestep, prompts['qwen'], height, width), 'sd3': self.forward_sd3(native['sd3'], timestep, prompts['sd3'])}
            native_next = {domain: scheduler.step(native_velocity[domain], timestep, native[domain]) for domain in VELOCITY_DOMAINS}
        shared = {domain: self.bridge.encode(native[domain], sigma, domain) for domain in VELOCITY_DOMAINS}
        shared_next = {domain: self.bridge.encode(native_next[domain], next_sigma, domain) for domain in VELOCITY_DOMAINS}
        shared_velocity = {domain: (shared_next[domain] - shared[domain]) / delta_sigma for domain in VELOCITY_DOMAINS}
        reconstruction = torch.stack([F.mse_loss(self.bridge.decode(shared[domain], sigma, domain).float(), native[domain].float()) for domain in VELOCITY_DOMAINS]).mean()
        cross_reconstruction = torch.stack([F.mse_loss(self.bridge.decode(shared['sd3'], sigma, 'qwen').float(), native['qwen'].float()), F.mse_loss(self.bridge.decode(shared['qwen'], sigma, 'sd3').float(), native['sd3'].float())]).mean()
        alignment = F.mse_loss(shared['sd3'].float(), shared['qwen'].float())
        velocity = F.mse_loss(shared_velocity['sd3'].float(), shared_velocity['qwen'].float())
        shared_norm = torch.stack([value.float().square().mean() for value in shared.values()]).mean()
        return (reconstruction, cross_reconstruction, alignment, velocity, shared_norm)

    def training_step(self, batch, batch_idx):
        (text, image) = (batch['text'], batch['image'])
        (height, width) = image.shape[-2:]
        image = image.to(device=self.device, dtype=self.qwen_pipe.torch_dtype)
        with torch.no_grad():
            clean = {'qwen': self.qwen_pipe.vae.encode(image), 'sd3': self.sd3_vae(image)}
            prompts = self.encode_prompts(text)
        if clean['qwen'].shape != clean['sd3'].shape:
            raise ValueError(f"Native latent shapes differ: qwen={clean['qwen'].shape}, sd3={clean['sd3'].shape}")
        noise = torch.randn_like(clean['qwen'])
        values = [self.losses_at_timestep(clean, noise, prompts, int(timestep_id), height, width) for timestep_id in self.sample_timestep_ids()]
        (reconstruction, cross_reconstruction, alignment, velocity, shared_norm) = [torch.stack(items).mean() for items in zip(*values)]
        loss = self.reconstruction_weight * reconstruction + self.cross_reconstruction_weight * cross_reconstruction + self.alignment_weight * alignment + self.velocity_weight * velocity + self.shared_norm_weight * shared_norm
        self.log('train_loss', loss, prog_bar=True)
        self.log('reconstruction_loss', reconstruction)
        self.log('cross_reconstruction_loss', cross_reconstruction)
        self.log('alignment_loss', alignment)
        self.log('velocity_alignment_loss', velocity, prog_bar=True)
        return loss

    def configure_optimizers(self):
        return torch.optim.AdamW(self.bridge.parameters(), lr=self.learning_rate)

    def on_save_checkpoint(self, checkpoint):
        checkpoint.clear()
        checkpoint.update({f'shared_latent_bridge.{name}': value for (name, value) in self.bridge.state_dict().items()})

def velocity_parse_args():
    parser = argparse.ArgumentParser(description='Joint SD3/Qwen shared bridge and one-step velocity alignment')
    parser.add_argument('--qwen_path', required=True)
    parser.add_argument('--sd3_path', required=True)
    parser.add_argument('--prompt_embedding_dir', required=True)
    parser.add_argument('--sampling_steps', type=int, default=50)
    parser.add_argument('--exponential_shift_mu', type=float, default=math.log(3.0))
    parser.add_argument('--timesteps_per_batch', type=int, default=1)
    parser.add_argument('--shared_channels', type=int, default=32)
    parser.add_argument('--hidden_channels', type=int, default=64)
    parser.add_argument('--bridge_num_res_blocks', type=int, default=3)
    parser.add_argument('--bridge_sigma_embedding_dim', type=int, default=128)
    parser.add_argument('--reconstruction_weight', type=float, default=1.0)
    parser.add_argument('--cross_reconstruction_weight', type=float, default=1.0)
    parser.add_argument('--alignment_weight', type=float, default=0.1)
    parser.add_argument('--velocity_weight', type=float, default=0.05)
    parser.add_argument('--shared_norm_weight', type=float, default=0.0001)
    parser.add_argument('--sd3_t5_sequence_length', type=int, default=512)
    return add_general_parsers(parser).parse_args()

def run_velocity(prepare_cache=False):
    args = velocity_parse_args()
    if not math.isfinite(args.velocity_weight) or args.velocity_weight <= 0:
        raise ValueError('LAE requires velocity_weight > 0')
    if prepare_cache:
        from .prompt_cache import prepare_prompt_cache
        prepare_prompt_cache(args.dataset_path, args.prompt_embedding_dir, args.qwen_path, args.sd3_path, precision=args.precision, t5_length=getattr(args, 'sd3_t5_sequence_length', 512))
    model = SD3VelocityLAE(torch_dtype=resolve_torch_dtype(args.precision), qwen_path=args.qwen_path, sd3_path=args.sd3_path, prompt_embedding_dir=args.prompt_embedding_dir, learning_rate=args.learning_rate, sampling_steps=args.sampling_steps, exponential_shift_mu=args.exponential_shift_mu, timesteps_per_batch=args.timesteps_per_batch, shared_channels=args.shared_channels, hidden_channels=args.hidden_channels, bridge_num_res_blocks=args.bridge_num_res_blocks, bridge_sigma_embedding_dim=args.bridge_sigma_embedding_dim, reconstruction_weight=args.reconstruction_weight, cross_reconstruction_weight=args.cross_reconstruction_weight, alignment_weight=args.alignment_weight, velocity_weight=args.velocity_weight, shared_norm_weight=args.shared_norm_weight, sd3_t5_sequence_length=args.sd3_t5_sequence_length)
    launch_training_task(model, args)

def main():
    frontend = argparse.ArgumentParser(add_help=False, description='Train an SD3/Qwen LAE with joint state and velocity alignment.')
    frontend.add_argument('--model', '--experiment', dest='model', choices=('sd3-qwen',), default='sd3-qwen')
    frontend.add_argument('--prepare-prompt-cache', action='store_true')
    frontend.add_argument('--list-models', action='store_true')
    (args, forwarded) = frontend.parse_known_args()
    if args.list_models:
        print('sd3-qwen')
        return
    if forwarded and forwarded[0] == '--':
        forwarded = forwarded[1:]
    if '--help' in forwarded or '-h' in forwarded:
        frontend.print_help()
    defaults = {'--learning_rate': '1e-4', '--velocity_weight': '0.05'}
    supplied = {flag.split('=', 1)[0] for flag in forwarded if flag.startswith('--')}
    for (flag, value) in defaults.items():
        if flag not in supplied:
            forwarded += [flag, value]
    sys.argv = [sys.argv[0], *forwarded]
    run_velocity(args.prepare_prompt_cache)
if __name__ == '__main__':
    main()
