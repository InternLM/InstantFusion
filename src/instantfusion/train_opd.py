import csv
from contextlib import contextmanager
from pathlib import Path
import torch
import torch.nn.functional as F
from lightning.pytorch.callbacks import ModelCheckpoint
from peft.tuners.tuners_utils import BaseTunerLayer
from diffsynth.trainers.text_to_image import LightningModelForT2ILoRA, add_general_parsers, launch_training_task
from .model_utils import SD3QwenBackbones, resolve_torch_dtype
from .cli import parse_training_args

class LossCurve:

    def __init__(self, log_dir):
        log_dir = Path(log_dir)
        log_dir.mkdir(parents=True, exist_ok=True)
        self.csv_path = log_dir / 'loss.csv'
        self.image_path = log_dir / 'loss_curve.png'
        self.batch_step = 0
        with self.csv_path.open('w', newline='') as file:
            csv.writer(file).writerow(('batch_step', 'supervised_step_ids', 'sigmas', 'train_loss', 'distillation_loss', 'shared_velocity_mse', 'reference_velocity_loss'))

    def append(self, supervised_step_ids, sigmas, loss, distillation_loss, shared_velocity_mse, reference_loss):
        with self.csv_path.open('a', newline='') as file:
            csv.writer(file).writerow((self.batch_step, ';'.join((str(step_id) for step_id in supervised_step_ids)), ';'.join((f'{sigma:.8g}' for sigma in sigmas)), loss, distillation_loss, shared_velocity_mse, reference_loss))
        self.batch_step += 1

    def save_image(self):
        from PIL import Image, ImageDraw
        steps = []
        losses = []
        with self.csv_path.open(newline='') as file:
            for row in csv.DictReader(file):
                steps.append(int(row['batch_step']))
                losses.append(float(row['train_loss']))
        (width, height) = (1280, 800)
        (left, top, right, bottom) = (100, 70, 40, 80)
        image = Image.new('RGB', (width, height), 'white')
        draw = ImageDraw.Draw(image)
        draw.rectangle((left, top, width - right, height - bottom), outline='black', width=2)
        minimum = min(losses)
        maximum = max(losses)
        loss_range = max(maximum - minimum, 1e-12)
        plot_width = width - left - right
        plot_height = height - top - bottom
        points = []
        for (index, loss) in enumerate(losses):
            x = left + plot_width * index / max(len(losses) - 1, 1)
            y = top + plot_height * (maximum - loss) / loss_range
            points.append((x, y))
        if len(points) == 1:
            (x, y) = points[0]
            draw.ellipse((x - 3, y - 3, x + 3, y + 3), fill='#1769aa')
        else:
            draw.line(points, fill='#1769aa', width=2)
        draw.text((left, 25), 'Sigma-weighted selected-step OPSD with SD3 velocity reference', fill='black')
        draw.text((left, height - 45), f'Batch step: 0 - {steps[-1]}', fill='black')
        draw.text((10, top), f'max {maximum:.6g}', fill='black')
        draw.text((10, height - bottom - 15), f'min {minimum:.6g}', fill='black')
        image.save(self.image_path)
LORA_TARGET_MODULES = 'a_to_qkv,b_to_qkv,a_to_out,b_to_out,ff_a.0,ff_a.2,ff_b.0,ff_b.2'
DEFAULT_CHECKPOINT_EVERY_N_TRAIN_STEPS = 50

def euler_step(latents, velocity, delta_sigma):
    return latents + delta_sigma.to(velocity.dtype) * velocity

def encode_velocity(bridge, current, next_value, sigma, next_sigma, domain):
    current_shared = bridge.encode(current, sigma, domain)
    next_shared = bridge.encode(next_value, next_sigma, domain)
    return (current_shared, (next_shared - current_shared) / (next_sigma - sigma))

def sample_supervised_steps(num_steps, count, device):
    if count < 1 or count > num_steps:
        raise ValueError('supervised step count must be in [1, num_steps]')
    distributed = torch.distributed.is_available() and torch.distributed.is_initialized()
    if not distributed or torch.distributed.get_rank() == 0:
        step_ids = torch.randperm(num_steps, device=device)[:count]
    else:
        step_ids = torch.empty(count, dtype=torch.long, device=device)
    if distributed:
        torch.distributed.broadcast(step_ids, src=0)
    return step_ids.sort().values.tolist()

def validate_supervised_step_ids(num_steps, step_ids):
    step_ids = tuple(step_ids)
    if not step_ids:
        raise ValueError('--supervised_step_ids must not be empty')
    if len(set(step_ids)) != len(step_ids):
        raise ValueError('--supervised_step_ids must not contain duplicates')
    if any((step_id < 0 or step_id >= num_steps for step_id in step_ids)):
        raise ValueError('--supervised_step_ids must be in [0, sampling_steps)')
    return tuple(sorted(step_ids))

@contextmanager
def lora_disabled(model):
    layers = [module for module in model.modules() if isinstance(module, BaseTunerLayer)]
    for layer in layers:
        layer.enable_adapters(False)
    try:
        yield
    finally:
        for layer in layers:
            layer.enable_adapters(True)

class SD3OPD(LightningModelForT2ILoRA):

    def __init__(self, torch_dtype, qwen_path, sd3_path, prompt_embedding_dir, bridge_checkpoint, height, width, learning_rate=0.0001, sampling_steps=20, supervised_steps_per_image=1, supervised_step_ids=None, checkpoint_every_n_train_steps=DEFAULT_CHECKPOINT_EVERY_N_TRAIN_STEPS, reference_weight=0.1, use_gradient_checkpointing=False, lora_rank=16, lora_alpha=16, lora_target_modules=LORA_TARGET_MODULES, init_lora_weights='kaiming', pretrained_lora_path=None):
        super().__init__(learning_rate=learning_rate, use_gradient_checkpointing=use_gradient_checkpointing)
        if sampling_steps < 1:
            raise ValueError('--sampling_steps must be at least 1')
        if supervised_step_ids is None:
            if not 1 <= supervised_steps_per_image <= sampling_steps:
                raise ValueError('--supervised_steps_per_image must be in [1, sampling_steps]')
            fixed_supervised_step_ids = None
        else:
            fixed_supervised_step_ids = validate_supervised_step_ids(sampling_steps, supervised_step_ids)
        if reference_weight < 0:
            raise ValueError('--reference_weight must be non-negative')
        if checkpoint_every_n_train_steps < 1:
            raise ValueError('--checkpoint_every_n_train_steps must be at least 1')
        if height % 8 != 0 or width % 8 != 0:
            raise ValueError('--height and --width must be divisible by 8')
        self.backbones = SD3QwenBackbones(qwen_path=qwen_path, sd3_path=sd3_path, prompt_embedding_dir=prompt_embedding_dir, bridge_checkpoint=bridge_checkpoint, torch_dtype=torch_dtype, sampling_steps=sampling_steps, device=self.device)
        first_sigma = float(self.backbones.qwen_pipe.scheduler.sigmas[0])
        if abs(first_sigma - 1.0) > 1e-06:
            raise ValueError(f'The first scheduler sigma must be 1, got {first_sigma}')
        self.add_lora_to_model(self.backbones.sd3_dit, lora_rank=lora_rank, lora_alpha=lora_alpha, lora_target_modules=lora_target_modules, init_lora_weights=init_lora_weights, pretrained_lora_path=pretrained_lora_path)
        self.height = height
        self.width = width
        self.supervised_step_ids = fixed_supervised_step_ids
        self.supervised_steps_per_image = len(fixed_supervised_step_ids) if fixed_supervised_step_ids is not None else supervised_steps_per_image
        self.reference_weight = reference_weight
        self.checkpoint_every_n_train_steps = checkpoint_every_n_train_steps
        self.loss_curve = None

    def on_train_start(self):
        if self.trainer.is_global_zero:
            self.loss_curve = LossCurve(self.logger.log_dir)

    def training_step(self, batch, batch_idx):
        (text, image) = (batch['text'], batch['image'])
        if image.shape[0] != 1:
            raise ValueError('SD3/Qwen prompt caches require batch_size=1')
        (height, width) = (self.height, self.width)
        self.backbones.qwen_pipe.eval()
        self.backbones.bridge.eval()
        self.backbones.sd3_dit.train()
        prompts = self.backbones.encode_prompts(text, self.device)
        initial_noise = torch.randn((1, 16, height // 8, width // 8), device=self.device, dtype=self.backbones.torch_dtype)
        sd3_latents = initial_noise.clone()
        qwen_initial_latents = initial_noise.clone()
        scheduler = self.backbones.qwen_pipe.scheduler
        if self.supervised_step_ids is None:
            supervised_step_ids = sample_supervised_steps(len(scheduler.timesteps), self.supervised_steps_per_image, self.device)
        else:
            supervised_step_ids = list(self.supervised_step_ids)
        supervised_step_set = set(supervised_step_ids)
        distillation_losses = []
        shared_velocity_mses = []
        reference_losses = []
        sd3_velocity_norms = []
        qwen_velocity_norms = []
        velocity_cosines = []
        reference_velocity_cosines = []
        shared_state_gaps = []
        supervised_sigmas = []
        for (step_id, timestep_cpu) in enumerate(scheduler.timesteps):
            sigma = scheduler.sigmas[step_id].to(self.device, torch.float32)
            if step_id + 1 < len(scheduler.sigmas):
                next_sigma = scheduler.sigmas[step_id + 1].to(self.device, torch.float32)
            else:
                next_sigma = torch.zeros((), device=self.device, dtype=torch.float32)
            delta_sigma = next_sigma - sigma
            if float(delta_sigma) == 0.0:
                raise ValueError(f'Zero sigma step at index {step_id}')
            timestep = timestep_cpu.reshape(1).to(self.device)
            sd3_latents = sd3_latents.detach()
            if step_id not in supervised_step_set:
                with torch.no_grad():
                    sd3_velocity = self.backbones.forward_sd3(sd3_latents, timestep, prompts['sd3'], False)
                    sd3_latents = euler_step(sd3_latents, sd3_velocity, delta_sigma)
                continue
            with torch.no_grad():
                sd3_shared = self.backbones.bridge.encode(sd3_latents, sigma, 'sd3')
                if step_id == 0:
                    qwen_latents = qwen_initial_latents
                else:
                    qwen_latents = self.backbones.bridge.decode(sd3_shared, sigma, 'qwen').to(self.backbones.torch_dtype)
                qwen_velocity = self.backbones.forward_qwen(qwen_latents, timestep, prompts['qwen'], height, width)
                qwen_next = euler_step(qwen_latents, qwen_velocity, delta_sigma)
                (qwen_shared, qwen_shared_velocity) = encode_velocity(self.backbones.bridge, qwen_latents, qwen_next, sigma, next_sigma, 'qwen')
                with lora_disabled(self.backbones.sd3_dit):
                    reference_velocity = self.backbones.forward_sd3(sd3_latents, timestep, prompts['sd3'], False)
            sd3_velocity = self.backbones.forward_sd3(sd3_latents, timestep, prompts['sd3'], self.use_gradient_checkpointing)
            sd3_next = euler_step(sd3_latents, sd3_velocity, delta_sigma)
            sd3_shared_next = self.backbones.bridge.encode(sd3_next, next_sigma, 'sd3')
            sd3_shared_velocity = (sd3_shared_next - sd3_shared) / delta_sigma
            shared_velocity_mse = F.mse_loss(sd3_shared_velocity.float(), qwen_shared_velocity.float())
            shared_velocity_mses.append(shared_velocity_mse)
            distillation_losses.append(sigma.square() * shared_velocity_mse)
            reference_losses.append(F.mse_loss(sd3_velocity.float(), reference_velocity.float()))
            sd3_velocity_norms.append(sd3_shared_velocity.float().square().mean().sqrt())
            qwen_velocity_norms.append(qwen_shared_velocity.float().square().mean().sqrt())
            velocity_cosines.append(F.cosine_similarity(sd3_shared_velocity.float().flatten(1), qwen_shared_velocity.float().flatten(1)).mean())
            reference_velocity_cosines.append(F.cosine_similarity(sd3_velocity.float().flatten(1), reference_velocity.float().flatten(1)).mean())
            shared_state_gaps.append(F.mse_loss(sd3_shared.float(), qwen_shared.float()))
            supervised_sigmas.append(sigma)
            sd3_latents = sd3_next
        distillation_loss = torch.stack(distillation_losses).mean()
        shared_velocity_mse = torch.stack(shared_velocity_mses).mean()
        reference_loss = torch.stack(reference_losses).mean()
        loss = distillation_loss + self.reference_weight * reference_loss
        self.log('train_loss', loss, prog_bar=True)
        self.log('distillation_loss', distillation_loss, prog_bar=True)
        self.log('shared_velocity_mse', shared_velocity_mse)
        self.log('reference_velocity_loss', reference_loss, prog_bar=True)
        self.log('supervised_step_id', sum(supervised_step_ids) / len(supervised_step_ids))
        self.log('supervised_sigma', torch.stack(supervised_sigmas).mean())
        self.log('supervised_steps_per_image', self.supervised_steps_per_image)
        self.log('sd3_shared_velocity_norm', torch.stack(sd3_velocity_norms).mean())
        self.log('qwen_shared_velocity_norm', torch.stack(qwen_velocity_norms).mean())
        self.log('shared_velocity_cosine', torch.stack(velocity_cosines).mean())
        self.log('reference_velocity_cosine', torch.stack(reference_velocity_cosines).mean())
        self.log('shared_state_gap', torch.stack(shared_state_gaps).mean())
        self.log('initial_native_noise_gap', F.mse_loss(initial_noise, qwen_initial_latents))
        if self.trainer.is_global_zero:
            self.loss_curve.append(supervised_step_ids, [float(sigma) for sigma in supervised_sigmas], float(loss.detach()), float(distillation_loss.detach()), float(shared_velocity_mse.detach()), float(reference_loss.detach()))
        return loss

    def on_train_epoch_end(self):
        if self.trainer.is_global_zero:
            self.loss_curve.save_image()

    def on_train_end(self):
        if self.trainer.is_global_zero:
            self.loss_curve.save_image()

    def configure_callbacks(self):
        return [ModelCheckpoint(filename='step-{step:06d}', auto_insert_metric_name=False, save_top_k=-1, every_n_train_steps=self.checkpoint_every_n_train_steps, save_on_train_epoch_end=False, save_on_exception=True), ModelCheckpoint(filename='epoch-{epoch:03d}-step-{step:06d}', auto_insert_metric_name=False, save_top_k=-1, every_n_epochs=1, save_on_train_epoch_end=True)]

    def configure_optimizers(self):
        parameters = [parameter for parameter in self.backbones.sd3_dit.parameters() if parameter.requires_grad]
        if not parameters:
            raise ValueError('SD3 has no trainable LoRA parameters')
        return torch.optim.AdamW(parameters, lr=self.learning_rate)

    def on_save_checkpoint(self, checkpoint):
        trainable = {name for (name, parameter) in self.backbones.sd3_dit.named_parameters() if parameter.requires_grad}
        checkpoint.clear()
        checkpoint.update({name: value for (name, value) in self.backbones.sd3_dit.state_dict().items() if name in trainable})

def parse_args(argv=None):
    return parse_training_args('opd', add_general_parsers, argv)


def run_opd(argv=None):
    args = parse_args(argv)
    if args.prepare_prompt_cache:
        from .prompt_cache import prepare_prompt_cache
        prepare_prompt_cache(args.dataset_path, args.prompt_embedding_dir, args.qwen_path, args.sd3_path, precision=args.precision)
    model = SD3OPD(torch_dtype=resolve_torch_dtype(args.precision), qwen_path=args.qwen_path, sd3_path=args.sd3_path, prompt_embedding_dir=args.prompt_embedding_dir, bridge_checkpoint=args.bridge_checkpoint, height=args.height, width=args.width, learning_rate=args.learning_rate, sampling_steps=args.sampling_steps, supervised_steps_per_image=args.supervised_steps_per_image, supervised_step_ids=args.supervised_step_ids, checkpoint_every_n_train_steps=args.checkpoint_every_n_train_steps, reference_weight=args.reference_weight, use_gradient_checkpointing=args.use_gradient_checkpointing, lora_rank=args.lora_rank, lora_alpha=args.lora_alpha, lora_target_modules=LORA_TARGET_MODULES, init_lora_weights=args.init_lora_weights, pretrained_lora_path=args.pretrained_lora_path)
    launch_training_task(model, args)

def main(argv=None):
    run_opd(argv)


if __name__ == "__main__":
    main()
