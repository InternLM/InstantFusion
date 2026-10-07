import json
import math
from pathlib import Path
import torch
from tqdm import tqdm
from diffsynth import ModelManager
from diffsynth.models.model_manager import load_model_from_single_file
from diffsynth.models.sd3_dit import SD3DiT
from diffsynth.models.sd3_vae_decoder import SD3VAEDecoder
from diffsynth.models.utils import load_state_dict
from diffsynth.schedulers import FlowMatchScheduler
from .bridge import build_sd3_bridge, load_bridge_checkpoint
from .model_utils import expand_qwen_image_component_path, resolve_torch_dtype
from .prompt_cache import _prepare_qwen, _prepare_sd3, load_prompt_embedding_cache
from .qwen import QwenPipeline
from .cli import parse_inference_args


def parse_args(argv=None):
    return parse_inference_args(argv)


def load_models(args, dtype):
    target = args.direction.split('_to_')[1]
    manager = ModelManager(torch_dtype=dtype, device=args.device)
    components = ['transformer'] + (['vae'] if target == 'qwen' or args.save_switch_clean else [])
    manager.load_models([expand_qwen_image_component_path(Path(args.qwen_path) / name) for name in components])
    qwen = QwenPipeline(device=args.device, torch_dtype=dtype, turbo_exponential_shift_mu=args.exponential_shift_mu)
    qwen.dit = manager.fetch_model('qwen_image_dit')
    if 'vae' in components:
        qwen.vae = manager.fetch_model('qwen_image_vae')
    del manager
    names, classes = ['sd3_dit'], [SD3DiT]
    if target == 'sd3' or args.save_switch_clean:
        names.append('sd3_vae_decoder')
        classes.append(SD3VAEDecoder)
    state = load_state_dict(str(args.sd3_path))
    loaded_names, loaded_models = load_model_from_single_file(state, names, classes, 'civitai', dtype, args.device)
    del state
    sd3 = dict(zip(loaded_names, loaded_models))
    if qwen.dit is None or ('vae' in components and qwen.vae is None) or any(sd3.get(name) is None for name in names):
        raise ValueError('Required denoiser or VAE could not be loaded')
    bridge = build_sd3_bridge(shared_channels=args.shared_channels, hidden_channels=args.hidden_channels,
                              num_res_blocks=args.num_res_blocks, sigma_embedding_dim=args.sigma_embedding_dim)
    load_bridge_checkpoint(bridge, args.lae_checkpoint)
    bridge = bridge.to(device=args.device, dtype=torch.float32)
    for model in (qwen, bridge, *sd3.values()):
        model.requires_grad_(False).eval()
    print(f'Denoisers and LAE resident on {args.device}; CPU offload disabled', flush=True)
    return qwen, sd3, bridge


@torch.inference_mode()
def generate(initial_latents, transitions, source_velocity, target_velocity, bridge, source, target,
             threshold=0.12, min_step=2, max_step=None, switch_step=None, progress=False):
    transitions = list(transitions)
    if switch_step is None:
        max_step = len(transitions) - 1 if max_step is None else max_step
        if not math.isfinite(threshold) or threshold <= 0 or not 2 <= min_step <= max_step < len(transitions):
            raise ValueError('Invalid adaptive threshold or switch window')
    elif not 1 <= switch_step < len(transitions):
        raise ValueError('Fixed switch must leave at least one target step')
    latents = initial_latents
    previous_clean = None
    record = None
    trace = []
    iterator = tqdm(transitions, desc=f'{source} -> {target}', disable=not progress)
    for completed_step, (sigma_value, next_sigma_value) in enumerate(iterator, start=1):
        sigma = torch.as_tensor(sigma_value, device=latents.device, dtype=torch.float32).reshape(())
        next_sigma = torch.as_tensor(next_sigma_value, device=latents.device, dtype=torch.float32).reshape(())
        velocity = (source_velocity if record is None else target_velocity)(latents, sigma)
        native_next = latents + velocity * (next_sigma - sigma).to(velocity.dtype)
        switch_now = False
        if record is None and (switch_step is None or completed_step == switch_step):
            predicted_clean = latents - velocity * sigma.to(velocity.dtype)
            zero = sigma.new_zeros(())
            clean_shared = bridge.encode(predicted_clean, zero, source).float()
            change = None
            if switch_step is None and previous_clean is not None:
                change = float((torch.linalg.vector_norm(clean_shared - previous_clean) /
                                torch.linalg.vector_norm(previous_clean).clamp_min(1e-8)).item())
            previous_clean = clean_shared.detach()
            trace.append({'step': completed_step, 'sigma': float(sigma.item()), 'relative_clean_change': change})
            if switch_step is not None:
                switch_now, reason = completed_step == switch_step, 'fixed_step'
            elif completed_step >= min_step and change is not None and change < threshold:
                switch_now, reason = True, 'threshold'
            else:
                switch_now, reason = completed_step >= max_step, 'max_step'
            if switch_now:
                record = {'switch_step': completed_step, 'switch_sigma': float(sigma.item()),
                          'switch_next_sigma': float(next_sigma.item()), 'relative_clean_change': change,
                          'switch_reason': reason, 'source_clean_latent': predicted_clean.detach(),
                          'target_clean_latent': bridge.decode(clean_shared, zero, target).detach()}
                print(f'Switch {source} -> {target} after step {completed_step}: {reason}, relative L2={change}', flush=True)
        latents = bridge.translate(native_next, next_sigma, source, target).to(native_next.dtype) if switch_now else native_next
    if record is None:
        raise RuntimeError('Switch did not trigger')
    return {**record, 'final_latent': latents, 'adaptive_trace': trace}


def guided_velocity(domain, qwen, sd3, positive, negative, args):
    scale = args.qwen_cfg if domain == 'qwen' else args.sd3_cfg
    def forward(latents, sigma):
        timestep = (sigma * 1000.0).reshape(1)
        def predict(embedding):
            if domain == 'qwen':
                return qwen.forward_dit(latents, timestep=timestep, prompt_emb=embedding, height=args.height, width=args.width)
            return sd3['sd3_dit'](latents, timestep=timestep, **embedding, use_gradient_checkpointing=False)
        prediction = predict(positive)
        if scale == 1.0:
            return prediction
        unconditional = predict(negative)
        return unconditional + scale * (prediction - unconditional)
    return forward


@torch.inference_mode()
def save_image(domain, latents, path, qwen, sd3, dtype):
    decoded = qwen.vae.decode(latents.to(dtype=dtype), tiled=False) if domain == 'qwen' else sd3['sd3_vae_decoder'](latents.to(dtype=dtype), tiled=False)
    qwen.vae_output_to_image(decoded.detach()).save(path)


def output_files(args, index, seed):
    root = Path(args.output_dir)
    stem = f'sample_{index:04d}_seed{seed}'
    files = {'final': root / f'{stem}.png', 'metadata': root / f'{stem}.json'}
    if args.save_switch_clean:
        files.update({'source_clean': root / f'{stem}_source_clean.png', 'target_clean': root / f'{stem}_target_clean.png'})
    return files


@torch.inference_mode()
def main(argv=None):
    args = parse_args(argv)
    device = torch.device(args.device)
    if device.type == 'cuda' and not torch.cuda.is_available():
        raise RuntimeError('CUDA is unavailable; inference will not silently fall back to CPU')
    for name in ('qwen_path', 'sd3_path', 'lae_checkpoint'):
        if not Path(getattr(args, name)).exists():
            raise FileNotFoundError(f'Missing {name}: {getattr(args, name)}')
    samples = [prompt for prompt in args.prompt for _ in range(args.num_images)]
    jobs = [(prompt, args.seed + index) for index, prompt in enumerate(samples)]
    for index, (_, seed) in enumerate(jobs):
        if not args.overwrite:
            for path in output_files(args, index, seed).values():
                if path.exists():
                    raise FileExistsError(f'Refusing to overwrite {path}; use --overwrite explicitly')
    dtype = resolve_torch_dtype(args.precision)
    cache = Path(args.prompt_embedding_dir) if args.prompt_embedding_dir else Path(args.output_dir) / 'prompt_cache'
    prompts = list(dict.fromkeys(prompt for prompt, _ in jobs))
    print('Preparing missing prompt embeddings before loading the denoisers', flush=True)
    qwen_prompts = list(dict.fromkeys(prompts + ([args.negative_prompt] if args.qwen_cfg != 1.0 else [])))
    sd3_prompts = list(dict.fromkeys(prompts + ([args.negative_prompt] if args.sd3_cfg != 1.0 else [])))
    _prepare_qwen(qwen_prompts, args.qwen_path, cache, dtype, args.device, 1)
    _prepare_sd3(sd3_prompts, args.sd3_path, cache, dtype, args.device, args.t5_sequence_length)
    qwen, sd3, bridge = load_models(args, dtype)
    scheduler = FlowMatchScheduler(sigma_min=0.0, sigma_max=1.0, extra_one_step=True, exponential_shift=True,
                                  exponential_shift_mu=args.exponential_shift_mu, shift_terminal=None)
    scheduler.set_timesteps(args.steps, exponential_shift_mu=args.exponential_shift_mu)
    sigmas = scheduler.sigmas.to(args.device, dtype=torch.float32)
    transitions = list(zip(sigmas, torch.cat([sigmas[1:], sigmas.new_zeros(1)])))
    source, target = args.direction.split('_to_')
    def embedding(domain, text):
        return load_prompt_embedding_cache(cache, domain, text, args.device, dtype, args.t5_sequence_length)
    negatives = {domain: embedding(domain, args.negative_prompt) if scale != 1.0 else None
                 for domain, scale in (('sd3', args.sd3_cfg), ('qwen', args.qwen_cfg))}
    Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    for index, (prompt, seed) in enumerate(jobs):
        print(f'Image {index + 1}/{len(jobs)}, seed={seed}: {prompt}', flush=True)
        velocities = {domain: guided_velocity(domain, qwen, sd3, embedding(domain, prompt), negatives[domain], args) for domain in ('sd3', 'qwen')}
        noise = qwen.generate_noise((1, 16, args.height // 8, args.width // 8), seed=seed, device=args.device, dtype=dtype)
        result = generate(noise, transitions, velocities[source], velocities[target], bridge, source, target,
                          args.adaptive_threshold, args.min_switch_step, args.max_switch_step, args.switch_step, progress=True)
        files = output_files(args, index, seed)
        save_image(target, result['final_latent'], files['final'], qwen, sd3, dtype)
        if args.save_switch_clean:
            save_image(source, result['source_clean_latent'], files['source_clean'], qwen, sd3, dtype)
            save_image(target, result['target_clean_latent'], files['target_clean'], qwen, sd3, dtype)
        metadata = {name: value for name, value in result.items() if not isinstance(value, torch.Tensor)}
        metadata.update({'prompt': prompt, 'seed': seed, 'source': source, 'target': target, 'config': vars(args),
                         'files': {name: str(path) for name, path in files.items()}})
        files['metadata'].write_text(json.dumps(metadata, indent=2, ensure_ascii=False) + '\n', encoding='utf-8')
        print(f'Saved {files["final"]}', flush=True)


if __name__ == '__main__':
    main()
