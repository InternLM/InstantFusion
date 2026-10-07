import inspect
import math
from functools import wraps
from pathlib import Path
import torch
from diffsynth import ModelManager
from diffsynth.models.model_manager import load_model_from_single_file
from diffsynth.models.sd3_dit import SD3DiT
from diffsynth.models.utils import load_state_dict
from .qwen import QwenPipeline
from .bridge import build_sd3_bridge as build_shared_latent_bridge, load_bridge_checkpoint
EXPONENTIAL_SHIFT_MU = math.log(3.0)
SHARED_CHANNELS = 32
BRIDGE_HIDDEN_CHANNELS = 64
BRIDGE_NUM_RES_BLOCKS = 3
BRIDGE_SIGMA_EMBEDDING_DIM = 128
SD3_T5_SEQUENCE_LENGTH = 512

def expand_qwen_image_component_path(path):
    path = Path(path)
    if not path.is_dir():
        return str(path)
    if path.name == 'transformer':
        files = sorted(path.glob('diffusion_pytorch_model*.safetensors'))
    elif path.name == 'text_encoder':
        files = sorted(path.glob('model*.safetensors'))
    elif path.name == 'vae':
        files = sorted(path.glob('diffusion_pytorch_model*.safetensors'))
    else:
        files = []
    return [str(file) for file in files] if files else str(path)

def resolve_torch_dtype(precision):
    if precision in ['32', 'fp32', 'float32']:
        return torch.float32
    if precision in ['bf16', 'bfloat16']:
        return torch.bfloat16
    if precision in ['16', '16-mixed', 'fp16', 'float16']:
        return torch.float16
    raise ValueError(f'Unsupported precision: {precision}')

def load_sd3_vae_encoder(checkpoint_path, torch_dtype, device):
    from diffsynth.models.model_manager import load_model_from_single_file
    from diffsynth.models.sd3_vae_encoder import SD3VAEEncoder
    from diffsynth.models.utils import load_state_dict
    state_dict = load_state_dict(checkpoint_path)
    (names, models) = load_model_from_single_file(state_dict, ['sd3_vae_encoder'], [SD3VAEEncoder], 'civitai', torch_dtype, device)
    del state_dict
    encoder = dict(zip(names, models)).get('sd3_vae_encoder')
    if encoder is None:
        raise ValueError(f'Could not load SD3 VAE encoder from {checkpoint_path}')
    return encoder

def load_sd3_dit(checkpoint_path, torch_dtype, device):
    state_dict = load_state_dict(str(checkpoint_path))
    (names, models) = load_model_from_single_file(state_dict, ['sd3_dit'], [SD3DiT], 'civitai', torch_dtype, device)
    del state_dict
    model = dict(zip(names, models)).get('sd3_dit')
    if model is None:
        raise ValueError(f'Could not load sd3_dit from {checkpoint_path}')
    return model

class SD3QwenBackbones(torch.nn.Module):

    def __init__(self, qwen_path, sd3_path, prompt_embedding_dir, bridge_checkpoint, torch_dtype, sampling_steps, device):
        super().__init__()
        qwen_path = Path(qwen_path)
        sd3_path = Path(sd3_path)
        prompt_embedding_dir = Path(prompt_embedding_dir)
        bridge_checkpoint = Path(bridge_checkpoint)
        if not qwen_path.is_dir():
            raise FileNotFoundError(f'Missing Qwen model: {qwen_path}')
        if not sd3_path.is_file():
            raise FileNotFoundError(f'Missing SD3 checkpoint: {sd3_path}')
        if not prompt_embedding_dir.is_dir():
            raise FileNotFoundError(f'Missing prompt embedding cache: {prompt_embedding_dir}')
        if not bridge_checkpoint.exists():
            raise FileNotFoundError(f'Missing shared-latent bridge checkpoint: {bridge_checkpoint}')
        manager = ModelManager(torch_dtype=torch_dtype, device=device)
        manager.load_models([expand_qwen_image_component_path(qwen_path / 'transformer')])
        self.qwen_pipe = QwenPipeline(device=device, torch_dtype=torch_dtype, qwen_tokenizer_path=str(qwen_path / 'tokenizer'), turbo_exponential_shift_mu=EXPONENTIAL_SHIFT_MU)
        self.qwen_pipe.dit = manager.fetch_model('qwen_image_dit')
        if self.qwen_pipe.dit is None:
            raise ValueError(f'Could not load Qwen DiT from {qwen_path}')
        self.qwen_pipe.scheduler.set_timesteps(sampling_steps, exponential_shift_mu=EXPONENTIAL_SHIFT_MU)
        self.sd3_dit = load_sd3_dit(sd3_path, torch_dtype, device)
        self.bridge = build_shared_latent_bridge(bridge_type='resblock', native_channels=16, shared_channels=SHARED_CHANNELS, hidden_channels=BRIDGE_HIDDEN_CHANNELS, num_res_blocks=BRIDGE_NUM_RES_BLOCKS, sigma_embedding_dim=BRIDGE_SIGMA_EMBEDDING_DIM)
        load_bridge_checkpoint(self.bridge, bridge_checkpoint)
        self.prompt_embedding_dir = prompt_embedding_dir
        self.sd3_t5_sequence_length = SD3_T5_SEQUENCE_LENGTH
        self.torch_dtype = torch_dtype
        self.qwen_pipe.requires_grad_(False).eval()
        self.sd3_dit.requires_grad_(False).eval()
        self.bridge.requires_grad_(False).eval()

    def encode_prompts(self, text, device):
        from .prompt_cache import single_prompt, load_prompt_embedding_cache
        prompt = single_prompt(text)
        return {domain: load_prompt_embedding_cache(self.prompt_embedding_dir, domain, prompt, device, self.torch_dtype, self.sd3_t5_sequence_length) for domain in ('sd3', 'qwen')}

    def forward_qwen(self, latents, timestep, prompt, height, width):
        self.qwen_pipe.device = latents.device
        return self.qwen_pipe.forward_dit(latents, timestep=timestep, prompt_emb=prompt, height=height, width=width)

    def forward_sd3(self, latents, timestep, prompt, use_gradient_checkpointing):
        return self.sd3_dit(latents, timestep=timestep, **prompt, use_gradient_checkpointing=use_gradient_checkpointing)

def convert_qwen_text_encoder_state_dict(_converter, state_dict):
    from transformers import Qwen2_5_VLModel
    nested = 'self.language_model' in inspect.getsource(Qwen2_5_VLModel.__init__)
    converted = {}
    for (name, value) in state_dict.items():
        if name.startswith('visual.') or name.startswith('model.visual.'):
            continue
        if name.startswith('model.language_model.') and (not nested):
            name = 'model.' + name[len('model.language_model.'):]
        elif name.startswith('model.') and nested and (not name.startswith('model.language_model.')):
            name = 'model.language_model.' + name[len('model.'):]
        converted[name] = value
    return converted

def install_qwen_encoder_compatibility():
    from diffsynth.models.qwen_image_text_encoder import QwenImageTextEncoder, QwenImageTextEncoderStateDictConverter
    if not getattr(QwenImageTextEncoder, '_instantfusion_text_only', False):
        original_init = QwenImageTextEncoder.__init__

        @wraps(original_init)
        def text_only_init(self, *args, **kwargs):
            original_init(self, *args, **kwargs)
            self.model.visual = None
        QwenImageTextEncoder.__init__ = text_only_init
        QwenImageTextEncoder._instantfusion_text_only = True
    QwenImageTextEncoderStateDictConverter.from_diffusers = convert_qwen_text_encoder_state_dict
