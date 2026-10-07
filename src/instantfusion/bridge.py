import math
from functools import partial
from pathlib import Path
import torch
import torch.nn.functional as F

def sigma_map(sigma, reference):
    sigma = torch.as_tensor(sigma, device=reference.device, dtype=reference.dtype).flatten()
    if sigma.numel() == 1:
        sigma = sigma.expand(reference.shape[0])
    if sigma.numel() != reference.shape[0]:
        raise ValueError('sigma must be scalar or have one value per batch item')
    return sigma.reshape(-1, 1, 1, 1).expand(reference.shape[0], 1, reference.shape[-2], reference.shape[-1])

class SigmaConv(torch.nn.Module):

    def __init__(self, input_channels, output_channels, hidden_channels):
        super().__init__()
        self.net = torch.nn.Sequential(torch.nn.Conv2d(input_channels + 1, hidden_channels, 3, padding=1), torch.nn.SiLU(), torch.nn.Conv2d(hidden_channels, output_channels, 3, padding=1))

    def forward(self, value, sigma):
        dtype = self.net[0].weight.dtype
        value = value.to(dtype=dtype)
        return self.net(torch.cat([value, sigma_map(sigma, value)], dim=1))

def sinusoidal_sigma_embedding(sigma, dim, reference):
    sigma = torch.as_tensor(sigma, device=reference.device, dtype=torch.float32).flatten()
    if sigma.numel() == 1:
        sigma = sigma.expand(reference.shape[0])
    if sigma.numel() != reference.shape[0]:
        raise ValueError('sigma must be scalar or have one value per batch item')
    half_dim = dim // 2
    frequencies = torch.exp(-math.log(10000.0) * torch.arange(half_dim, device=reference.device, dtype=torch.float32) / max(half_dim - 1, 1))
    arguments = sigma[:, None] * 1000.0 * frequencies[None]
    embedding = torch.cat([torch.cos(arguments), torch.sin(arguments)], dim=-1)
    if embedding.shape[-1] < dim:
        embedding = torch.nn.functional.pad(embedding, (0, dim - embedding.shape[-1]))
    return embedding.to(dtype=reference.dtype)

def group_count(channels, maximum=8):
    for groups in range(min(maximum, channels), 0, -1):
        if channels % groups == 0:
            return groups
    return 1

class SigmaConditionedResBlock(torch.nn.Module):

    def __init__(self, channels=64, sigma_embedding_dim=128):
        super().__init__()
        groups = group_count(channels)
        self.norm1 = torch.nn.GroupNorm(groups, channels)
        self.conv1 = torch.nn.Conv2d(channels, channels, 3, padding=1)
        self.norm2 = torch.nn.GroupNorm(groups, channels)
        self.conv2 = torch.nn.Conv2d(channels, channels, 3, padding=1)
        self.modulation = torch.nn.Sequential(torch.nn.SiLU(), torch.nn.Linear(sigma_embedding_dim, channels * 4))
        self.layer_scale = torch.nn.Parameter(torch.full((1, channels, 1, 1), 0.001))

    @staticmethod
    def modulate(value, shift, scale):
        return value * (1 + scale[:, :, None, None]) + shift[:, :, None, None]

    def forward(self, value, sigma_embedding):
        (shift1, scale1, shift2, scale2) = self.modulation(sigma_embedding).chunk(4, dim=1)
        hidden = self.modulate(self.norm1(value), shift1, scale1)
        hidden = self.conv1(torch.nn.functional.silu(hidden))
        hidden = self.modulate(self.norm2(hidden), shift2, scale2)
        hidden = self.conv2(torch.nn.functional.silu(hidden))
        return value + self.layer_scale * hidden

class ResidualSigmaMapper(torch.nn.Module):

    def __init__(self, input_channels, output_channels, hidden_channels=64, num_res_blocks=3, sigma_embedding_dim=128):
        super().__init__()
        self.input_channels = int(input_channels)
        self.output_channels = int(output_channels)
        self.input_proj = torch.nn.Conv2d(self.input_channels + 1, hidden_channels, 3, padding=1)
        self.sigma_embedding = torch.nn.Sequential(torch.nn.Linear(sigma_embedding_dim, sigma_embedding_dim), torch.nn.SiLU(), torch.nn.Linear(sigma_embedding_dim, sigma_embedding_dim))
        self.blocks = torch.nn.ModuleList([SigmaConditionedResBlock(hidden_channels, sigma_embedding_dim=sigma_embedding_dim) for _ in range(num_res_blocks)])
        self.output_norm = torch.nn.GroupNorm(group_count(hidden_channels), hidden_channels)
        self.output_proj = torch.nn.Conv2d(hidden_channels, self.output_channels, 3, padding=1)
        torch.nn.init.zeros_(self.output_proj.weight)
        torch.nn.init.zeros_(self.output_proj.bias)

    def fixed_skip(self, value):
        if self.output_channels == self.input_channels:
            return value
        if self.output_channels < self.input_channels:
            return value[:, :self.output_channels]
        padding = value.new_zeros(value.shape[0], self.output_channels - self.input_channels, value.shape[-2], value.shape[-1])
        return torch.cat([value, padding], dim=1)

    def forward(self, value, sigma):
        dtype = self.input_proj.weight.dtype
        value = value.to(dtype=dtype)
        embedding = sinusoidal_sigma_embedding(sigma, self.sigma_embedding[0].in_features, value)
        embedding = self.sigma_embedding(embedding)
        hidden = self.input_proj(torch.cat([value, sigma_map(sigma, value)], dim=1))
        for block in self.blocks:
            hidden = block(hidden, embedding)
        residual = self.output_proj(torch.nn.functional.silu(self.output_norm(hidden)))
        return self.fixed_skip(value) + residual

def normalize_checkpoint_key(name):
    for prefix in ('module.', 'model.', '_forward_module.'):
        while name.startswith(prefix):
            name = name[len(prefix):]
    return name

def bridge_state_dict_from_checkpoint(state_dict, bridge):
    valid_keys = set(bridge.state_dict())
    output = {}
    for (name, value) in state_dict.items():
        if not isinstance(value, torch.Tensor):
            continue
        name = normalize_checkpoint_key(name)
        for prefix in ('shared_latent_bridge.', 'bridge.'):
            if name.startswith(prefix):
                name = name[len(prefix):]
                break
        if name in valid_keys:
            output[name] = value.float()
    return output

def checkpoint_shards(path):
    path = Path(path)
    if path.is_file():
        return [path]
    files = []
    for pattern in ('*.safetensors', '*.bin', '*.pt', '*.pth', '*.ckpt'):
        files.extend(path.glob(pattern))
    return sorted(set(files))

def load_bridge_checkpoint(bridge, checkpoint_path):
    from diffsynth.models.utils import load_state_dict
    loaded = set()
    for shard in checkpoint_shards(checkpoint_path):
        state_dict = load_state_dict(str(shard))
        bridge_state = bridge_state_dict_from_checkpoint(state_dict, bridge)
        if bridge_state:
            bridge.load_state_dict(bridge_state, strict=False)
            loaded.update(bridge_state)
        del state_dict, bridge_state
    missing = set(bridge.state_dict()) - loaded
    if missing:
        raise ValueError('Bridge checkpoint is incomplete or still a partitioned DeepSpeed checkpoint; convert it with zero_to_fp32.py first. Missing: ' + ', '.join(sorted(missing)))
    print(f'Loaded {len(loaded)} shared-latent bridge tensors from {checkpoint_path}')

class PlainConvSigmaMapper(torch.nn.Module):

    def __init__(self, input_channels=16, output_channels=32, hidden_channels=64, num_res_blocks=3):
        super().__init__()
        depth = 2 * int(num_res_blocks) + 2
        if depth != 8:
            raise ValueError(f'This baseline is defined to match the current 3-ResBlock LAE depth (8 convolutions), got depth={depth}')
        layers = [torch.nn.Conv2d(input_channels + 1, hidden_channels, 3, padding=1), torch.nn.SiLU()]
        for _ in range(depth - 2):
            layers.extend([torch.nn.Conv2d(hidden_channels, hidden_channels, 3, padding=1), torch.nn.SiLU()])
        layers.append(torch.nn.Conv2d(hidden_channels, output_channels, 3, padding=1))
        self.net = torch.nn.Sequential(*layers)

    def forward(self, value, sigma):
        value = value.to(dtype=self.net[0].weight.dtype)
        return self.net(torch.cat([value, sigma_map(sigma, value)], dim=1))

class SharedLatentBridge(torch.nn.Module):

    def __init__(self, channels, shared_channels=32, hidden_channels=64, num_res_blocks=3,
                 sigma_embedding_dim=128, bridge_type="resblock", packed_domain=None):
        super().__init__()
        self.channels = dict(channels)
        self.packed_domain = packed_domain
        self.shared_channels = int(shared_channels)
        if bridge_type in ("resblock", "plain-conv") and shared_channels < max(self.channels.values()):
            raise ValueError("shared_channels must cover the native channel count")
        def mapper(source, target):
            if bridge_type == "resblock":
                return ResidualSigmaMapper(source, target, hidden_channels, num_res_blocks, sigma_embedding_dim)
            if bridge_type == "legacy":
                return SigmaConv(source, target, hidden_channels)
            if bridge_type == "plain-conv":
                return PlainConvSigmaMapper(source, target, hidden_channels, num_res_blocks)
            raise ValueError("Unknown bridge architecture: " + bridge_type)
        self.encoders = torch.nn.ModuleDict({domain: mapper(count, self.shared_channels) for domain, count in self.channels.items()})
        self.decoders = torch.nn.ModuleDict({domain: mapper(self.shared_channels, count) for domain, count in self.channels.items()})

    def check_domain(self, domain):
        if domain not in self.channels:
            raise ValueError("Unknown domain: " + domain)

    @staticmethod
    def unpack_packed(latents):
        if latents.ndim != 4 or latents.shape[1] != 128:
            raise ValueError("Expected packed [B,128,H/16,W/16] FLUX.2 latent")
        return F.pixel_shuffle(latents, 2)

    @staticmethod
    def pack_unpacked(latents):
        if latents.ndim != 4 or latents.shape[1] != 32:
            raise ValueError("Expected unpacked [B,32,H/8,W/8] FLUX.2 latent")
        return F.pixel_unshuffle(latents, 2)

    def sample_paired_noise(self, qwen_reference, packed_reference):
        if self.packed_domain is None:
            raise ValueError("Correlated 32-to-16 noise is only used by FLUX.2/Klein")
        unpacked = self.unpack_packed(packed_reference)
        if qwen_reference.shape[1] != 16 or qwen_reference.shape[0] != unpacked.shape[0] or qwen_reference.shape[-2:] != unpacked.shape[-2:]:
            raise ValueError("Packed/Qwen noise references must have matching batch/spatial shapes")
        noise = torch.randn_like(unpacked)
        qwen_noise = (noise[:, :16] + noise[:, 16:]) * 2.0 ** (-0.5)
        return {"qwen": qwen_noise.to(qwen_reference), self.packed_domain: self.pack_unpacked(noise)}

    def encode(self, value, sigma, domain):
        self.check_domain(domain)
        if domain == self.packed_domain:
            value = self.unpack_packed(value)
        return self.encoders[domain](value, sigma)

    def decode(self, value, sigma, domain):
        self.check_domain(domain)
        value = self.decoders[domain](value, sigma)
        return self.pack_unpacked(value) if domain == self.packed_domain else value

    def translate(self, value, sigma, source, target):
        return self.decode(self.encode(value, sigma, source), sigma, target)


def build_bridge(model, native_channels=16, shared_channels=32, hidden_channels=64,
                 num_res_blocks=3, sigma_embedding_dim=128, bridge_type="resblock"):
    domains = {
        "sd3-qwen": ("sd3", "qwen"), "flux1-qwen": ("flux1", "qwen"),
        "zimage-qwen": ("z_image", "qwen"), "flux2-klein-qwen": ("flux2_klein", "qwen"),
        "flux2-qwen": ("qwen", "flux2"), "flux1-sd3-qwen": ("flux1", "sd3", "qwen"),
        "plain-conv": ("flux1", "sd3", "qwen"),
    }
    if model not in domains:
        raise ValueError("Unknown LAE model: " + model)
    packed = "flux2_klein" if model == "flux2-klein-qwen" else "flux2" if model == "flux2-qwen" else None
    if model == "flux2-klein-qwen" and bridge_type != "resblock":
        raise ValueError("The Klein experiment uses a ResBlock LAE")
    channels = {domain: 32 if domain == packed else native_channels for domain in domains[model]}
    return SharedLatentBridge(channels, shared_channels, hidden_channels, num_res_blocks,
                             sigma_embedding_dim, "plain-conv" if model == "plain-conv" else bridge_type, packed)


build_sd3_bridge = partial(build_bridge, "sd3-qwen")
build_flux1_bridge = partial(build_bridge, "flux1-qwen")
build_zimage_bridge = partial(build_bridge, "zimage-qwen")
build_flux2_bridge = partial(build_bridge, "flux2-qwen")
KleinBridge = partial(build_bridge, "flux2-klein-qwen")
build_triple_bridge = partial(build_bridge, "flux1-sd3-qwen")
build_plain_bridge = partial(build_bridge, "plain-conv")
