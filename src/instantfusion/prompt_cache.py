import csv
import gc
import hashlib
import os
from pathlib import Path
import torch
from .model_utils import expand_qwen_image_component_path as expand_component, resolve_torch_dtype as resolve_dtype, convert_qwen_text_encoder_state_dict, install_qwen_encoder_compatibility
from .qwen import QwenPipeline as QwenImagePipeline

def unique_prompts(root: str | Path) -> list[str]:
    metadata = Path(root) / 'train' / 'metadata.csv'
    with metadata.open(newline='', encoding='utf-8') as handle:
        prompts = [str(row['text']) for row in csv.DictReader(handle)]
    return list(dict.fromkeys(prompts))

def single_prompt(text):
    if isinstance(text, str):
        return text
    if len(text) != 1:
        raise ValueError('Precomputed prompt embeddings require batch_size=1.')
    return str(text[0])

def prompt_embedding_cache_path(cache_dir, component, text):
    digest = hashlib.sha256(str(text).encode('utf-8')).hexdigest()
    return Path(cache_dir) / component / digest[:2] / f'{digest}.pt'

def load_prompt_embedding_cache(cache_dir, component, text, device, dtype, sd3_t5_sequence_length):
    path = prompt_embedding_cache_path(cache_dir, component, text)
    value = torch.load(path, map_location='cpu', weights_only=True)
    if component == 'qwen':
        return {'prompt_emb': value['prompt_emb'].to(device=device, dtype=dtype), 'prompt_emb_mask': value['prompt_emb_mask'].to(device=device)}
    if component == 'sd3':
        cached_length = int(value['t5_sequence_length'])
        if cached_length != int(sd3_t5_sequence_length):
            raise ValueError(f'Cached SD3 sequence length {cached_length} does not match {sd3_t5_sequence_length}.')
        return {'prompt_emb': value['prompt_emb'].to(device=device, dtype=dtype), 'pooled_prompt_emb': value['pooled_prompt_emb'].to(device=device, dtype=dtype)}
    raise ValueError(f'Unknown prompt embedding component: {component}')

cache_path = prompt_embedding_cache_path

def _save(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix('.tmp')
    torch.save(value, temporary)
    os.replace(temporary, path)

def _prepare_qwen(prompts, qwen_path, output_path, dtype, device, batch_size):
    install_qwen_encoder_compatibility()
    missing = [prompt for prompt in prompts if not cache_path(output_path, 'qwen', prompt).is_file()]
    if not missing:
        return
    from diffsynth import ModelManager
    from diffsynth.models.qwen_image_text_encoder import QwenImageTextEncoderStateDictConverter
    manager = ModelManager(torch_dtype=dtype, device=device)
    original = QwenImageTextEncoderStateDictConverter.from_diffusers
    QwenImageTextEncoderStateDictConverter.from_diffusers = convert_qwen_text_encoder_state_dict
    try:
        manager.load_models([expand_component(Path(qwen_path) / 'text_encoder')])
    finally:
        QwenImageTextEncoderStateDictConverter.from_diffusers = original
    pipe = QwenImagePipeline(device=device, torch_dtype=dtype, qwen_tokenizer_path=str(Path(qwen_path) / 'tokenizer'))
    pipe.text_encoder = manager.fetch_model('qwen_image_text_encoder').to(device)
    from transformers import Qwen2Tokenizer
    pipe.tokenizer = Qwen2Tokenizer.from_pretrained(pipe.qwen_tokenizer_path)
    pipe.text_encoder.eval()
    with torch.inference_mode():
        for start in range(0, len(missing), batch_size):
            batch = missing[start:start + batch_size]
            encoded = pipe.encode_prompt(batch)
            for (index, prompt) in enumerate(batch):
                length = int(encoded['prompt_emb_mask'][index].sum())
                _save(cache_path(output_path, 'qwen', prompt), {'prompt_emb': encoded['prompt_emb'][index:index + 1, :length].cpu(), 'prompt_emb_mask': encoded['prompt_emb_mask'][index:index + 1, :length].cpu()})
    del pipe, manager
    gc.collect()
    torch.cuda.empty_cache()

def _prepare_sd3(prompts, sd3_path, output_path, dtype, device, t5_length):
    missing = [prompt for prompt in prompts if not cache_path(output_path, 'sd3', prompt).is_file()]
    if not missing:
        return
    from diffsynth.models.model_manager import load_model_from_single_file
    from diffsynth.models.sd3_text_encoder import SD3TextEncoder1, SD3TextEncoder2, SD3TextEncoder3
    from diffsynth.models.utils import load_state_dict
    from diffsynth.prompters import SD3Prompter
    names = ['sd3_text_encoder_1', 'sd3_text_encoder_2']
    classes = [SD3TextEncoder1, SD3TextEncoder2]
    state = load_state_dict(str(sd3_path))
    if any(('t5xxl' in key for key in state)):
        names.append('sd3_text_encoder_3')
        classes.append(SD3TextEncoder3)
    (loaded_names, loaded_models) = load_model_from_single_file(state, names, classes, 'civitai', dtype, device)
    models = dict(zip(loaded_names, loaded_models))
    encoder1 = models['sd3_text_encoder_1'].to(device).eval()
    encoder2 = models['sd3_text_encoder_2'].to(device).eval()
    encoder3 = models.get('sd3_text_encoder_3')
    if encoder3 is not None:
        encoder3 = encoder3.to(device).eval()
    prompter = SD3Prompter()
    prompter.fetch_models(encoder1, encoder2, encoder3)
    with torch.inference_mode():
        for prompt in missing:
            (prompt_emb, pooled_prompt_emb) = prompter.encode_prompt(prompt, device=device, positive=True, t5_sequence_length=t5_length)
            _save(cache_path(output_path, 'sd3', prompt), {'prompt_emb': prompt_emb.cpu(), 'pooled_prompt_emb': pooled_prompt_emb.cpu(), 't5_sequence_length': int(t5_length)})
    del prompter, models, encoder1, encoder2, encoder3
    gc.collect()
    torch.cuda.empty_cache()

def prepare_prompt_cache(dataset_path, output_path, qwen_path, sd3_path, precision='bf16', device='cuda', batch_size=4, t5_length=512):
    prompts = unique_prompts(dataset_path)
    dtype = resolve_dtype(precision)
    print(f'Preparing prompt cache for {len(prompts)} unique prompts')
    _prepare_qwen(prompts, qwen_path, output_path, dtype, device, batch_size)
    _prepare_sd3(prompts, sd3_path, output_path, dtype, device, t5_length)
