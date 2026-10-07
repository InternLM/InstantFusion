import math
import torch
from diffsynth.pipelines.base import BasePipeline
from diffsynth.pipelines.qwen_image import model_fn_qwen_image
from diffsynth.schedulers import FlowMatchScheduler

class QwenPipeline(BasePipeline):

    def __init__(self, device='cuda', torch_dtype=torch.bfloat16, qwen_tokenizer_path=None, turbo_exponential_shift_mu=math.log(2.5)):
        super().__init__(device=device, torch_dtype=torch_dtype, height_division_factor=16, width_division_factor=16)
        self.scheduler = FlowMatchScheduler(sigma_min=0.0, sigma_max=1.0, extra_one_step=True, exponential_shift=True, exponential_shift_mu=turbo_exponential_shift_mu, shift_terminal=None)
        self.qwen_tokenizer_path = qwen_tokenizer_path
        self.text_encoder = None
        self.dit = None
        self.vae = None
        self.tokenizer = None
        self.model_names = ['text_encoder', 'dit', 'vae']

    @staticmethod
    def _as_prompt_list(prompt):
        if isinstance(prompt, str):
            return [prompt]
        if isinstance(prompt, tuple):
            return list(prompt)
        return prompt

    @staticmethod
    def _extract_masked_hidden(hidden_states, mask):
        bool_mask = mask.bool()
        valid_lengths = bool_mask.sum(dim=1)
        selected = hidden_states[bool_mask]
        return torch.split(selected, valid_lengths.tolist(), dim=0)

    def encode_prompt(self, prompt):
        prompt = self._as_prompt_list(prompt)
        template = '<|im_start|>system\nDescribe the image by detailing the color, shape, size, texture, quantity, text, spatial relationships of the objects and background:<|im_end|>\n<|im_start|>user\n{}<|im_end|>\n<|im_start|>assistant\n'
        drop_idx = 34
        text = [template.format(item) for item in prompt]
        model_inputs = self.tokenizer(text, max_length=4096 + drop_idx, padding=True, truncation=True, return_tensors='pt').to(self.device)
        hidden_states = self.text_encoder(input_ids=model_inputs.input_ids, attention_mask=model_inputs.attention_mask, output_hidden_states=True)[-1]
        split_hidden_states = self._extract_masked_hidden(hidden_states, model_inputs.attention_mask)
        split_hidden_states = [item[drop_idx:] for item in split_hidden_states]
        attention_masks = [torch.ones(item.size(0), dtype=torch.long, device=item.device) for item in split_hidden_states]
        max_seq_len = max((item.size(0) for item in split_hidden_states))
        prompt_emb = torch.stack([torch.cat([item, item.new_zeros(max_seq_len - item.size(0), item.size(1))]) for item in split_hidden_states])
        prompt_emb_mask = torch.stack([torch.cat([item, item.new_zeros(max_seq_len - item.size(0))]) for item in attention_masks])
        return {'prompt_emb': prompt_emb.to(dtype=self.torch_dtype, device=self.device), 'prompt_emb_mask': prompt_emb_mask}

    def forward_dit(self, latents, timestep, prompt_emb, height, width):
        return model_fn_qwen_image(dit=self.dit, latents=latents, timestep=timestep, height=height, width=width, use_gradient_checkpointing=False, use_gradient_checkpointing_offload=False, checkpoint_determinism_check='none', **prompt_emb)
