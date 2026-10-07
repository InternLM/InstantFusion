"""Small command-line interfaces for the training and inference entry points.

Routine inputs stay on the command line. Research-specific options can be put in
a JSON file using their Python argument names (underscores, not dashes).
"""

import argparse
import json
import math
import re
import sys
from pathlib import Path


LAE_DEFAULTS = {
    'sampling_steps': 50,
    'exponential_shift_mu': math.log(3.0),
    'timesteps_per_batch': 1,
    'shared_channels': 32,
    'hidden_channels': 64,
    'bridge_num_res_blocks': 3,
    'bridge_sigma_embedding_dim': 128,
    'reconstruction_weight': 1.0,
    'cross_reconstruction_weight': 1.0,
    'alignment_weight': 0.1,
    'velocity_weight': 0.05,
    'shared_norm_weight': 0.0001,
    'sd3_t5_sequence_length': 512,
}

OPD_DEFAULTS = {
    'sampling_steps': 20,
    'supervised_steps_per_image': 1,
    'supervised_step_ids': (1, 2, 4, 8),
    'checkpoint_every_n_train_steps': 50,
    'reference_weight': 0.1,
}

OPD_TRAINER_DEFAULTS = {
    'learning_rate': 1e-5,
    'lora_rank': 64,
    'lora_alpha': 64,
    'init_lora_weights': 'kaiming',
    'accumulate_grad_batches': 2,
}

INFERENCE_DEFAULTS = {
    'direction': 'sd3_to_qwen',
    'negative_prompt': '',
    'num_images': 1,
    'seed': 42,
    'height': 512,
    'width': 512,
    'steps': 50,
    'switch_step': None,
    'adaptive_threshold': 0.12,
    'min_switch_step': 2,
    'max_switch_step': None,
    'exponential_shift_mu': math.log(3.0),
    'sd3_cfg': 4.0,
    'qwen_cfg': 4.0,
    't5_sequence_length': 512,
    'shared_channels': 32,
    'hidden_channels': 64,
    'num_res_blocks': 3,
    'sigma_embedding_dim': 128,
    'precision': 'bf16',
    'device': 'cuda',
    'prompt_embedding_dir': None,
    'output_dir': 'outputs/inference',
    'save_switch_clean': False,
    'overwrite': False,
}

TRAIN_ALIASES = {
    '--qwen_path': '--qwen-path',
    '--sd3_path': '--sd3-path',
    '--dataset_path': '--dataset-path',
    '--output_path': '--output-path',
    '--prompt_embedding_dir': '--prompt-cache',
    '--bridge_checkpoint': '--bridge-checkpoint',
    '--max_epochs': '--epochs',
    '--steps_per_epoch': '--steps-per-epoch',
    '--batch_size': '--batch-size',
    '--sampling_steps': '--sampling-steps',
    '--supervised_step_ids': '--supervised-step-ids',
}


def _normalize_aliases(argv, aliases):
    tokens = sys.argv[1:] if argv is None else argv
    normalized = []
    for token in tokens:
        name, separator, value = token.partition('=')
        normalized.append(aliases.get(name, name) + separator + value)
    return normalized


def _read_config(path, parser):
    if path is None:
        return {}
    try:
        data = json.loads(Path(path).read_text(encoding='utf-8'))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        parser.error(f'Cannot read --config {path}: {exc}')
    if not isinstance(data, dict):
        parser.error('--config must contain a JSON object')
    return data


def _size(value):
    match = re.fullmatch(r'(\d+)(?:[xX](\d+))?', value)
    if not match:
        raise argparse.ArgumentTypeError('use HEIGHTxWIDTH, for example 512x768')
    height = int(match.group(1))
    width = int(match.group(2) or match.group(1))
    if height < 16 or width < 16 or height % 16 or width % 16:
        raise argparse.ArgumentTypeError('height and width must be positive multiples of 16')
    return height, width


def _step_ids(value):
    if isinstance(value, str):
        parts = value.split(',')
        try:
            ids = tuple(int(part.strip()) for part in parts)
        except ValueError as exc:
            raise argparse.ArgumentTypeError('step IDs must be comma-separated integers') from exc
    elif isinstance(value, list) and all(type(item) is int for item in value):
        ids = tuple(value)
    else:
        raise argparse.ArgumentTypeError('step IDs must be a comma-separated string or integer list')
    if not ids or len(ids) != len(set(ids)) or any(item < 0 for item in ids):
        raise argparse.ArgumentTypeError('step IDs must be unique, nonnegative integers')
    return ids


def _model_override(key, value, default, parser):
    if key == 'supervised_step_ids':
        if value is None:
            return None
        try:
            return _step_ids(value)
        except argparse.ArgumentTypeError as exc:
            parser.error(f'{key}: {exc}')
    expected = type(default)
    if expected is int and type(value) is not int:
        parser.error(f'{key} must be an integer')
    if expected is float and (type(value) not in (int, float) or not math.isfinite(value)):
        parser.error(f'{key} must be a finite number')
    if expected is str and not isinstance(value, str):
        parser.error(f'{key} must be a string')
    return value


def _add_generic_value(argv, action, value, parser):
    option = next((name for name in action.option_strings if name.startswith('--')), None)
    if option is None:
        parser.error(f'Unsupported trainer setting: {action.dest}')
    if isinstance(action, argparse._StoreTrueAction):
        if type(value) is not bool:
            parser.error(f'{action.dest} must be true or false')
        if value:
            argv.append(option)
    elif isinstance(action, argparse._StoreFalseAction):
        if type(value) is not bool:
            parser.error(f'{action.dest} must be true or false')
        if not value:
            argv.append(option)
    elif value is not None:
        if isinstance(value, (dict, list, bool)):
            parser.error(f'{action.dest} must be a scalar value')
        argv.extend((option, str(value)))


def parse_training_args(stage, general_parser_factory, argv=None):
    if stage not in ('lae', 'opd'):
        raise ValueError(f'Unknown training stage: {stage}')
    parser = argparse.ArgumentParser(description=f'Train the InstantFusion {stage.upper()} stage')
    parser.add_argument('--qwen-path', required=True)
    parser.add_argument('--sd3-path', required=True)
    parser.add_argument('--dataset-path', required=True)
    parser.add_argument('--output-path', required=True)
    parser.add_argument('--prompt-cache', dest='prompt_embedding_dir', default='./prompt_cache')
    if stage == 'opd':
        parser.add_argument('--bridge-checkpoint', required=True)
    parser.add_argument('--prepare-prompt-cache', action='store_true', help='Prepare missing prompt embeddings before training')
    parser.add_argument('--size', type=_size, metavar='HEIGHTxWIDTH', help='Image size; a single value means square')
    parser.add_argument('--epochs', dest='max_epochs', type=int)
    parser.add_argument('--steps-per-epoch', dest='steps_per_epoch', type=int)
    parser.add_argument('--batch-size', type=int)
    parser.add_argument('--precision', choices=('32', '16', '16-mixed', 'bf16'))
    parser.add_argument('--sampling-steps', type=int)
    if stage == 'opd':
        parser.add_argument('--supervised-step-ids', type=_step_ids)
    parser.add_argument('--config', help='JSON file for advanced trainer and model settings')
    public = parser.parse_args(_normalize_aliases(argv, TRAIN_ALIASES))
    config = _read_config(public.config, parser)
    model_defaults = LAE_DEFAULTS if stage == 'lae' else OPD_DEFAULTS

    generic = general_parser_factory(argparse.ArgumentParser(add_help=False))
    generic_actions = {action.dest: action for action in generic._actions}
    blocked = {'dataset_path', 'output_path', 'qwen_path', 'sd3_path', 'prompt_embedding_dir', 'bridge_checkpoint'}
    unknown = set(config) - set(model_defaults) - set(generic_actions)
    if unknown:
        parser.error(f'Unknown --config setting(s): {", ".join(sorted(unknown))}')
    disallowed = set(config) & blocked
    if disallowed:
        parser.error(f'Use CLI arguments for: {", ".join(sorted(disallowed))}')

    generic_argv = ['--dataset_path', public.dataset_path, '--output_path', public.output_path]
    if stage == 'opd':
        for key, value in OPD_TRAINER_DEFAULTS.items():
            _add_generic_value(generic_argv, generic_actions[key], value, parser)
    for key, value in config.items():
        if key in generic_actions:
            _add_generic_value(generic_argv, generic_actions[key], value, parser)
    if public.size is not None:
        height, width = public.size
        generic_argv.extend(('--height', str(height), '--width', str(width)))
    for key in ('max_epochs', 'steps_per_epoch', 'batch_size', 'precision'):
        value = getattr(public, key)
        if value is not None:
            _add_generic_value(generic_argv, generic_actions[key], value, parser)
    args = generic.parse_args(generic_argv)

    for key, value in model_defaults.items():
        setattr(args, key, value)
    if stage == 'opd' and 'supervised_steps_per_image' in config and config.get('supervised_step_ids') is not None:
        parser.error('Choose either supervised_steps_per_image or supervised_step_ids in --config')
    if stage == 'opd' and 'supervised_steps_per_image' in config and 'supervised_step_ids' not in config and public.supervised_step_ids is None:
        args.supervised_step_ids = None
    for key, value in config.items():
        if key in model_defaults:
            setattr(args, key, _model_override(key, value, model_defaults[key], parser))
    if public.sampling_steps is not None:
        args.sampling_steps = public.sampling_steps
    if stage == 'opd' and public.supervised_step_ids is not None:
        args.supervised_step_ids = public.supervised_step_ids
    for key in ('qwen_path', 'sd3_path', 'prompt_embedding_dir', 'prepare_prompt_cache'):
        setattr(args, key, getattr(public, key))
    if stage == 'opd':
        args.bridge_checkpoint = public.bridge_checkpoint
        if args.supervised_step_ids is not None and any(step >= args.sampling_steps for step in args.supervised_step_ids):
            parser.error('supervised_step_ids must be smaller than sampling_steps')
        if args.supervised_step_ids is None and not 1 <= args.supervised_steps_per_image <= args.sampling_steps:
            parser.error('supervised_steps_per_image must be between 1 and sampling_steps')
    if args.sampling_steps < 2:
        parser.error('sampling_steps must be at least 2')
    if args.max_epochs < 1 or args.steps_per_epoch < 1 or args.batch_size < 1:
        parser.error('epochs, steps per epoch, and batch size must be positive')
    return args


def _validate_inference_config(config, parser):
    unknown = set(config) - set(INFERENCE_DEFAULTS)
    if unknown:
        parser.error(f'Unknown --config setting(s): {", ".join(sorted(unknown))}')
    for key, value in config.items():
        default = INFERENCE_DEFAULTS[key]
        if key in ('switch_step', 'max_switch_step'):
            valid = value is None or type(value) is int
        elif key == 'prompt_embedding_dir':
            valid = value is None or isinstance(value, str)
        elif type(default) is float:
            valid = type(value) in (int, float) and math.isfinite(value)
        else:
            valid = type(value) is type(default)
        if not valid:
            parser.error(f'Invalid value for {key} in --config')


def parse_inference_args(argv=None):
    parser = argparse.ArgumentParser(description='Generate with SD3/Qwen model handoff')
    parser.add_argument('--qwen-path', required=True)
    parser.add_argument('--sd3-path', required=True)
    parser.add_argument('--lae-checkpoint', required=True)
    parser.add_argument('--prompt', action='append', required=True, help='May be repeated for multiple prompts')
    parser.add_argument('--direction', choices=('sd3_to_qwen', 'qwen_to_sd3'))
    parser.add_argument('--output-dir')
    parser.add_argument('--size', type=_size, metavar='HEIGHTxWIDTH')
    parser.add_argument('--steps', type=int)
    parser.add_argument('--seed', type=int)
    parser.add_argument('--num-images', type=int)
    parser.add_argument('--negative-prompt')
    parser.add_argument('--switch-step', type=int, help='Use a fixed handoff instead of adaptive switching')
    parser.add_argument('--adaptive-threshold', type=float)
    parser.add_argument('--overwrite', action='store_true')
    parser.add_argument('--config', help='JSON file for advanced inference settings')
    public = parser.parse_args(_normalize_aliases(argv, {'--bridge-checkpoint': '--lae-checkpoint'}))
    config = _read_config(public.config, parser)
    _validate_inference_config(config, parser)
    values = {**INFERENCE_DEFAULTS, **config}
    for key in ('direction', 'output_dir', 'steps', 'seed', 'num_images', 'negative_prompt', 'switch_step', 'adaptive_threshold'):
        value = getattr(public, key)
        if value is not None:
            values[key] = value
    if public.size is not None:
        values['height'], values['width'] = public.size
    if public.overwrite:
        values['overwrite'] = True
    values.update(qwen_path=public.qwen_path, sd3_path=public.sd3_path,
                  lae_checkpoint=public.lae_checkpoint, prompt=public.prompt)
    args = argparse.Namespace(**values)

    if args.direction not in ('sd3_to_qwen', 'qwen_to_sd3'):
        parser.error('direction must be sd3_to_qwen or qwen_to_sd3')
    if args.precision not in ('bf16', 'fp16', 'fp32'):
        parser.error('precision must be bf16, fp16, or fp32')
    if args.steps < 2 or args.num_images < 1:
        parser.error('steps must be at least 2 and num_images must be positive')
    if args.height < 16 or args.width < 16 or args.height % 16 or args.width % 16:
        parser.error('height and width must be positive multiples of 16')
    if not math.isfinite(args.exponential_shift_mu) or not all(math.isfinite(value) for value in (args.sd3_cfg, args.qwen_cfg)):
        parser.error('schedule shift and CFG scales must be finite')
    if args.t5_sequence_length < 1 or any(getattr(args, key) < 1 for key in ('shared_channels', 'hidden_channels', 'num_res_blocks', 'sigma_embedding_dim')):
        parser.error('text length and bridge dimensions must be positive')
    if args.switch_step is not None:
        if not 1 <= args.switch_step < args.steps:
            parser.error('switch_step must be in [1, steps - 1]')
    else:
        if not math.isfinite(args.adaptive_threshold) or args.adaptive_threshold <= 0:
            parser.error('adaptive_threshold must be finite and positive')
        args.max_switch_step = args.max_switch_step if args.max_switch_step is not None else args.steps - 1
        if not 2 <= args.min_switch_step <= args.max_switch_step < args.steps:
            parser.error('adaptive window must satisfy 2 <= min <= max < steps')
    return args
