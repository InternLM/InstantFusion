import contextlib
import io
import json
import sys
import tempfile
import unittest
from pathlib import Path


sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'src'))
from instantfusion.cli import parse_inference_args, parse_training_args  # noqa: E402


def general_parser(parser):
    """The DiffSynth fields consumed by the compact training interface."""
    parser.add_argument('--dataset_path', required=True)
    parser.add_argument('--output_path', default='./')
    parser.add_argument('--height', type=int, default=1024)
    parser.add_argument('--width', type=int, default=1024)
    parser.add_argument('--max_epochs', type=int, default=1)
    parser.add_argument('--steps_per_epoch', type=int, default=500)
    parser.add_argument('--batch_size', type=int, default=1)
    parser.add_argument('--precision', default='16-mixed')
    parser.add_argument('--learning_rate', type=float, default=1e-4)
    parser.add_argument('--lora_rank', type=int, default=4)
    parser.add_argument('--lora_alpha', type=float, default=4.0)
    parser.add_argument('--init_lora_weights', default='kaiming')
    parser.add_argument('--accumulate_grad_batches', type=int, default=1)
    parser.add_argument('--use_gradient_checkpointing', action='store_true')
    parser.add_argument('--pretrained_lora_path')
    return parser


TRAIN_PATHS = [
    '--qwen-path', 'qwen', '--sd3-path', 'sd3',
    '--dataset-path', 'data', '--output-path', 'out',
]
INFER_PATHS = [
    '--qwen-path', 'qwen', '--sd3-path', 'sd3',
    '--lae-checkpoint', 'lae.ckpt', '--prompt', 'a lighthouse',
]


class CliTests(unittest.TestCase):
    def config_file(self, root, values):
        path = Path(root) / 'config.json'
        path.write_text(json.dumps(values), encoding='utf-8')
        return str(path)

    def test_lae_config_and_cli_precedence(self):
        with tempfile.TemporaryDirectory() as root:
            config = self.config_file(root, {
                'height': 768, 'width': 768, 'learning_rate': 2e-4,
                'sampling_steps': 30, 'velocity_weight': 0.08,
            })
            args = parse_training_args('lae', general_parser, TRAIN_PATHS + [
                '--config', config, '--size', '512x768', '--sampling-steps', '40',
                '--prepare-prompt-cache',
            ])
        self.assertEqual((args.height, args.width), (512, 768))
        self.assertEqual(args.sampling_steps, 40)
        self.assertEqual(args.learning_rate, 2e-4)
        self.assertEqual(args.velocity_weight, 0.08)
        self.assertTrue(args.prepare_prompt_cache)
        self.assertEqual(args.prompt_embedding_dir, './prompt_cache')

    def test_opd_defaults_and_random_step_override(self):
        base = TRAIN_PATHS + ['--bridge-checkpoint', 'lae.ckpt']
        args = parse_training_args('opd', general_parser, base)
        self.assertEqual(args.supervised_step_ids, (1, 2, 4, 8))
        self.assertEqual(args.lora_rank, 64)
        self.assertEqual(args.accumulate_grad_batches, 2)
        with tempfile.TemporaryDirectory() as root:
            config = self.config_file(root, {'supervised_steps_per_image': 3})
            args = parse_training_args('opd', general_parser, base + ['--config', config])
        self.assertIsNone(args.supervised_step_ids)
        self.assertEqual(args.supervised_steps_per_image, 3)

    def test_legacy_training_option_names_still_parse(self):
        args = parse_training_args('lae', general_parser, [
            '--qwen_path=qwen', '--sd3_path', 'sd3',
            '--dataset_path', 'data', '--output_path', 'out',
            '--max_epochs', '3', '--batch_size', '2',
        ])
        self.assertEqual(args.max_epochs, 3)
        self.assertEqual(args.batch_size, 2)

    def test_inference_defaults_and_advanced_config(self):
        args = parse_inference_args(INFER_PATHS)
        self.assertEqual(args.max_switch_step, 49)
        self.assertEqual((args.height, args.width), (512, 512))
        with tempfile.TemporaryDirectory() as root:
            config = self.config_file(root, {'sd3_cfg': 5.0, 'height': 768, 'save_switch_clean': True})
            args = parse_inference_args(INFER_PATHS + [
                '--config', config, '--size', '512x768', '--switch-step', '10',
            ])
        self.assertEqual((args.height, args.width), (512, 768))
        self.assertEqual(args.sd3_cfg, 5.0)
        self.assertTrue(args.save_switch_clean)
        self.assertEqual(args.switch_step, 10)

    def test_invalid_config_key_is_rejected(self):
        with tempfile.TemporaryDirectory() as root:
            config = self.config_file(root, {'nonexistent_setting': 1})
            with contextlib.redirect_stderr(io.StringIO()), self.assertRaises(SystemExit):
                parse_inference_args(INFER_PATHS + ['--config', config])


if __name__ == '__main__':
    unittest.main()
