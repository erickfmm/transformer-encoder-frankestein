import unittest
from types import SimpleNamespace
from unittest.mock import Mock, patch

from src import cli


class TrainCliGpuTempFlagTests(unittest.TestCase):
    def test_forward_gpu_temp_flags_to_training_main(self):
        mocked_train_main = Mock(return_value=0)
        fake_training_main_module = SimpleNamespace(main=mocked_train_main)
        with patch.dict("sys.modules", {"src.training.main": fake_training_main_module}):
            exit_code = cli.main(
                [
                    "train",
                    "--config-name",
                    "mini",
                    "--device",
                    "cuda",
                    "--gpu-temp-guard",
                    "--gpu-temp-pause-threshold-c",
                    "91",
                    "--gpu-temp-resume-threshold-c",
                    "80",
                    "--gpu-temp-critical-threshold-c",
                    "95",
                    "--gpu-temp-poll-interval-seconds",
                    "30",
                ]
            )
        self.assertEqual(exit_code, 0)

        forwarded_argv = mocked_train_main.call_args[0][0]
        self.assertIn("--gpu-temp-guard", forwarded_argv)
        self.assertIn("--gpu-temp-pause-threshold-c", forwarded_argv)
        self.assertIn("--gpu-temp-resume-threshold-c", forwarded_argv)
        self.assertIn("--gpu-temp-critical-threshold-c", forwarded_argv)
        self.assertIn("--gpu-temp-poll-interval-seconds", forwarded_argv)

    def test_forward_no_gpu_temp_guard_flag(self):
        mocked_train_main = Mock(return_value=0)
        fake_training_main_module = SimpleNamespace(main=mocked_train_main)
        with patch.dict("sys.modules", {"src.training.main": fake_training_main_module}):
            exit_code = cli.main(["train", "--no-gpu-temp-guard"])
        self.assertEqual(exit_code, 0)
        forwarded_argv = mocked_train_main.call_args[0][0]
        self.assertIn("--no-gpu-temp-guard", forwarded_argv)


if __name__ == "__main__":
    unittest.main()
