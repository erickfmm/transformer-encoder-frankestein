import os
import tempfile
import unittest
from importlib.util import find_spec
from types import SimpleNamespace
from unittest.mock import Mock

TORCH_AVAILABLE = find_spec("torch") is not None
_IMPORTS_OK = False
if TORCH_AVAILABLE:
    try:
        from src.training.trainer import TitanTrainer, TrainingConfig
        _IMPORTS_OK = True
    except ImportError:
        pass


@unittest.skipUnless(_IMPORTS_OK, "torch and tqdm required for trainer tests")
class TrainerThermalOffloadTests(unittest.TestCase):
    def test_enforce_critical_temperature_uses_offload_path(self):
        trainer = TitanTrainer.__new__(TitanTrainer)
        trainer.training_config = TrainingConfig(switch_on_thermal=True)
        trainer.gpu_temp_guard = SimpleNamespace(
            is_active=True,
            read_temperature_c=Mock(return_value=96.0),
            critical_threshold_c=95.0,
            pause_threshold_c=90.0,
            wait_until_safe=Mock(),
        )
        trainer._thermal_offload_active = False
        trainer._force_cpu_only_after_gpu_error = False
        trainer._last_guard_temp_c = None
        trainer._handle_critical_thermal_offload = Mock(return_value="thermal_offload_cpu_mode")

        action = TitanTrainer._enforce_thermal_guard(trainer, epoch=0, batch_idx=4)

        self.assertEqual(action, "thermal_offload_cpu_mode")
        trainer._handle_critical_thermal_offload.assert_called_once_with(0, 4, 96.0)
        trainer.gpu_temp_guard.wait_until_safe.assert_not_called()

    def test_enforce_noncritical_temperature_uses_pause_only_path(self):
        wait_result = SimpleNamespace(temp_c=80.0, repair_action="thermal_pause_30s")
        trainer = TitanTrainer.__new__(TitanTrainer)
        trainer.training_config = TrainingConfig(switch_on_thermal=True)
        trainer.gpu_temp_guard = SimpleNamespace(
            is_active=True,
            read_temperature_c=Mock(return_value=91.0),
            critical_threshold_c=95.0,
            pause_threshold_c=90.0,
            wait_until_safe=Mock(return_value=wait_result),
        )
        trainer._thermal_offload_active = False
        trainer._force_cpu_only_after_gpu_error = False
        trainer._last_guard_temp_c = None
        trainer._handle_critical_thermal_offload = Mock(return_value="should_not_run")

        action = TitanTrainer._enforce_thermal_guard(trainer, epoch=1, batch_idx=2)

        self.assertEqual(action, "thermal_pause_30s")
        trainer._handle_critical_thermal_offload.assert_not_called()
        trainer.gpu_temp_guard.wait_until_safe.assert_called_once()
        self.assertAlmostEqual(trainer._last_guard_temp_c, 80.0)

    def test_critical_offload_switches_to_cpu_training_mode(self):
        trainer = TitanTrainer.__new__(TitanTrainer)
        trainer.training_config = TrainingConfig(switch_on_thermal=True)
        trainer._last_guard_temp_c = None
        trainer._thermal_offload_active = False
        trainer._thermal_offload_started_monotonic = None
        trainer._thermal_last_poll_monotonic = 0.0
        trainer._thermal_emergency_last_checkpoint = None
        trainer._force_cpu_only_after_gpu_error = False

        calls = []

        def _record(name, ret=None):
            def _inner(*args, **kwargs):
                calls.append(name)
                return ret

            return _inner

        trainer._save_thermal_emergency_checkpoint = _record("save", "/tmp/emergency.pt")
        trainer._offload_training_state_to_cpu = _record("offload")

        action = TitanTrainer._handle_critical_thermal_offload(
            trainer,
            epoch=2,
            batch_idx=1,
            temp_c=97.5,
        )

        self.assertEqual(calls[:2], ["save", "offload"])
        self.assertEqual(action, "thermal_offload_cpu_mode")
        self.assertTrue(trainer._thermal_offload_active)
        self.assertEqual(trainer._thermal_emergency_last_checkpoint, "/tmp/emergency.pt")
        self.assertAlmostEqual(trainer._last_guard_temp_c, 97.5)

    def test_monitor_offload_reloads_gpu_after_resume_temp(self):
        trainer = TitanTrainer.__new__(TitanTrainer)
        trainer.training_config = TrainingConfig(switch_on_thermal=True)
        trainer._thermal_offload_active = True
        trainer._thermal_offload_started_monotonic = 10.0
        trainer._thermal_last_poll_monotonic = 0.0
        trainer._thermal_emergency_last_checkpoint = "/tmp/emergency.pt"
        trainer._force_cpu_only_after_gpu_error = False
        trainer.gpu_temp_guard = SimpleNamespace(
            poll_interval_seconds=0.01,
            resume_threshold_c=80.0,
            read_temperature_c=Mock(return_value=79.0),
        )
        trainer._last_guard_temp_c = None
        trainer._reload_training_state_to_device = Mock()

        from unittest.mock import patch

        with patch("src.training.trainer.time.perf_counter", return_value=20.0):
            action = TitanTrainer._monitor_thermal_offload_and_maybe_reload(
                trainer,
                epoch=0,
                batch_idx=0,
            )

        self.assertEqual(action, "thermal_onload_gpu_10s")
        trainer._reload_training_state_to_device.assert_called_once()
        self.assertFalse(trainer._thermal_offload_active)

    def test_thermal_emergency_retention_matches_rolling_limit(self):
        trainer = TitanTrainer.__new__(TitanTrainer)
        trainer.training_config = TrainingConfig(max_rolling_checkpoints=2)
        trainer.thermal_emergency_checkpoints = []

        with tempfile.TemporaryDirectory() as tmpdir:
            paths = []
            for idx in range(3):
                path = os.path.join(tmpdir, f"thermal_{idx}.pt")
                with open(path, "w", encoding="utf-8") as handle:
                    handle.write("checkpoint")
                paths.append(path)

            TitanTrainer._register_thermal_emergency_checkpoint(trainer, paths[0])
            TitanTrainer._register_thermal_emergency_checkpoint(trainer, paths[1])
            TitanTrainer._register_thermal_emergency_checkpoint(trainer, paths[2])

            self.assertFalse(os.path.exists(paths[0]))
            self.assertEqual(trainer.thermal_emergency_checkpoints, [paths[1], paths[2]])

    def test_cuda_error_forces_cpu_only_when_switch_enabled(self):
        trainer = TitanTrainer.__new__(TitanTrainer)
        trainer.training_config = TrainingConfig(switch_on_thermal=True)
        trainer._force_cpu_only_after_gpu_error = False
        trainer._thermal_offload_active = False
        trainer._thermal_offload_started_monotonic = None
        trainer._thermal_last_poll_monotonic = 0.0
        trainer._thermal_emergency_last_checkpoint = None
        trainer.device = "cuda"
        trainer.global_step = 7
        trainer._register_thermal_emergency_checkpoint = Mock()
        trainer.save_checkpoint = Mock(return_value="/tmp/gpu_error.pt")
        trainer._offload_training_state_to_cpu = Mock()

        handled = TitanTrainer._switch_to_cpu_only_after_gpu_error(
            trainer,
            exc=RuntimeError("CUDA out of memory"),
            epoch=0,
            batch_idx=3,
        )

        self.assertTrue(handled)
        self.assertTrue(trainer._force_cpu_only_after_gpu_error)
        trainer._offload_training_state_to_cpu.assert_called_once()


if __name__ == "__main__":
    unittest.main()
