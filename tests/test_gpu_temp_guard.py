import subprocess
import unittest
from importlib.util import find_spec
from unittest.mock import patch

TORCH_AVAILABLE = find_spec("torch") is not None
if TORCH_AVAILABLE:
    from src.utils.gpu_temp_guard import GPUTelemetryError, GPUTemperatureGuard


@unittest.skipUnless(TORCH_AVAILABLE, "torch is required for gpu temp guard tests")
class GpuTempGuardTests(unittest.TestCase):
    def test_wait_until_safe_pauses_and_resumes(self):
        guard = GPUTemperatureGuard(
            enabled=False,
            device="cpu",
            pause_threshold_c=90.0,
            resume_threshold_c=80.0,
            poll_interval_seconds=30.0,
        )
        guard._active = True  # Force active path for deterministic unit test.
        with (
            patch.object(guard, "read_temperature_c", side_effect=[91.0, 89.0, 79.0]),
            patch("src.utils.gpu_temp_guard.time.sleep", return_value=None),
        ):
            result = guard.wait_until_safe(context="unit-test")

        self.assertTrue(result.paused)
        self.assertEqual(result.checks_during_pause, 2)
        self.assertAlmostEqual(result.temp_c, 79.0)
        self.assertIn("thermal_pause", result.repair_action)

    def test_nvidia_smi_failure_is_fatal(self):
        guard = GPUTemperatureGuard(enabled=False, device="cpu")
        with patch(
            "src.utils.gpu_temp_guard.subprocess.run",
            side_effect=subprocess.CalledProcessError(1, ["nvidia-smi"], stderr="gpu lost"),
        ):
            with self.assertRaises(GPUTelemetryError):
                guard._read_temperature_from_nvidia_smi()


if __name__ == "__main__":
    unittest.main()
