import importlib.util
import unittest
from pathlib import Path
from types import SimpleNamespace

ROOT = Path(__file__).resolve().parent.parent
spec = importlib.util.spec_from_file_location("measure_candidates", ROOT / "scripts/measure_candidates.py")
mc = importlib.util.module_from_spec(spec)
spec.loader.exec_module(mc)


class RuntimesFilter(unittest.TestCase):
    """--runtimes measures on the named devices only; without it every device of each candidate runs."""

    def queued(self, runtimes, names):
        seen = []
        orig = mc.run_queue
        mc.run_queue = lambda jobs, logs: seen.extend(n for n, _ in jobs)
        try:
            a = SimpleNamespace(runtimes=runtimes, sessions="e01", policy="p.toml", data="data")
            mc.measure(a, names, ROOT / "build/none", Path("/tmp/edgegate-test-none"))
        finally:
            mc.run_queue = orig
        return sorted(seen)

    def test_hailo_only(self):
        self.assertEqual(self.queued(["hailo"], ["bgr", "no_norm", "w4"]), ["bgr_hailo", "no_norm_hailo", "w4_hailo"])

    def test_default_is_every_device(self):
        self.assertEqual(self.queued(None, ["bgr", "w4"]), ["bgr_hailo", "bgr_trt", "w4_hailo"])

    def test_reference_is_not_a_device_run(self):
        self.assertEqual(self.queued(["trt"], ["reference", "clean"]), ["clean_trt", "reference_onnx"])


if __name__ == "__main__":
    unittest.main()
