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


class ReplayFromRecords(unittest.TestCase):
    """Without build/ (a replay of committed records), judge takes the hashes from the records."""

    def test_hashes_come_from_the_records(self):
        import json
        import tempfile
        from test_gate import NOW, policy, record
        with tempfile.TemporaryDirectory() as tmp:
            out = Path(tmp) / "records"
            for sub, rec in (("reference", record("onnx-gx10")), ("clean", record("trt")), ("clean", record("hailo8"))):
                (out / sub).mkdir(parents=True, exist_ok=True)
                (out / sub / f"{rec['backend']}.json").write_text(json.dumps(rec))
            rows = mc.judge(policy(), ["clean"], Path(tmp) / "no-build", out, NOW)
        by = {(r["candidate"], r["backend"]): r for r in rows}
        self.assertTrue(all(r["replay"] for r in rows))
        self.assertEqual(by[("clean", "trt")]["verdict"], "PASS")
        self.assertEqual(by[("clean", "hailo8")]["verdict"], "PASS")
        self.assertEqual(by[("clean", "trt")]["artifact_sha256"], "t" * 64)

    def test_with_build_the_files_are_hashed(self):
        if not (ROOT / "build/s3/student.onnx").exists():
            self.skipTest("build/s3 is not here")
        rows = mc.judge(mc.load_policy(str(ROOT / "policies/package_entrance.toml")), ["clean"],
                        ROOT / "build/s3", ROOT / "results/records/s3")
        self.assertFalse(any(r["replay"] for r in rows))


if __name__ == "__main__":
    unittest.main()
