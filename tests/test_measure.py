import hashlib
import importlib.util
import json
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
spec = importlib.util.spec_from_file_location("measure", ROOT / "scripts/measure.py")
measure = importlib.util.module_from_spec(spec)
spec.loader.exec_module(measure)


def make_session(data, name="e01"):
    sdir = data / "sessions" / name
    (sdir / "frames").mkdir(parents=True)
    for i in range(3):
        (sdir / "frames" / f"{i:06d}.jpg").write_bytes(bytes([i]) * 100)
    (sdir / "frames.csv").write_text("seq,t\n0,0.0\n1,0.1\n2,0.2\n")
    (sdir / "meta.json").write_text(json.dumps({"split": "eval", "lighting": "day"}))
    (data / "gt").mkdir(exist_ok=True)
    (data / "gt" / f"{name}.json").write_text(json.dumps({"duration_s": 0.2, "events": []}))
    return sdir


class EvalManifestTest(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.data = Path(self.tmp.name)
        self.sdir = make_session(self.data)

    def tearDown(self):
        self.tmp.cleanup()

    def hash(self):
        return measure.eval_set_hash(measure.eval_manifest(["e01"], self.data)[0])

    def test_changing_an_image_changes_the_eval_set(self):
        before = self.hash()
        (self.sdir / "frames" / "000001.jpg").write_bytes(b"\x07" * 100)  # same name, same count
        self.assertNotEqual(self.hash(), before)

    def test_marks_and_meta_are_in_it(self):
        before = self.hash()
        (self.data / "gt" / "e01.json").write_text(json.dumps({"duration_s": 0.2, "events": [{"arrive": 0.1}]}))
        after_gt = self.hash()
        (self.sdir / "meta.json").write_text(json.dumps({"split": "eval", "lighting": "night"}))
        self.assertEqual(len({before, after_gt, self.hash()}), 3)

    def test_digest_matches_the_device_formula(self):
        # run_backend.py: sha256 over "NAME SHA256\n" per frame, in name order
        d = hashlib.sha256()
        for p in sorted((self.sdir / "frames").glob("*.jpg")):
            d.update(f"{p.name} {hashlib.sha256(p.read_bytes()).hexdigest()}\n".encode())
        manifest, _ = measure.eval_manifest(["e01"], self.data)
        self.assertEqual(manifest["e01"]["frames_sha256"], d.hexdigest())


if __name__ == "__main__":
    unittest.main()
