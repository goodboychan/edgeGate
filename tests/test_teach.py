import importlib.util
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
spec = importlib.util.spec_from_file_location("teach", ROOT / "scripts/teach.py")
teach = importlib.util.module_from_spec(spec)
spec.loader.exec_module(teach)

POLICY = {"task": {"instruction": "Tell me when a package is left at the entrance."},
          "measure": {"classes": ["package", "person"]}}
SPEC = {"task": "package-at-entrance", "instruction": "tell me when a package is left at the entrance",
        "detect": ["package", "person"], "event": "a package set down in the zone and left there",
        "not_event": "a person walking through, or a package carried through", "watch": ["night", "small boxes"]}


class Teach(unittest.TestCase):
    def test_spec_is_read_out_of_prose(self):
        self.assertEqual(teach.spec_from('Sure.\n{"task": "x", "detect": ["package"]}\nDone.')["detect"], ["package"])
        self.assertIsNone(teach.spec_from("no json here"))
        self.assertIsNone(teach.spec_from("{not json}"))

    def test_matching_spec_has_no_differences(self):
        self.assertEqual(teach.compare(SPEC, POLICY), [])  # case and punctuation do not count

    def test_differences_are_named(self):
        out = teach.compare({**SPEC, "detect": ["parcel", "person"], "watch": []}, POLICY)
        self.assertEqual(len(out), 2)
        self.assertTrue(out[0].startswith("classes:"))
        self.assertEqual(out[1], "watch: missing from the spec")
        out = teach.compare({**SPEC, "instruction": "Tell me when someone is at the door"}, POLICY)
        self.assertTrue(out[0].startswith("instruction:"))


if __name__ == "__main__":
    unittest.main()
