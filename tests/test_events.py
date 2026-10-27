import json
import tempfile
import unittest
from pathlib import Path

from edgegate.events import alerts, evaluate, inside, percentile, score_session, summarize
from edgegate.gate import load_policy

ROOT = Path(__file__).resolve().parent.parent
BOX_IN = [0.4, 0.6, 0.6, 0.9, 0.9]     # bottom-center (0.5, 0.9) is in the zone
BOX_OUT = [0.4, 0.1, 0.6, 0.3, 0.9]    # bottom-center (0.5, 0.3) is above it
BOX_WEAK = [0.4, 0.6, 0.6, 0.9, 0.3]


def timeline(spans, end, fps=10, box=BOX_IN):
    """Frames at fps from 0 to end; `box` is present during [start, stop) spans."""
    frames = []
    for k in range(int(end * fps)):
        t = k / fps
        frames.append((t, [box] if any(a <= t < b for a, b in spans) else []))
    return frames


class EventsTest(unittest.TestCase):
    def setUp(self):
        self.p = load_policy(ROOT / "policies/package_entrance.toml")["events"]

    def test_inside(self):
        zone = self.p["zone"]
        self.assertTrue(inside((0.5, 0.9), zone))
        self.assertFalse(inside((0.5, 0.3), zone))
        self.assertFalse(inside((0.1, 0.9), zone))

    def test_one_arrival_one_alert_after_dwell(self):
        a = alerts(timeline([(10, 60)], 100), self.p)
        self.assertEqual(len(a), 1)
        self.assertAlmostEqual(a[0], 12.0, places=5)

    def test_short_visit_does_not_alert(self):
        self.assertEqual(alerts(timeline([(10, 11.5)], 100), self.p), [])

    def test_out_of_zone_and_weak_boxes_ignored(self):
        self.assertEqual(alerts(timeline([(10, 60)], 100, box=BOX_OUT), self.p), [])
        self.assertEqual(alerts(timeline([(10, 60)], 100, box=BOX_WEAK), self.p), [])

    def test_flicker_within_gap_keeps_presence(self):
        # 0.3 s dropouts are shorter than gap_s, so dwell keeps counting.
        spans = [(10, 10.8), (11.1, 11.9), (12.2, 40)]
        a = alerts(timeline(spans, 60), self.p)
        self.assertEqual(len(a), 1)
        self.assertLess(a[0], 12.5)

    def test_brief_dropout_does_not_realert(self):
        # 2 s dropout breaks presence but is shorter than clear_s.
        self.assertEqual(len(alerts(timeline([(10, 30), (32, 60)], 100), self.p)), 1)

    def test_long_absence_rearms(self):
        self.assertEqual(len(alerts(timeline([(10, 30), (40, 60)], 100), self.p)), 2)

    def test_score_session(self):
        events = [{"arrive": 10, "remove": 50}, {"arrive": 100, "remove": None},
                  {"arrive": 200, "remove": 260}]
        # 12: match #0; 30: dup of #0; 70: false; 103: match #1; 230: late for #2
        s = score_session([12, 30, 70, 103, 230], events, window=10)
        self.assertEqual((s["matched"], s["duplicate"], s["false"], s["late"]), (2, 1, 1, 1))
        self.assertEqual(sorted(s["delays"]), [2, 3])

    def test_summarize_pools_sessions(self):
        a = {"events": 3, "matched": 3, "duplicate": 0, "late": 0, "false": 1, "delays": [1, 2, 3]}
        b = {"events": 1, "matched": 0, "duplicate": 1, "late": 0, "false": 0, "delays": []}
        m = summarize([a, b], duration_s=1800)
        self.assertEqual(m["event_recall"], 0.75)
        self.assertEqual(m["false_alerts_per_h"], 2.0)
        self.assertEqual(m["duplicate_alerts"], 1)
        self.assertEqual(m["detection_delay_s_p95"], 3)

    def test_empty_inputs_give_none(self):
        self.assertIsNone(percentile([], 95))
        m = summarize([{"events": 0, "matched": 0, "duplicate": 0, "late": 0, "false": 0, "delays": []}], 600)
        self.assertIsNone(m["event_recall"])
        self.assertIsNone(m["detection_delay_s_p95"])

    def test_evaluate_files_and_missing_session(self):
        policy = load_policy(ROOT / "policies/package_entrance.toml")
        with tempfile.TemporaryDirectory() as tmp:
            gt, dets = Path(tmp, "gt"), Path(tmp, "dets")
            gt.mkdir()
            dets.mkdir()
            Path(gt, "s01.json").write_text(json.dumps(
                {"session": "s01", "duration_s": 100, "events": [{"arrive": 10, "remove": 60}]}))
            with open(Path(dets, "s01.jsonl"), "w") as f:
                for t, d in timeline([(10, 60)], 100):
                    f.write(json.dumps({"t": t, "dets": d}) + "\n")
            metrics, _ = evaluate(policy, gt, dets)
            self.assertEqual(metrics["event_recall"], 1.0)
            self.assertEqual(metrics["false_alerts_per_h"], 0.0)
            Path(gt, "s02.json").write_text(json.dumps({"session": "s02", "duration_s": 100, "events": []}))
            with self.assertRaises(FileNotFoundError):
                evaluate(policy, gt, dets)


if __name__ == "__main__":
    unittest.main()
