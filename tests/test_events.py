import json
import tempfile
import unittest
from pathlib import Path

from edgegate.events import ArrivalDetector, alerts, evaluate, inside, percentile, score_session, summarize
from edgegate.gate import load_policy

ROOT = Path(__file__).resolve().parent.parent
BOX_A = [0.3, 0.6, 0.45, 0.9, 0.9]     # bottom-center (0.375, 0.9) is in the zone
BOX_B = [0.55, 0.6, 0.7, 0.9, 0.9]     # second package next to A
BOX_OUT = [0.4, 0.1, 0.6, 0.3, 0.9]    # bottom-center (0.5, 0.3) is above the zone
BOX_WEAK = [0.4, 0.6, 0.6, 0.9, 0.3]
PERSON = [0.2, 0.2, 0.6, 1.0, 0.9, "person"]   # stands over BOX_A


def timeline(spans, end, fps=10, start=0.0):
    """Frames at fps over [start, end); spans are (from, to, box) when that box is seen."""
    frames = []
    for k in range(int(start * fps), int(end * fps)):
        t = k / fps
        frames.append((t, [box for a, b, box in spans if a <= t < b]))
    return frames


class AlertRuleTest(unittest.TestCase):
    def setUp(self):
        self.p = load_policy(ROOT / "policies/package_entrance.toml")["events"]

    def alerts(self, spans, end=100, **kw):
        return alerts(timeline(spans, end, **kw), self.p)[0]

    def test_inside(self):
        zone = self.p["zone"]
        self.assertTrue(inside((0.5, 0.9), zone))
        self.assertFalse(inside((0.5, 0.3), zone))
        self.assertFalse(inside((0.1, 0.9), zone))

    def test_dwell_is_time_not_frames(self):
        for fps in (5, 10, 30):
            a = self.alerts([(10, 60, BOX_A)], fps=fps)
            self.assertEqual(len(a), 1)
            self.assertAlmostEqual(a[0], 12.0, places=5)

    def test_walking_through_with_a_package_does_not_alert(self):
        self.assertEqual(self.alerts([(10, 11.5, BOX_A)]), [])

    def test_out_of_zone_and_weak_boxes_ignored(self):
        self.assertEqual(self.alerts([(10, 60, BOX_OUT)]), [])
        self.assertEqual(self.alerts([(10, 60, BOX_WEAK)]), [])

    def test_missed_detections_while_placing_give_one_alert(self):
        spans = [(10, 10.8, BOX_A), (11.1, 11.9, BOX_A), (12.5, 40, BOX_A)]
        self.assertEqual(len(self.alerts(spans)), 1)

    def test_one_alert_while_the_package_stays(self):
        self.assertEqual(len(self.alerts([(10, 30, BOX_A), (31, 60, BOX_A)])), 1)

    def test_empty_for_clear_s_then_placed_again_is_a_new_arrival(self):
        self.assertEqual(len(self.alerts([(10, 30, BOX_A), (33, 60, BOX_A)])), 2)

    def test_hidden_by_a_person_is_not_empty(self):
        # A person stands over the package for 10 s; it is not a new arrival afterwards.
        spans = [(10, 30, BOX_A), (30, 40, PERSON), (40, 60, BOX_A)]
        self.assertEqual(len(self.alerts(spans)), 1)

    def test_second_package_next_to_the_first_is_out_of_scope(self):
        self.assertEqual(len(self.alerts([(10, 90, BOX_A), (40, 90, BOX_B)])), 1)

    def test_package_present_at_start_is_not_alerted(self):
        self.assertEqual(self.alerts([(0, 60, BOX_A)]), [])

    def test_camera_reconnect_with_package_present_is_not_alerted(self):
        frames = timeline([(10, 100, BOX_A)], 30) + timeline([(10, 100, BOX_A)], 60, start=40)
        times, outages = alerts(frames, self.p)
        self.assertEqual(len(times), 1)  # only the original arrival at ~12 s
        self.assertEqual(len(outages), 1)

    def test_arrival_during_outage_is_never_alerted(self):
        # Known limit: the package arrived at 35 s while the camera was away.
        frames = timeline([], 30) + timeline([(35, 90, BOX_A)], 90, start=40)
        times, outages = alerts(frames, self.p)
        self.assertEqual(times, [])
        s = score_session(times, [{"arrive": 35, "remove": None}], 10, outages=outages)
        self.assertEqual((s["matched"], s["outage_misses"]), (0, 1))

    def test_arrival_after_reconnect_is_alerted(self):
        frames = timeline([], 30) + timeline([(50, 90, BOX_A)], 90, start=40)
        self.assertEqual(len(alerts(frames, self.p)[0]), 1)

    def test_status_is_unknown_while_the_camera_is_away(self):
        det = ArrivalDetector(self.p)
        self.assertEqual(det.status(0), "unknown")
        for t, dets in timeline([], 5):
            det.update(t, dets)
        self.assertEqual(det.status(5), "empty")
        self.assertEqual(det.status(10), "unknown")


class ScoringTest(unittest.TestCase):
    def test_score_session(self):
        events = [{"arrive": 10, "remove": 50}, {"arrive": 100, "remove": None},
                  {"arrive": 200, "remove": 260}]
        # 12: match #0; 30: dup of #0; 70: false; 103: match #1; 230: late for #2
        s = score_session([12, 30, 70, 103, 230], events, window=10)
        self.assertEqual((s["matched"], s["duplicate"], s["false"], s["late"]), (2, 1, 1, 1))
        self.assertEqual(sorted(s["delays"]), [2, 3])

    def test_notify_delay_counts_toward_the_window(self):
        events = [{"arrive": 10, "remove": None}]
        self.assertEqual(score_session([18], events, window=10, notify_s=1.5)["matched"], 1)
        s = score_session([18], events, window=10, notify_s=2.5)  # received at 20.5 s
        self.assertEqual((s["matched"], s["late"]), (0, 1))

    def test_second_package_is_scored_apart_from_recall(self):
        # #1 arrives while #0 is still there (removed together at 60); #2 comes after both.
        events = [{"arrive": 10, "remove": 60}, {"arrive": 20, "remove": 60}, {"arrive": 100, "remove": None}]
        s = score_session([12, 103], events, window=10)  # the rule does not alert for #1
        self.assertEqual((s["events"], s["matched"], s["out_of_scope"], s["out_of_scope_matched"]), (2, 2, 1, 0))
        self.assertEqual([e["second"] for e in s["per_event"]], [False, True, False])
        self.assertEqual(summarize([s], 600)["second_packages"], 1)
        # An alert in #1's window is #1's, not a duplicate or false, and not in recall or delays.
        s = score_session([12, 22, 103], events, window=10)
        self.assertEqual((s["matched"], s["out_of_scope_matched"], s["duplicate"], s["false"]), (2, 1, 0, 0))
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
                for t, d in timeline([(10, 60, BOX_A)], 100):
                    f.write(json.dumps({"t": t, "dets": d}) + "\n")
            metrics, _ = evaluate(policy, gt, dets)
            self.assertEqual(metrics["event_recall"], 1.0)
            self.assertEqual(metrics["false_alerts_per_h"], 0.0)
            Path(gt, "s02.json").write_text(json.dumps({"session": "s02", "duration_s": 100, "events": []}))
            with self.assertRaises(FileNotFoundError):
                evaluate(policy, gt, dets)


if __name__ == "__main__":
    unittest.main()
