import copy
import unittest
from datetime import datetime, timezone
from pathlib import Path

from edgegate.gate import HOLD, PASS, apply_veto, decide, load_policy

ROOT = Path(__file__).resolve().parent.parent
NOW = datetime(2026, 10, 31, 12, 0, tzinfo=timezone.utc)
MODEL = "m" * 64
EVAL = "e" * 64

GOOD = {"event_recall": 0.95, "false_alerts_per_h": 0.2, "duplicate_alerts": 0,
        "detection_delay_s_p95": 3.0, "latency_ms_p95": 20.0, "peak_mem_mb": 300.0}


def record(backend, **metric_overrides):
    return {"schema": 1, "model_sha256": MODEL, "backend": backend, "device": "test",
            "eval_set_sha256": EVAL, "measured_at": "2026-10-31T10:00:00Z",
            "tool": {"ok": True, "error": None}, "metrics": {**GOOD, **metric_overrides}}


class GateTest(unittest.TestCase):
    def setUp(self):
        self.policy = load_policy(ROOT / "policies/package_entrance.toml")
        self.policy["task"]["eval_set_sha256"] = EVAL
        self.ref = record("onnx-gx10", event_recall=0.96)

    def one(self, rec, backend=None, records=None):
        backend = backend or rec["backend"]
        recs = records if records is not None else [self.ref, rec]
        return decide(self.policy, recs, MODEL, [backend], NOW)[0]

    def assertHeldFor(self, d, check):
        self.assertEqual(d.verdict, HOLD)
        self.assertIn(check, [c for c, _ in d.reasons])

    def test_good_record_passes(self):
        d = self.one(record("trt"))
        self.assertEqual(d.verdict, PASS, d.reasons)

    def test_missing_record_holds(self):
        self.assertHeldFor(self.one(None, backend="trt", records=[self.ref]), "missing")

    def test_policy_without_eval_set_holds(self):
        self.policy["task"]["eval_set_sha256"] = "unset"
        rec = record("trt")
        rec["eval_set_sha256"] = "unset"
        self.assertHeldFor(self.one(rec), "policy")

    def test_tool_failure_holds(self):
        rec = record("trt")
        rec["tool"] = {"ok": False, "error": "ssh timeout"}
        self.assertHeldFor(self.one(rec), "tool")

    def test_tool_ok_must_be_true_not_truthy(self):
        rec = record("trt")
        rec["tool"] = {"ok": "yes"}
        self.assertHeldFor(self.one(rec), "tool")

    def test_other_model_holds(self):
        rec = record("trt")
        rec["model_sha256"] = "x" * 64
        self.assertHeldFor(self.one(rec), "model")

    def test_other_eval_set_holds(self):
        rec = record("trt")
        rec["eval_set_sha256"] = "x" * 64
        self.assertHeldFor(self.one(rec), "eval_set")

    def test_stale_holds(self):
        rec = record("trt")
        rec["measured_at"] = "2026-10-29T10:00:00Z"
        self.assertHeldFor(self.one(rec), "stale")

    def test_future_or_naive_time_holds(self):
        for t in ("2026-11-01T10:00:00Z", "2026-10-31T10:00:00", "yesterday", None):
            rec = record("trt")
            rec["measured_at"] = t
            self.assertHeldFor(self.one(rec), "time")

    def test_nan_inf_missing_and_wrong_type_hold(self):
        for v in (float("nan"), float("inf"), None, "20", True):
            self.assertHeldFor(self.one(record("trt", latency_ms_p95=v)), "latency_ms_p95")
        rec = record("trt")
        del rec["metrics"]["peak_mem_mb"]
        self.assertHeldFor(self.one(rec), "peak_mem_mb")

    def test_thresholds(self):
        self.assertHeldFor(self.one(record("trt", event_recall=0.85)), "event_recall")
        self.assertHeldFor(self.one(record("trt", false_alerts_per_h=3.0)), "false_alerts_per_h")
        self.assertHeldFor(self.one(record("trt", duplicate_alerts=1)), "duplicate_alerts")

    def test_backend_override(self):
        self.assertEqual(self.one(record("trt", latency_ms_p95=80.0)).verdict, PASS)
        self.assertHeldFor(self.one(record("hailo8", latency_ms_p95=80.0)), "latency_ms_p95")

    def test_drop_against_reference(self):
        # 0.91 clears min 0.9 but is 0.05+ below the 0.96 reference.
        self.assertHeldFor(self.one(record("trt", event_recall=0.905)), "event_recall")
        self.assertEqual(self.one(record("trt", event_recall=0.92)).verdict, PASS)

    def test_missing_or_invalid_reference_holds(self):
        self.assertHeldFor(self.one(record("trt"), records=[record("trt")]), "event_recall")
        bad_ref = copy.deepcopy(self.ref)
        bad_ref["tool"]["ok"] = False
        self.assertHeldFor(self.one(record("trt"), records=[bad_ref, record("trt")]), "event_recall")

    def test_latest_record_wins(self):
        old = record("trt", event_recall=0.5)
        old["measured_at"] = "2026-10-31T08:00:00Z"
        self.assertEqual(self.one(None, "trt", [self.ref, old, record("trt")]).verdict, PASS)

    def test_veto_only_demotes(self):
        ds = decide(self.policy, [self.ref, record("trt"), record("hailo8", event_recall=0.5)],
                    MODEL, ["trt", "hailo8"], NOW)
        apply_veto(ds, {"trt": "boxes shifted left on night frames", "hailo8": "looks fine"})
        trt, hailo = ds
        self.assertHeldFor(trt, "veto")
        self.assertEqual(trt.vetoed_by, "glimmer")
        self.assertEqual(hailo.verdict, HOLD)
        self.assertIsNone(hailo.vetoed_by)  # held by the rules, not the veto


if __name__ == "__main__":
    unittest.main()
