import importlib.util
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
spec = importlib.util.spec_from_file_location("compare_gates", ROOT / "scripts/compare_gates.py")
compare_gates = importlib.util.module_from_spec(spec)
spec.loader.exec_module(compare_gates)
compare, summary = compare_gates.compare, compare_gates.summary

LIMITS = {"event_recall": {"min": 0.9}, "false_alerts_per_h": {"max": 1.0}}
MODEL = "m" * 64
CFG = "e" * 64
AT = "2026-11-02T14:34:34+00:00"


def row(candidate, backend, recall, false, verdict="HOLD", reasons=(), measured_at=AT):
    return {"run": "s3", "candidate": candidate, "backend": backend, "verdict": verdict, "reasons": list(reasons),
            "metrics": {"event_recall": recall, "false_alerts_per_h": false}, "measured_at": measured_at,
            "model_sha256": MODEL, "artifact_sha256": candidate[0] * 64}


def audit(backend, artifact, vetoed=False, measured_at=AT, cfg=CFG):
    """An audit as load_audits lists it, naming the record file of that measurement."""
    name = f"{backend}_{artifact[:12]}_{measured_at.replace(':', '')}.json"
    return {(MODEL, backend, artifact): [{"path": "a.json", "audited_at": "2026-11-03T00:00:00+00:00",
                                         "vetoed_by": "glimmer" if vetoed else None,
                                         "reason": "lamp-night recall 0.5 (E3)" if vetoed else None,
                                         "records": {f"runs/records/s3/clean/{name}".split("/")[-1]},
                                         "eval_config_sha256": cfg}]}


class CompareGates(unittest.TestCase):
    def test_every_policy_ships_what_it_should(self):
        rows = [row("clean", "trt", 0.96, 4.2, reasons=["false_alerts_per_h: 4.2 > max 1"]),
                row("w4", "hailo8", 0.27, 0.4, reasons=["event_recall: 0.27 < min 0.9"]),
                row("no_norm", "trt", 0.0, 0.0, reasons=["event_recall: 0 < min 0.9", "detection_delay_s_p95: None"]),
                row("good", "trt", 0.95, 0.2, verdict="PASS")]
        rows = compare(rows, {}, LIMITS, CFG)
        self.assertEqual([r["always deploy"] for r in rows], ["ships"] * 4)
        # The single-metric gate ships the quiet builds, however many arrivals they miss.
        self.assertEqual([r["false alerts only"] for r in rows], ["holds", "ships", "ships", "ships"])
        self.assertEqual(rows[0]["rules"], "HOLD (false_alerts_per_h)")
        self.assertEqual(rows[2]["rules"], "HOLD (event_recall, detection_delay_s_p95)")
        self.assertEqual(rows[3]["rules"], "PASS")
        self.assertEqual(rows[3]["rules + Glimmer"], "PASS (not audited)")
        lines = summary(rows, LIMITS)
        self.assertEqual(lines[0], "always deploy: ships all 4 builds; 2 would miss more than 10% of arrivals, "
                                   "1 would raise more than 1 false alert/h")
        self.assertTrue(lines[1].startswith("false alerts only: ships 3: s3 w4 hailo8 (recall 0.27, 0.4 false/h)"))
        self.assertEqual(lines[2], "rules: ships 1: s3 good trt (recall 0.95, 0.2 false/h)")
        self.assertEqual(lines[3], "rules + Glimmer: ships 1: s3 good trt (recall 0.95, 0.2 false/h); 0 audited, 0 vetoed")

    def test_veto_only_turns_a_pass_into_hold(self):
        rows = [row("good", "trt", 0.95, 0.2, verdict="PASS"),
                row("clean", "hailo8", 0.96, 4.2, reasons=["false_alerts_per_h: 4.2 > max 1"])]
        audits = {**audit("trt", "g" * 64, vetoed=True), **audit("hailo8", "c" * 64, vetoed=True)}
        rows = compare(rows, audits, LIMITS, CFG)
        self.assertEqual(rows[0]["rules + Glimmer"], "HOLD (veto: lamp-night recall 0.5 (E3))")
        self.assertEqual(rows[1]["rules + Glimmer"], "HOLD (rules), audited")  # the rules already held it
        self.assertTrue(summary(rows, LIMITS)[3].startswith("rules + Glimmer: ships 0; 2 audited, 2 vetoed"))

    def test_audit_of_another_file_does_not_count(self):
        rows = compare([row("good", "trt", 0.95, 0.2, verdict="PASS")], audit("trt", "x" * 64, vetoed=True), LIMITS, CFG)
        self.assertEqual(rows[0]["rules + Glimmer"], "PASS (not audited)")

    def test_audit_of_an_earlier_measurement_does_not_count(self):
        # same model, same file, but the audit named the record of an earlier measurement:
        # it says nothing about this one (deploy.py refuses it the same way)
        earlier = audit("trt", "g" * 64, vetoed=True, measured_at="2026-11-01T10:00:00+00:00")
        rows = compare([row("good", "trt", 0.95, 0.2, verdict="PASS")], earlier, LIMITS, CFG)
        self.assertEqual(rows[0]["rules + Glimmer"], "PASS (not audited)")
        # other evaluation settings: not evidence either
        other = audit("trt", "g" * 64, vetoed=True, cfg="0" * 64)
        rows = compare([row("good", "trt", 0.95, 0.2, verdict="PASS")], other, LIMITS, CFG)
        self.assertEqual(rows[0]["rules + Glimmer"], "PASS (not audited)")
        # the newest audit does not name this record but an older one does: the older one counts
        both = {**audit("trt", "g" * 64, vetoed=True)}
        both[(MODEL, "trt", "g" * 64)].insert(0, {**earlier[(MODEL, "trt", "g" * 64)][0], "audited_at": "2026-11-04T00:00:00+00:00"})
        rows = compare([row("good", "trt", 0.95, 0.2, verdict="PASS")], both, LIMITS, CFG)
        self.assertEqual(rows[0]["rules + Glimmer"], "HOLD (veto: lamp-night recall 0.5 (E3))")


if __name__ == "__main__":
    unittest.main()
