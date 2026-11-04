import unittest

from edgegate.audit import Audit, slice_rows
from edgegate.gate import HOLD, PASS, _parse_time

from tests import audit_cases as cases


def audit(case):
    policy = cases.policy()
    records = [cases.record(case, b) for b in ("onnx-gx10", "trt", "hailo8")]
    return Audit(policy, records, cases.MODEL, cases.ARTIFACTS, _parse_time(cases.NOW))


class SyntheticCasesTest(unittest.TestCase):
    def test_extra_check_cases_pass_the_rules(self):
        # The point of these cases: summary metrics pass, the problem is in a slice.
        for case in ("control", "night_gap", "small_miss", "occluded"):
            a = audit(case)
            self.assertEqual({b: d.verdict for b, d in a.decisions.items()}, {"trt": PASS, "hailo8": PASS}, case)

    def test_rule_hold_case(self):
        self.assertEqual(audit("rule_hold").decisions["hailo8"].verdict, HOLD)


class SliceTest(unittest.TestCase):
    def test_small_packages_show_the_miss(self):
        a = audit("small_miss")
        r = a.slice_metrics("hailo8", "size")
        self.assertEqual(r["slices"]["small"]["recall"], 0.5)
        self.assertEqual(r["slices"]["small"]["on_time_minus_reference"], -3)
        self.assertEqual(r["reference"]["slices"]["small"]["recall"], 1.0)

    def test_occluded_counts_late_apart(self):
        r = audit("occluded").slice_metrics("trt", "tag")["slices"]["occluded"]
        self.assertEqual((r["on_time"], r["late"], r["missed"]), (3, 1, 2))

    def test_coverage_has_no_night(self):
        c = audit("night_gap").eval_coverage()
        self.assertNotIn("night", c["by_lighting"])
        self.assertEqual(c["by_lighting"]["day"]["hours"], 1.5)

    def test_lighting_rows_carry_hours_and_false_alerts(self):
        rec = cases.record("control", "trt")
        rec["per_session"][0]["false"] = 3
        rows = slice_rows(rec, "lighting", [0.08, 0.15])
        self.assertEqual(rows["day"]["false_alerts"], 3)
        self.assertEqual(rows["day"]["false_alerts_per_h"], 3.0)

    def test_second_package_is_out_of_scope_as_in_the_gate(self):
        # events.py leaves a second package out of recall; the slices must count the same events.
        rec = cases.record("control", "trt")
        before = {by: slice_rows(rec, by, [0.08, 0.15]) for by in ("lighting", "session", "size", "tag")}
        first = rec["per_session"][0]["per_event"][0]
        rec["per_session"][0]["per_event"].append({**first, "outcome": "missed", "second": True})
        for by, rows in before.items():
            after = slice_rows(rec, by, [0.08, 0.15])
            for k, r in after.items():
                self.assertEqual({**r, "out_of_scope": 0}, rows[k], (by, k))
            self.assertEqual(sum(r["out_of_scope"] for r in after.values()), 1, by)

    def test_unknown_slice_and_backend(self):
        a = audit("control")
        self.assertIn("error", a.slice_metrics("trt", "color"))
        err = a.slice_metrics("jetson-orin", "size")["error"]
        self.assertIn("'trt'", err)  # the error names the backends to use

    def test_evidence_ids_count_up(self):
        a = audit("control")
        self.assertEqual(a.gate_decisions()["evidence_id"], "E1")
        self.assertEqual(a.slice_metrics("trt", "size")["evidence_id"], "E2")


class ProposeHoldTest(unittest.TestCase):
    def test_accepted_with_evidence_about_that_backend(self):
        a = audit("small_miss")
        e = a.slice_metrics("hailo8", "size")["evidence_id"]
        self.assertTrue(a.propose_hold("hailo8", "small packages missed", [e])["accepted"])
        self.assertIn("hailo8", a.vetoes)

    def test_coverage_counts_for_any_backend(self):
        a = audit("night_gap")
        e = a.eval_coverage()["evidence_id"]
        self.assertTrue(a.propose_hold("trt", "no night sessions", [e])["accepted"])

    def test_rejected_without_evidence(self):
        a = audit("control")
        self.assertFalse(a.propose_hold("trt", "looks risky", [])["accepted"])
        self.assertFalse(a.propose_hold("trt", "looks risky", ["E9"])["accepted"])
        self.assertEqual(a.vetoes, {})
        self.assertEqual(len(a.rejected), 2)

    def test_rejected_with_evidence_about_another_backend(self):
        a = audit("small_miss")
        e = a.slice_metrics("hailo8", "size")["evidence_id"]
        self.assertFalse(a.propose_hold("trt", "small packages missed", [e])["accepted"])

    def test_summary_alone_is_not_evidence_about_a_backend(self):
        a = audit("control")
        e = a.gate_decisions()["evidence_id"]
        self.assertFalse(a.propose_hold("trt", "recall is close to the limit", [e])["accepted"])

    def test_cannot_touch_a_rule_hold(self):
        a = audit("rule_hold")
        e = a.slice_metrics("hailo8", "size")["evidence_id"]
        r = a.propose_hold("hailo8", "slow", [e])
        self.assertFalse(r["accepted"])
        self.assertNotIn("hailo8", a.vetoes)


if __name__ == "__main__":
    unittest.main()
