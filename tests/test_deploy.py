import importlib.util
import io
import json
import sys
import tempfile
import unittest
from contextlib import redirect_stdout
from datetime import datetime, timedelta, timezone
from pathlib import Path
from unittest import mock

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "tests"))
sys.path.insert(0, str(ROOT / "scripts"))
from test_gate import MODEL, NOW, art, policy, record  # noqa: E402


def load(name):
    spec = importlib.util.spec_from_file_location(name, ROOT / "scripts" / f"{name}.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


deploy = load("deploy")
live = load("live_device")


class ReplayNeverDeploys(unittest.TestCase):
    """--now replays a decision; a replay must never start a runner, and an unreadable
    --now must be refused outright rather than silently judged as 'now'."""

    def run_deploy(self, tmp, now):
        rec = tmp / "trt.json"
        rec.write_text(json.dumps(record("trt")))
        ref = tmp / "ref.json"
        ref.write_text(json.dumps(record("onnx-gx10", event_recall=0.96)))
        eng = tmp / "student_fp16.engine"
        eng.write_bytes(b"engine")
        real = deploy.sha256_file
        argv = ["deploy.py", "policy.toml", str(rec), str(ref), "--model-sha", MODEL, "--artifact", f"trt={eng}",
                "--notify", "http://push.example/topic"] + (["--now", now] if now else [])
        out = io.StringIO()
        with mock.patch.object(deploy, "load_policy", lambda p: policy()), \
                mock.patch.object(deploy, "sha256_file", lambda p: art("trt") if str(p).endswith(".engine") else real(p)), \
                mock.patch.object(deploy, "start") as start, mock.patch.object(deploy, "ROOT", tmp), \
                mock.patch.object(deploy.sys, "argv", argv), redirect_stdout(out):
            deploy.main()
        return start, out.getvalue()

    def test_a_replayed_pass_starts_nothing(self):
        with tempfile.TemporaryDirectory() as d:
            start, out = self.run_deploy(Path(d), NOW.isoformat())
            self.assertIn("PASS as of", out)
            self.assertIn("deploys nothing", out)
            self.assertEqual(start.call_count, 0)
            self.assertEqual(list((Path(d) / "runs/approvals").glob("*.json")), [])

    def test_an_unreadable_now_is_refused_before_anything_runs(self):
        with tempfile.TemporaryDirectory() as d:
            with mock.patch.object(deploy.sys, "stderr", io.StringIO()):
                with self.assertRaises(SystemExit):
                    self.run_deploy(Path(d), "yesterday evening")
            self.assertFalse((Path(d) / "runs/approvals").exists())


def approval(measured_h_ago=2.0, approved_h_ago=1.0, **over):
    """A PASS approval like deploy.py writes: the record measured measured_h_ago hours
    ago, approved approved_h_ago hours ago, expiring 24 h after the measurement."""
    now = datetime.now(timezone.utc)
    measured = now - timedelta(hours=measured_h_ago)
    ap = {"verdict": "PASS", "artifact_sha256": "a" * 64, "device_id": "dev-1", "runtime": "tensorrt 10.3.0",
          "approved_at": (now - timedelta(hours=approved_h_ago)).isoformat(timespec="seconds"),
          "records_measured_at": measured.isoformat(timespec="seconds"),
          "expires_at": (measured + timedelta(hours=24)).isoformat(timespec="seconds"),
          "record": {"path": "trt.json", "sha256": "c" * 64, "measured_at": measured.isoformat(timespec="seconds")},
          "max_age_hours": 24, "events": {"dwell_s": 2.0, "zone": [[0, 0], [1, 0], [1, 1]]}, "measure": {"det_conf": 0.5}}
    ap.update(over)
    ap.setdefault("eval_config_sha256", deploy.eval_config_sha256({"events": ap["events"], "measure": ap["measure"]}))
    return ap


DEV = {"device_id": "dev-1", "runtime": "tensorrt 10.3.0"}


class ApprovalChecks(unittest.TestCase):
    """The runner re-checks a stored approval before it starts: the file, the device, the
    age and the settings hash. A stale or edited approval cannot restart a runner."""

    def test_a_fresh_matching_approval_starts(self):
        self.assertEqual(live.approval_problems(approval(), DEV, "a" * 64), [])

    def test_an_expired_approval_does_not(self):
        problems = live.approval_problems(approval(measured_h_ago=31, approved_h_ago=30), DEV, "a" * 64)
        self.assertTrue(any("older than 24 h" in p for p in problems), problems)

    def test_a_late_approval_expires_with_its_record(self):
        # approved an hour ago, but the record behind it is 25 h old: the approval must not
        # outlive the measurement (expiry counts from measured_at, not approved_at)
        problems = live.approval_problems(approval(measured_h_ago=25, approved_h_ago=1), DEV, "a" * 64)
        self.assertTrue(any("expired" in p for p in problems), problems)

    def test_a_future_approval_does_not(self):
        problems = live.approval_problems(approval(approved_h_ago=-2), DEV, "a" * 64)
        self.assertTrue(any("future" in p for p in problems), problems)

    def test_an_edited_expiry_does_not(self):
        late = (datetime.now(timezone.utc) + timedelta(hours=40)).isoformat(timespec="seconds")
        problems = live.approval_problems(approval(expires_at=late), DEV, "a" * 64)
        self.assertTrue(any("expires_at is later" in p for p in problems), problems)

    def test_an_approval_without_an_expiry_does_not(self):
        ap = approval()
        del ap["expires_at"]
        problems = live.approval_problems(ap, DEV, "a" * 64)
        self.assertTrue(any("expires_at" in p for p in problems), problems)

    def test_deploy_writes_the_expiry_from_the_oldest_record(self):
        rec = record("trt")
        ref = record("onnx-gx10", event_recall=0.96)
        ref["measured_at"] = "2026-10-30T20:00:00+00:00"  # older than the device record
        pol = policy()
        a = type("A", (), {"model_sha": MODEL, "files": {"trt": "x.engine"}, "audit": None, "policy": __file__})()
        with mock.patch.object(deploy, "sha256_file", lambda p: "f" * 64):
            ap = deploy.approval(pol, a, "trt", rec, "trt.json", art("trt"), None,
                                 min(deploy._parse_time(r["measured_at"]) for r in (rec, ref)))
        self.assertEqual(ap["records_measured_at"], "2026-10-30T20:00:00+00:00")
        self.assertEqual(ap["expires_at"], "2026-10-31T20:00:00+00:00")

    def test_changed_settings_do_not(self):
        ap = approval()
        ap["events"]["dwell_s"] = 3.0  # edited after approval: the hash no longer matches
        problems = live.approval_problems(ap, DEV, "a" * 64)
        self.assertTrue(any("eval_config_sha256" in p for p in problems), problems)

    def test_an_approval_without_an_age_does_not(self):
        ap = approval()
        del ap["max_age_hours"]
        problems = live.approval_problems(ap, DEV, "a" * 64)
        self.assertTrue(any("max_age_hours" in p for p in problems), problems)

    def test_other_file_or_device_does_not(self):
        self.assertTrue(live.approval_problems(approval(), DEV, "b" * 64))
        self.assertTrue(live.approval_problems(approval(), {**DEV, "device_id": "dev-2"}, "a" * 64))


class AuditBinding(unittest.TestCase):
    """An audit record is evidence about specific records and settings; deploy refuses one
    that names other records, other settings, or none at all."""

    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        tmp = Path(self.tmp.name)
        self.rec = tmp / "trt.json"
        self.rec.write_text(json.dumps(record("trt")))
        self.policy = policy()
        self.base = {"model_sha256": MODEL, "artifacts": {"trt": art("trt")}, "vetoes": {},
                     "final": [{"backend": "trt", "verdict": "PASS", "reasons": [], "vetoed_by": None}],
                     "record_sha256": {str(self.rec): deploy.sha256_file(self.rec)},
                     "eval_config_sha256": deploy.eval_config_sha256(self.policy)}

    def tearDown(self):
        self.tmp.cleanup()

    def write(self, **over):
        p = Path(self.tmp.name) / "audit.json"
        p.write_text(json.dumps({**self.base, **over}))
        return p

    def test_a_bound_audit_is_accepted(self):
        audit, vetoes = deploy.audit_vetoes(self.write(), MODEL, {"trt": art("trt")}, [self.rec], self.policy)
        self.assertEqual(vetoes, {})

    def test_an_unbound_audit_is_refused(self):
        with self.assertRaises(SystemExit):
            deploy.audit_vetoes(self.write(record_sha256={}), MODEL, {"trt": art("trt")}, [self.rec], self.policy)

    def test_an_audit_of_other_records_is_refused(self):
        other = Path(self.tmp.name) / "other.json"
        rec2 = record("trt")
        rec2["measured_at"] = "2026-10-31T11:00:00Z"  # a later measurement the audit never saw
        other.write_text(json.dumps(rec2))
        with self.assertRaises(SystemExit):
            deploy.audit_vetoes(self.write(), MODEL, {"trt": art("trt")}, [other], self.policy)

    def test_an_audit_of_other_settings_is_refused(self):
        with self.assertRaises(SystemExit):
            deploy.audit_vetoes(self.write(eval_config_sha256="0" * 64), MODEL, {"trt": art("trt")}, [self.rec], self.policy)


if __name__ == "__main__":
    unittest.main()
