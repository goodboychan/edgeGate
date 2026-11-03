#!/usr/bin/env python3
"""Compare four ways to ship the same builds, from the logged measurements and audits.

Every build (candidate x backend) with a measurement record under runs/records/RUN/ is
judged the way scripts/measure_candidates.py judges it, then placed under each policy:
  always deploy    ships every build; its measured recall and false alerts are what the
                   home would have lived with. A what-if from the records, never a
                   deployment: a held build is only ever run live as a labeled replay
                   (scripts/replay.py), with alerts marked as such.
  false alerts     ships every build at or under the false-alert limit and checks
                   nothing else: the single-metric gate a quick demo would use.
  rules            the policy verdict (edgegate/gate.py), as of --now if given.
  rules + Glimmer  the rules verdict plus the latest audit of that exact model and file
                   (runs/audits/, scripts/audit.py); a veto only turns PASS into HOLD.

Usage:  python3 scripts/compare_gates.py [--runs s1,s3] [--now ISO] [--json out.json]
"""
import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))
from edgegate.gate import PASS, _parse_time, load_policy  # noqa: E402
from measure_candidates import CANDIDATES, judge  # noqa: E402

POLICIES = ("always deploy", "false alerts only", "rules", "rules + Glimmer")


def builds(policy, runs, now=None, all_sets=False, records=None):
    """One row per measured build, in run order: the judge row plus its run name.

    Builds measured on another eval set are left out unless all_sets: their numbers are
    not comparable with the rest, and "always deploy" would ship them on those numbers.
    `records` is the records directory (default runs/records; results/records replays the
    committed snapshot, with the hashes read from the records since build/ is not in git).
    """
    records = Path(records) if records else ROOT / "runs/records"
    rows, left_out = [], 0
    for run in runs:
        for r in judge(policy, list(CANDIDATES), ROOT / "build" / run, records / run, now):
            if r["candidate"] == "reference" or not r["measured"]:
                continue
            if not all_sets and any(x.startswith("eval_set:") for x in r["reasons"]):
                left_out += 1
                continue
            rows.append({"run": run, **r})
    return rows, left_out


def load_audits(path):
    """{(model sha, backend, artifact sha): latest final verdict for that backend}."""
    out = {}
    for p in sorted(Path(path).glob("*.json")):
        a = json.loads(p.read_text())
        for f in a.get("final", []):
            key = (a.get("model_sha256"), f["backend"], (a.get("artifacts") or {}).get(f["backend"]))
            hit = {"path": str(p), "audited_at": a.get("audited_at") or "", "vetoed_by": f.get("vetoed_by"),
                   "reason": f["reasons"][-1][1] if f.get("vetoed_by") and f.get("reasons") else None}
            if key not in out or hit["audited_at"] > out[key]["audited_at"]:
                out[key] = hit
    return out


def compare(rows, audits, limits):
    """Add each policy's verdict to every row; the metrics are the latest record's."""
    fmax = limits["false_alerts_per_h"]["max"]
    for r in rows:
        fa = r["metrics"].get("false_alerts_per_h")
        r["always deploy"] = "ships"
        r["false alerts only"] = "-" if fa is None else "ships" if fa <= fmax else "holds"
        held = ", ".join(dict.fromkeys(x.split(":")[0] for x in r["reasons"]))
        r["rules"] = PASS if r["verdict"] == PASS else f"HOLD ({held})"
        audit = audits.get((r["model_sha256"], r["backend"], r["artifact_sha256"]))
        r["audited"], r["vetoed"] = audit is not None, bool(audit and audit["vetoed_by"])
        if r["verdict"] != PASS:
            r["rules + Glimmer"] = "HOLD (rules)" + (", audited" if audit else "")
        elif r["vetoed"]:
            r["rules + Glimmer"] = f"HOLD (veto: {audit['reason']})"
        else:
            r["rules + Glimmer"] = PASS + (" (audited)" if audit else " (not audited)")
    return rows


def name(r):
    return f"{r['run']} {r['candidate']} {r['backend']}"


def summary(rows, limits):
    """One line per policy: how many builds it ships and what that would have cost."""
    rmin, fmax = limits["event_recall"]["min"], limits["false_alerts_per_h"]["max"]
    recall = lambda r: r["metrics"].get("event_recall")  # noqa: E731
    false = lambda r: r["metrics"].get("false_alerts_per_h")  # noqa: E731
    low = [r for r in rows if recall(r) is None or recall(r) < rmin]
    noisy = [r for r in rows if false(r) is not None and false(r) > fmax]
    lines = [f"always deploy: ships all {len(rows)} builds; {len(low)} would miss more than "
             f"{100 * (1 - rmin):.0f}% of arrivals, {len(noisy)} would raise more than {fmax:g} false alert/h"]
    for pol in POLICIES[1:]:
        shipped = [r for r in rows if r[pol].startswith(("ships", PASS))]
        detail = ", ".join(f"{name(r)} (recall {recall(r):.2f}, {false(r):.2g} false/h)" for r in shipped)
        line = f"{pol}: ships {len(shipped)}" + (f": {detail}" if shipped else "")
        if pol == "rules + Glimmer":
            line += f"; {sum(r['audited'] for r in rows)} audited, {sum(r['vetoed'] for r in rows)} vetoed"
        lines.append(line)
    return lines


def show(rows, lines):
    fmt = lambda v: "-" if v is None else f"{v:.3g}" if isinstance(v, float) else str(v)  # noqa: E731
    print("| run | candidate | backend | recall | false/h | " + " | ".join(POLICIES) + " |")
    print("|" + " --- |" * (5 + len(POLICIES)))
    for r in rows:
        print(f"| {r['run']} | {r['candidate']} | {r['backend']} | {fmt(r['metrics'].get('event_recall'))} | "
              f"{fmt(r['metrics'].get('false_alerts_per_h'))} | " + " | ".join(r[p] for p in POLICIES) + " |")
    print()
    for line in lines:
        print(line)


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--runs", help="comma-separated run names (default: every run under runs/records)")
    ap.add_argument("--policy", default=str(ROOT / "policies/package_entrance.toml"))
    ap.add_argument("--records", default=str(ROOT / "runs/records"),
                    help="records directory (results/records replays the committed snapshot)")
    ap.add_argument("--audits", default=str(ROOT / "runs/audits"))
    ap.add_argument("--now", help="ISO time to judge staleness against (replays)")
    ap.add_argument("--all", action="store_true", help="also builds measured on another eval set")
    ap.add_argument("--json", help="also write the rows to this file")
    a = ap.parse_args()
    now = _parse_time(a.now) if a.now else None
    if a.now and now is None:
        sys.exit(f"bad --now {a.now!r} (need ISO time with timezone)")
    records = Path(a.records)
    runs = a.runs.split(",") if a.runs else sorted(p.name for p in records.iterdir()
                                                   if p.is_dir() and any(p.rglob("*.json")))
    policy = load_policy(a.policy)
    rows, left_out = builds(policy, runs, now, a.all, records)
    rows = compare(rows, load_audits(a.audits), policy["thresholds"])
    lines = summary(rows, policy["thresholds"])
    if left_out:
        lines.append(f"left out: {left_out} builds measured on another eval set (--all shows them)")
    show(rows, lines)
    if a.json:
        Path(a.json).write_text(json.dumps({"rows": rows, "summary": lines}, indent=2) + "\n")


if __name__ == "__main__":
    main()
