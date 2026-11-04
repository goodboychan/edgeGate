#!/usr/bin/env python3
"""Write one static HTML page from the logged runs: verdicts, metrics, why held, audits, approvals.

Everything on the page is read from runs/ at the time it is written: the rule verdicts
are decided again from the records (so a record older than max_age_hours shows as stale),
the audits come from runs/audits/, the approvals from runs/approvals/. No server, no
scripts on the page, nothing loaded from the network.

Usage:  python3 scripts/dashboard.py [--runs s1,s3] [--out runs/dashboard.html]
"""
import argparse
import html
import json
import math
import sys
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))
from edgegate.gate import PASS, eval_config_sha256, load_policy  # noqa: E402
from compare_gates import POLICIES, builds, compare, load_audits, summary  # noqa: E402
from measure_candidates import CANDIDATES, COLUMNS, judge  # noqa: E402

DIGITS = {"event_recall": 3, "false_alerts_per_h": 2, "duplicate_alerts": 0, "detection_delay_s_p95": 1,
          "latency_ms_p95": 1, "peak_mem_mb": 0}


def esc(s):
    return html.escape("" if s is None else str(s))


def num(name, v):
    return "-" if v is None else f"{v:.{DIGITS[name]}f}"


def pill(verdict):
    if verdict == PASS:
        return '<span class="pill pass">&#10003; PASS</span>'
    if verdict == "-":
        return '<span class="pill ref">reference</span>'
    return '<span class="pill hold">&#10007; HOLD</span>'


def chart(rows, limits):
    """Scatter of recall against false alerts per hour, with the pass region drawn in."""
    W, H, L, R, T, B = 720, 400, 56, 16, 28, 48
    pw, ph = W - L - R, H - T - B
    ys = [r["metrics"]["false_alerts_per_h"] for r in rows if r["metrics"].get("false_alerts_per_h") is not None]
    ymax = max(2, math.ceil(max(ys, default=1) + 0.5))
    rmin, fmax = limits["event_recall"]["min"], limits["false_alerts_per_h"]["max"]
    X = lambda v: L + v * pw  # noqa: E731
    Y = lambda v: T + ph - v / ymax * ph  # noqa: E731
    out = [f'<svg class="chart" viewBox="0 0 {W} {H}" role="img" aria-label="Event recall against false alerts '
           f'per hour for every build measured on the current eval set; the pass region is shaded">']
    for i in range(6):
        v = i / 5
        out.append(f'<line class="grid" x1="{X(v):.1f}" y1="{T}" x2="{X(v):.1f}" y2="{T + ph}"/>'
                   f'<text class="tick" x="{X(v):.1f}" y="{T + ph + 18}" text-anchor="middle">{v:g}</text>')
    for v in range(0, ymax + 1):
        out.append(f'<line class="grid" x1="{L}" y1="{Y(v):.1f}" x2="{L + pw}" y2="{Y(v):.1f}"/>'
                   f'<text class="tick" x="{L - 8}" y="{Y(v) + 4:.1f}" text-anchor="end">{v}</text>')
    out.append(f'<rect class="region" x="{X(rmin):.1f}" y="{Y(fmax):.1f}" width="{X(1) - X(rmin):.1f}" '
               f'height="{Y(0) - Y(fmax):.1f}"/>')
    out.append(f'<text class="label" x="{X(1) - 6:.1f}" y="{Y(fmax) - 8:.1f}" text-anchor="end">'
               f'PASS region: recall &#8805; {rmin:g}, &#8804; {fmax:g} false alert/h</text>')
    out.append(f'<line class="axis" x1="{L}" y1="{T + ph}" x2="{L + pw}" y2="{T + ph}"/>'
               f'<line class="axis" x1="{L}" y1="{T}" x2="{L}" y2="{T + ph}"/>')
    out.append(f'<text class="title" x="{L + pw / 2:.1f}" y="{H - 8}" text-anchor="middle">event recall</text>'
               f'<text class="title" transform="translate(14 {T + ph / 2:.1f}) rotate(-90)" text-anchor="middle">'
               f'false alerts per hour</text>')
    clean = [r for r in rows if r["candidate"] == "clean" and r["metrics"].get("event_recall") is not None]
    quiet = [r for r in rows if r["metrics"].get("false_alerts_per_h") is not None]
    labeled = set()
    if clean:
        labeled.add(id(max(clean, key=lambda r: r["metrics"]["event_recall"])))
    if quiet:
        labeled.add(id(min(quiet, key=lambda r: (r["metrics"]["false_alerts_per_h"], -(r["metrics"]["event_recall"] or 0)))))
    for r in rows:
        m = r["metrics"]
        if m.get("event_recall") is None or m.get("false_alerts_per_h") is None:
            continue
        x, y = X(m["event_recall"]), Y(min(m["false_alerts_per_h"], ymax))
        kind = "clean" if r["candidate"] == "clean" else "fault"
        name = f"{r['run']} {r['candidate']} {r['backend']}"
        tip = f"{name}: recall {m['event_recall']:.3f}, {m['false_alerts_per_h']:.2f} false/h, {r['verdict']}"
        shape = (f'<circle class="dot {kind}" cx="{x:.1f}" cy="{y:.1f}" r="6"/>' if r["backend"] == "trt" else
                 f'<rect class="dot {kind}" x="{x - 6:.1f}" y="{y - 6:.1f}" width="12" height="12" '
                 f'transform="rotate(45 {x:.1f} {y:.1f})"/>')
        out.append(f'<g>{shape}<title>{esc(tip)}</title></g>')
        if id(r) in labeled:
            anchor, dx = ("end", -10) if m["event_recall"] > 0.75 else ("start", 10)
            out.append(f'<text class="label" x="{x + dx:.1f}" y="{y + 4:.1f}" text-anchor="{anchor}">{esc(name)}</text>')
    out.append("</svg>")
    return "".join(out)


def reasons_html(reasons):
    return "<br>".join(esc(x) for x in reasons) if reasons else ""


def audits_html(path):
    recs = sorted((json.loads(p.read_text()) for p in Path(path).glob("*.json")),
                  key=lambda a: a.get("audited_at") or "", reverse=True)
    if not recs:
        return "<p>No audits yet.</p>"
    out = []
    for a in recs:
        rows = "".join(f"<tr><td>{esc(f['backend'])}</td><td>{pill(r['verdict'])}</td><td>{pill(f['verdict'])}"
                       f"{' (' + esc(f['vetoed_by']) + ')' if f.get('vetoed_by') else ''}</td>"
                       f"<td>{reasons_html(': '.join(x) for x in f['reasons'])}</td></tr>"
                       for r, f in zip(a["rules"], a["final"]))
        err = f'<p class="muted">Audit did not finish: {esc(a["error"])}</p>' if a.get("error") else ""
        out.append(f"""<details class="audit"><summary><b>{esc(a['label'])}</b> &middot; {esc(a['audited_at'])}
 &middot; model {esc(a['model_sha256'][:12])} &middot; {len(a['tool_calls'])} tool calls, {a['wall_s']} s
 &middot; {len(a['vetoes'])} veto(es), {len(a['rejected'])} rejected</summary>
<div class="scroll"><table><thead><tr><th>backend</th><th>rules</th><th>after audit</th><th>reasons</th></tr></thead>
<tbody>{rows}</tbody></table></div>{err}
<p class="muted">Glimmer's explanation ({esc(a['reasoning'])} reasoning, temperature {a['temperature']:g}):</p>
<pre>{esc(a.get('answer') or '(none)')}</pre></details>""")
    return "\n".join(out)


def approvals_html(path):
    files = sorted(Path(path).glob("*.json"), reverse=True)
    if not files:
        return ("<p>No approval records: no build has passed the rules, so <code>scripts/deploy.py</code> "
                "has started nothing. There is no override.</p>")
    rows = []
    for p in files:
        a = json.loads(p.read_text())
        rows.append(f"<tr><td>{esc(a['backend'])}</td><td>{esc(a['artifact_file'])}</td><td>{esc(a['approved_at'])}"
                    f"</td><td>{esc(a['device'].get('hostname') or a['device'].get('label'))} &middot; "
                    f"{esc(a['runtime'])}</td><td>{esc(p.name)}</td></tr>")
    return ('<div class="scroll"><table><thead><tr><th>backend</th><th>file</th><th>approved</th><th>device</th>'
            '<th>record</th></tr></thead><tbody>' + "".join(rows) + "</tbody></table></div>")


def page(policy, runs, now, audits_dir, approvals_dir, records=None):
    task, limits = policy["task"], policy["thresholds"]
    records = Path(records) if records else ROOT / "runs/records"
    comparable, left_out = builds(policy, runs, now, records=records)
    comparable = compare(comparable, load_audits(audits_dir), limits, eval_config_sha256(policy))
    lines = summary(comparable, limits)
    approved = sum(r["verdict"] == PASS for r in comparable)
    audited, vetoed = sum(r["audited"] for r in comparable), sum(r["vetoed"] for r in comparable)
    hero = f"{approved} of {len(comparable)}"

    table = []
    for run in runs:
        for r in judge(policy, list(CANDIDATES), ROOT / "build" / run, records / run, now):
            if r["candidate"] != "reference" and not r["measured"]:
                continue
            held = ", ".join(dict.fromkeys(x.split(":")[0] for x in r["reasons"]))
            table.append(f"<tr><td>{esc(run)}</td><td>{esc(r['candidate'])}</td><td>{esc(r['backend'])}</td>"
                         f"<td>{pill(r['verdict'])}</td>"
                         + "".join(f'<td class="n">{num(k, r["metrics"].get(k))}</td>' for k in COLUMNS)
                         + f'<td>{esc(held)}</td><td class="muted">{esc((r.get("measured_at") or "")[:16])}</td>'
                         f'<td class="why">{reasons_html(r["reasons"])}</td></tr>')
    comp = "".join(f"<tr><td>{esc(r['run'])}</td><td>{esc(r['candidate'])}</td><td>{esc(r['backend'])}</td>"
                   f"<td class='n'>{num('event_recall', r['metrics'].get('event_recall'))}</td>"
                   f"<td class='n'>{num('false_alerts_per_h', r['metrics'].get('false_alerts_per_h'))}</td>"
                   + "".join(f"<td>{esc(r[p])}</td>" for p in POLICIES) + "</tr>" for r in comparable)
    thresholds = ", ".join(f"{k} {' and '.join(f'{b} {v:g}' for b, v in lim.items())}" for k, lim in limits.items())
    stamp = (now or datetime.now(timezone.utc)).isoformat(timespec="minutes")
    return f"""<!doctype html>
<html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width, initial-scale=1">
<title>EdgeGate gate dashboard</title>
<style>
:root {{ color-scheme: light; --surface: #fcfcfb; --page: #f9f9f7; --ink: #0b0b0b; --ink-2: #52514e; --muted: #898781;
  --grid: #e1e0d9; --axis: #c3c2b7; --border: rgba(11,11,11,0.10); --clean: #2a78d6; --fault: #eb6834;
  --good: #0ca30c; --region: rgba(12,163,12,0.10); }}
@media (prefers-color-scheme: dark) {{ :root:not([data-theme="light"]) {{ color-scheme: dark; --surface: #1a1a19;
  --page: #0d0d0d; --ink: #f0efec; --ink-2: #c3c2b7; --muted: #898781; --grid: #2c2c2a; --axis: #383835;
  --border: rgba(255,255,255,0.10); --clean: #3987e5; --fault: #d95926; --region: rgba(12,163,12,0.16); }} }}
:root[data-theme="dark"] {{ color-scheme: dark; --surface: #1a1a19; --page: #0d0d0d; --ink: #f0efec; --ink-2: #c3c2b7;
  --muted: #898781; --grid: #2c2c2a; --axis: #383835; --border: rgba(255,255,255,0.10); --clean: #3987e5;
  --fault: #d95926; --region: rgba(12,163,12,0.16); }}
* {{ box-sizing: border-box; }}
body {{ margin: 0; padding: 24px 16px 48px; background: var(--page); color: var(--ink);
  font: 15px/1.45 system-ui, -apple-system, "Segoe UI", sans-serif; }}
main {{ max-width: 1100px; margin: 0 auto; }}
h1 {{ font-size: 24px; margin: 0 0 4px; }} h2 {{ font-size: 18px; margin: 32px 0 8px; }}
p {{ margin: 6px 0; }} .muted {{ color: var(--muted); }} .sub {{ color: var(--ink-2); }}
.hero {{ display: flex; flex-wrap: wrap; gap: 16px; margin: 16px 0; }}
.tile {{ background: var(--surface); border: 1px solid var(--border); border-radius: 8px; padding: 12px 16px; min-width: 180px; }}
.tile .v {{ font-size: 40px; font-weight: 600; line-height: 1.1; }} .tile .l {{ color: var(--ink-2); font-size: 13px; }}
.card {{ background: var(--surface); border: 1px solid var(--border); border-radius: 8px; padding: 12px 16px; }}
.scroll {{ overflow-x: auto; }}
table {{ border-collapse: collapse; width: 100%; font-size: 14px; }}
th, td {{ text-align: left; padding: 6px 8px; border-bottom: 1px solid var(--grid); vertical-align: top; }}
th {{ color: var(--ink-2); font-weight: 600; white-space: nowrap; }}
td.n {{ font-variant-numeric: tabular-nums; text-align: right; white-space: nowrap; }}
td.why {{ font-size: 12px; color: var(--ink-2); min-width: 220px; }}
.pill {{ display: inline-block; padding: 1px 8px; border-radius: 999px; font-size: 12px; font-weight: 600; white-space: nowrap; }}
.pill.hold {{ background: var(--grid); color: var(--ink); }} .pill.pass {{ background: var(--good); color: #fff; }}
.pill.ref {{ border: 1px solid var(--axis); color: var(--ink-2); font-weight: 500; }}
.legend {{ display: flex; flex-wrap: wrap; gap: 16px; font-size: 13px; color: var(--ink-2); margin: 4px 0 8px; align-items: center; }}
.legend span {{ display: inline-flex; align-items: center; gap: 6px; }}
.sw {{ width: 12px; height: 12px; border-radius: 50%; display: inline-block; }}
.chart {{ width: 100%; max-width: 720px; height: auto; display: block; background: var(--surface); border-radius: 8px; }}
.chart .grid {{ stroke: var(--grid); stroke-width: 1; }} .chart .axis {{ stroke: var(--axis); stroke-width: 1; }}
.chart .tick, .chart .label {{ fill: var(--muted); font-size: 12px; }} .chart .title {{ fill: var(--ink-2); font-size: 13px; }}
.chart .label {{ fill: var(--ink-2); }}
.chart .region {{ fill: var(--region); }}
.chart .dot {{ stroke: var(--surface); stroke-width: 2; }} .chart .dot.clean {{ fill: var(--clean); }} .chart .dot.fault {{ fill: var(--fault); }}
pre {{ white-space: pre-wrap; font-size: 13px; background: var(--page); padding: 8px 12px; border-radius: 6px; }}
details.audit {{ margin: 8px 0; }} summary {{ cursor: pointer; }}
code {{ font-size: 13px; }}
</style></head><body><main>
<h1>EdgeGate gate dashboard</h1>
<p class="sub">{esc(task['instruction'])} &middot; as of {esc(stamp)} &middot; eval set {esc(task['eval_set_sha256'][:12])}&hellip;
 &middot; records valid for {task['max_age_hours']} h</p>
<p class="muted">Thresholds: {esc(thresholds)}. Per-backend overrides: {esc(json.dumps(policy.get('backends', {})))}.
 Rules: <code>edgegate/gate.py</code>; Glimmer can only veto a PASS (<code>scripts/audit.py</code>).</p>
<div class="hero">
 <div class="tile"><div class="v">{hero}</div><div class="l">builds approved on the current eval set</div></div>
 <div class="tile"><div class="v">{audited}</div><div class="l">builds audited by Glimmer ({vetoed} vetoed)</div></div>
 <div class="tile"><div class="v">{len(list(Path(approvals_dir).glob('*.json')))}</div><div class="l">approval records (what deploy.py needs to start a runner)</div></div>
</div>

<h2>Where every build sits against the rules</h2>
<div class="legend"><span><i class="sw" style="background: var(--clean)"></i> clean build</span>
<span><i class="sw" style="background: var(--fault)"></i> fault-injected build</span>
<span>&#9679; trt (Jetson)</span><span>&#9670; hailo8 (Pi + Hailo-8)</span></div>
{chart(comparable, limits)}
<p class="muted">Builds measured on the current eval set only ({len(comparable)}; {left_out} older measurements on another eval set are in the table below as HOLD). Delay, latency and memory are checked too; see the table.</p>

<h2>Verdicts</h2>
<div class="scroll"><table><thead><tr><th>run</th><th>candidate</th><th>backend</th><th>verdict</th>
{"".join(f"<th>{esc(v)}</th>" for v in COLUMNS.values())}<th>held for</th><th>measured (UTC)</th><th>why</th></tr></thead>
<tbody>{"".join(table)}</tbody></table></div>

<h2>Always deploy, a false-alert-only gate, the rules, the rules plus Glimmer</h2>
<p class="sub">Same builds, same records, four ways to decide (<code>scripts/compare_gates.py</code>). "Always deploy" is a what-if from the records; a held build is only ever run live as a labeled replay.</p>
<div class="scroll"><table><thead><tr><th>run</th><th>candidate</th><th>backend</th><th>recall</th><th>false/h</th>
{"".join(f"<th>{esc(p)}</th>" for p in POLICIES)}</tr></thead><tbody>{comp}</tbody></table></div>
<ul>{"".join(f"<li>{esc(line)}</li>" for line in lines)}</ul>

<h2>Glimmer audits</h2>
{audits_html(audits_dir)}

<h2>Approval records</h2>
{approvals_html(approvals_dir)}

<p class="muted">Written by <code>scripts/dashboard.py</code> from <code>runs/records</code>, <code>runs/audits</code> and <code>runs/approvals</code>. Nothing on this page is live; run the script again after measuring.</p>
</main></body></html>
"""


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--runs", help="comma-separated run names (default: every run under the records directory)")
    ap.add_argument("--policy", default=str(ROOT / "policies/package_entrance.toml"))
    ap.add_argument("--records", default=str(ROOT / "runs/records"),
                    help="records directory (results/records replays the committed snapshot)")
    ap.add_argument("--audits", default=str(ROOT / "runs/audits"))
    ap.add_argument("--approvals", default=str(ROOT / "runs/approvals"))
    ap.add_argument("--now", help="ISO time to judge staleness against (replays)")
    ap.add_argument("--out", default=str(ROOT / "runs/dashboard.html"))
    a = ap.parse_args()
    from edgegate.gate import _parse_time
    now = _parse_time(a.now) if a.now else None
    if a.now and now is None:
        sys.exit(f"bad --now {a.now!r} (need ISO time with timezone)")
    records = Path(a.records)
    runs = a.runs.split(",") if a.runs else sorted(p.name for p in records.iterdir()
                                                   if p.is_dir() and any(p.rglob("*.json")))
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    Path(a.out).write_text(page(load_policy(a.policy), runs, now, a.audits, a.approvals, records))
    print(a.out)


if __name__ == "__main__":
    main()
