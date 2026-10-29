#!/usr/bin/env python3
"""Mark ground-truth arrivals in a recorded session, in the browser.

Serves one session on localhost: scrub the frames, mark the frame where the hand
lets go of the package (arrive) and where it is taken away (remove), drag a box
around the resting package (for the audit's size slice) and tag the scene (e.g.
occluded). Saves data/gt/<session>.json in the format edgegate/events.py reads.

Suggestions: if the session has SAM labels (labels_sam.jsonl), the policy's
arrival rule is run on the teacher boxes and the alert times are listed as
places to look. They only jump the view. The human marks every event by hand,
and should still scrub the whole session: an arrival the teacher missed has no
suggestion.

Usage:  python3 scripts/mark_gt.py data/sessions/e01 [--port 8765]
        then open http://127.0.0.1:8765 (from another machine: ssh -L 8765:127.0.0.1:8765 gx10)
Keys:   left/right 1 frame, shift 1 s, A arrive (new event at this frame), R remove
        (selected event), drag on the image = box of the selected event, O toggle
        "occluded", Delete remove event, N/P next/previous suggestion, S save,
        B brighten (display only, a gamma curve for the low-light sessions; the
        frames and the marks are unchanged)
"""
import argparse
import csv
import json
import os
import sys
from datetime import datetime, timezone
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from edgegate.events import alerts  # noqa: E402
from edgegate.gate import load_policy  # noqa: E402


def validate(gt, duration):
    """Problems with a GT dict, as strings; empty means it can be saved."""
    problems = []
    last = -1.0
    for i, e in enumerate(gt.get("events", [])):
        a, r = e.get("arrive"), e.get("remove")
        if not isinstance(a, (int, float)) or not 0 <= a <= duration:
            problems.append(f"event {i + 1}: arrive {a!r} outside 0..{duration:.1f}")
            continue
        if a < last:
            problems.append(f"event {i + 1}: arrivals must be in time order")
        last = a
        if r is not None and (not isinstance(r, (int, float)) or not a < r <= duration):
            problems.append(f"event {i + 1}: remove {r!r} must be after arrive and within the session")
        b = e.get("box")
        if b is not None and not (len(b) == 4 and 0 <= b[0] < b[2] <= 1 and 0 <= b[1] < b[3] <= 1):
            problems.append(f"event {i + 1}: bad box {b!r}")
    return problems


def suggestions(sess, policy):
    """Alert times of the arrival rule run on the SAM teacher boxes, if there are labels."""
    path = sess / "labels_sam.jsonl"
    if not path.exists():
        return []
    frames = sorted(((r["t"], r["dets"]) for r in map(json.loads, open(path))), key=lambda f: f[0])
    times, _ = alerts(frames, policy["events"])
    return [round(t, 3) for t in times]


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("session", help="data/sessions/<name>")
    ap.add_argument("--gt-dir", default=str(ROOT / "data/gt"))
    ap.add_argument("--policy", default=str(ROOT / "policies/package_entrance.toml"))
    ap.add_argument("--port", type=int, default=8765)
    a = ap.parse_args()
    sess = Path(a.session)
    name = sess.name
    policy = load_policy(a.policy)
    frames = [(int(r["seq"]), float(r["t"])) for r in csv.DictReader(open(sess / "frames.csv"))]
    meta = json.loads((sess / "meta.json").read_text()) if (sess / "meta.json").exists() else {}
    duration = float(meta.get("duration_s") or frames[-1][1])
    gt_path = Path(a.gt_dir) / f"{name}.json"
    gt = json.loads(gt_path.read_text()) if gt_path.exists() else {"session": name, "duration_s": duration, "events": []}
    sam = {}
    if (sess / "labels_sam.jsonl").exists():
        sam = {r["seq"]: r["dets"] for r in map(json.loads, open(sess / "labels_sam.jsonl"))}
    info = {"session": name, "lighting": meta.get("lighting"), "split": meta.get("split"), "note": meta.get("note"),
            "duration_s": duration, "frames": frames, "gt": gt, "zone": policy["events"]["zone"],
            "suggestions": suggestions(sess, policy), "dwell_s": policy["events"]["dwell_s"],
            "outages": meta.get("outages", [])}

    class Handler(BaseHTTPRequestHandler):
        def log_message(self, *args):
            pass

        def send(self, code, body, ctype="application/json"):
            self.send_response(code)
            self.send_header("Content-Type", ctype)
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def do_GET(self):
            if self.path == "/":
                self.send(200, PAGE.encode(), "text/html; charset=utf-8")
            elif self.path == "/api/session":
                self.send(200, json.dumps(info).encode())
            elif self.path.startswith("/frame/"):
                seq = self.path[len("/frame/"):].split(".")[0]
                p = sess / "frames" / f"{int(seq):06d}.jpg" if seq.isdigit() else None
                if p is None or not p.exists():
                    self.send(404, b"{}")
                else:
                    self.send(200, p.read_bytes(), "image/jpeg")
            elif self.path.startswith("/sam/"):
                seq = self.path[len("/sam/"):]
                self.send(200, json.dumps(sam.get(int(seq), None) if seq.isdigit() else None).encode())
            else:
                self.send(404, b"{}")

        def do_POST(self):
            if self.path != "/api/save":
                return self.send(404, b"{}")
            body = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
            events = sorted(body.get("events", []), key=lambda e: e.get("arrive", 0))
            new = {"session": name, "duration_s": duration, "events": events,
                   "marked_at": datetime.now(timezone.utc).isoformat(timespec="seconds")}
            problems = validate(new, duration)
            if problems:
                return self.send(400, json.dumps({"problems": problems}).encode())
            gt_path.parent.mkdir(parents=True, exist_ok=True)
            tmp = gt_path.with_suffix(".tmp")
            tmp.write_text(json.dumps(new, indent=2) + "\n")
            os.replace(tmp, gt_path)  # never a half-written GT file
            info["gt"] = new
            self.send(200, json.dumps({"saved": str(gt_path), "events": len(events)}).encode())

    print(f"{name}: {len(frames)} frames, {duration:.0f} s, {len(gt['events'])} marked, "
          f"{len(info['suggestions'])} suggestions -> http://127.0.0.1:{a.port}")
    ThreadingHTTPServer(("127.0.0.1", a.port), Handler).serve_forever()


PAGE = r"""<!doctype html>
<html><head><meta charset="utf-8"><title>EdgeGate GT marking</title>
<style>
body{margin:0;font:14px system-ui,sans-serif;background:#111;color:#ddd;display:flex;height:100vh}
#main{flex:1;display:flex;flex-direction:column;padding:8px;min-width:0}
#wrap{position:relative;flex:1;min-height:0}
canvas{width:100%;height:100%;object-fit:contain;cursor:crosshair}
#bar{display:flex;gap:8px;align-items:center;padding:6px 0}
#slider{flex:1}
#side{width:340px;padding:8px;background:#1b1b1b;overflow:auto}
.ev{padding:6px;margin:4px 0;border:1px solid #333;border-radius:4px;cursor:pointer}
.ev.sel{border-color:#4af}
button{background:#333;color:#ddd;border:1px solid #555;border-radius:4px;padding:4px 8px;cursor:pointer}
#msg{min-height:1.4em;color:#fc6}
small{color:#999}
</style></head><body>
<div id="main">
  <div id="wrap"><canvas id="cv"></canvas></div>
  <div id="bar"><span id="time"></span><input id="slider" type="range" min="0" value="0">
    <label><input id="showsam" type="checkbox" checked> SAM boxes</label>
    <label>brighten <select id="bright"><option value="1">off</option><option value="1.8">γ 1.8</option>
      <option value="2.5">γ 2.5</option><option value="3.5">γ 3.5</option></select></label></div>
</div>
<div id="side">
  <b id="title"></b><div><small id="meta"></small></div>
  <p><small>A arrive · R remove · drag box · O occluded · Del delete · N/P suggestion · S save · B brighten · ←/→ frame, shift 1 s</small></p>
  <div><button onclick="save()">Save (S)</button> <span id="dirty"></span></div>
  <div id="msg"></div>
  <h4>Events</h4><div id="events"></div>
  <h4>Suggestions <small>(teacher boxes, may miss events)</small></h4><div id="sugg"></div>
</div>
<script>
let S, i = 0, sel = -1, events = [], dirty = false, img = new Image(), sam = null, drag = null, lut = null;
const cv = document.getElementById('cv'), cx = cv.getContext('2d'), slider = document.getElementById('slider');
const fmt = t => t == null ? '—' : (Math.floor(t / 60) + ':' + (t % 60).toFixed(2).padStart(5, '0'));
const idxAt = t => { let k = S.frames.findIndex(f => f[1] >= t); return k < 0 ? S.frames.length - 1 : k; };
fetch('/api/session').then(r => r.json()).then(d => {
  S = d; events = JSON.parse(JSON.stringify(d.gt.events || []));
  slider.max = S.frames.length - 1;
  document.getElementById('title').textContent = S.session;
  document.getElementById('meta').textContent = `${S.split || '?'} · ${S.lighting || '?'} · ${fmt(S.duration_s)} · ${S.note || ''}` +
    (S.outages.length ? ` · ${S.outages.length} camera outage(s)` : '');
  go(0); list();
});
function go(k) {
  i = Math.max(0, Math.min(S.frames.length - 1, k)); slider.value = i;
  const [seq, t] = S.frames[i];
  document.getElementById('time').textContent = `${fmt(t)}  #${seq}`;
  img.onload = draw; img.src = '/frame/' + seq;
  sam = null;
  if (document.getElementById('showsam').checked) fetch('/sam/' + seq).then(r => r.json()).then(d => { sam = d; draw(); });
}
function draw() {
  if (!img.naturalWidth) return;
  cv.width = img.naturalWidth; cv.height = img.naturalHeight;
  const W = cv.width, H = cv.height, t = S.frames[i][1];
  cx.drawImage(img, 0, 0);
  if (lut) { const px = cx.getImageData(0, 0, W, H), a = px.data;
    for (let k = 0; k < a.length; k += 4) { a[k] = lut[a[k]]; a[k + 1] = lut[a[k + 1]]; a[k + 2] = lut[a[k + 2]]; }
    cx.putImageData(px, 0, 0); }
  cx.lineWidth = 2; cx.strokeStyle = '#ff0'; cx.setLineDash([8, 6]); cx.beginPath();
  S.zone.forEach(([x, y], k) => k ? cx.lineTo(x * W, y * H) : cx.moveTo(x * W, y * H)); cx.closePath(); cx.stroke();
  cx.setLineDash([]);
  if (sam) for (const d of sam) { cx.strokeStyle = d[5] === 'person' ? '#f6a' : '#6cf';
    cx.strokeRect(d[0] * W, d[1] * H, (d[2] - d[0]) * W, (d[3] - d[1]) * H); }
  events.forEach((e, k) => { const live = e.arrive <= t && (e.remove == null || t <= e.remove);
    if (e.box && (live || k === sel)) { cx.lineWidth = k === sel ? 4 : 2; cx.strokeStyle = '#4f4';
      cx.strokeRect(e.box[0] * W, e.box[1] * H, (e.box[2] - e.box[0]) * W, (e.box[3] - e.box[1]) * H); } });
  if (drag) { cx.strokeStyle = '#4f4'; cx.strokeRect(drag[0], drag[1], drag[2] - drag[0], drag[3] - drag[1]); }
}
function list() {
  document.getElementById('events').innerHTML = events.map((e, k) =>
    `<div class="ev ${k === sel ? 'sel' : ''}" onclick="pick(${k})">#${k + 1} arrive ${fmt(e.arrive)} · remove ${fmt(e.remove)}<br>
     <small>${e.box ? 'box ✓' : 'no box'} · tags: ${(e.tags || []).join(', ') || '—'}</small></div>`).join('') || '<small>none</small>';
  document.getElementById('sugg').innerHTML = S.suggestions.map(t =>
    `<div class="ev" onclick="go(idxAt(${t} - ${S.dwell_s} - 1))">alert at ${fmt(t)} <small>(release ≈ before ${fmt(t - S.dwell_s)})</small></div>`).join('') || '<small>none</small>';
  document.getElementById('dirty').textContent = dirty ? 'unsaved' : '';
}
function pick(k) { sel = k; go(idxAt(events[k].arrive)); list(); }
function touch() { dirty = true; list(); draw(); }
function save() {
  fetch('/api/save', {method: 'POST', body: JSON.stringify({events})}).then(r => r.json()).then(d => {
    document.getElementById('msg').textContent = d.problems ? d.problems.join('; ') : `saved ${d.events} events`;
    if (!d.problems) { dirty = false; list(); } });
}
slider.oninput = () => go(+slider.value);
document.getElementById('showsam').onchange = () => go(i);
const bright = document.getElementById('bright');
bright.onchange = () => { const g = +bright.value;  // display only: a gamma curve, the frame file is untouched
  lut = g > 1 ? Uint8ClampedArray.from({length: 256}, (_, v) => 255 * Math.pow(v / 255, 1 / g)) : null; draw(); };
document.onkeydown = ev => {
  const t = S.frames[i][1], step = ev.shiftKey ? 15 : 1;
  if (ev.key === 'ArrowRight') go(i + step);
  else if (ev.key === 'ArrowLeft') go(i - step);
  else if (ev.key === 'a' || ev.key === 'A') { events.push({arrive: t, remove: null, tags: []});
    events.sort((x, y) => x.arrive - y.arrive); sel = events.findIndex(e => e.arrive === t); touch(); }
  else if ((ev.key === 'r' || ev.key === 'R') && sel >= 0) { events[sel].remove = t; touch(); }
  else if ((ev.key === 'o' || ev.key === 'O') && sel >= 0) { const g = events[sel].tags = events[sel].tags || [];
    const k = g.indexOf('occluded'); k < 0 ? g.push('occluded') : g.splice(k, 1); touch(); }
  else if (ev.key === 'Delete' && sel >= 0 && confirm('Delete event #' + (sel + 1) + '?')) { events.splice(sel, 1); sel = -1; touch(); }
  else if (ev.key === 'n' || ev.key === 'N') { const s = S.suggestions.find(x => x - S.dwell_s - 1 > t + 0.01); if (s != null) go(idxAt(s - S.dwell_s - 1)); }
  else if (ev.key === 'p' || ev.key === 'P') { const s = [...S.suggestions].reverse().find(x => x - S.dwell_s - 1 < t - 0.01); if (s != null) go(idxAt(s - S.dwell_s - 1)); }
  else if (ev.key === 's' || ev.key === 'S') save();
  else if (ev.key === 'b' || ev.key === 'B') { bright.selectedIndex = (bright.selectedIndex + 1) % bright.options.length; bright.onchange(); }
  else return;
  ev.preventDefault();
};
const pos = ev => { const r = cv.getBoundingClientRect(), sc = Math.min(r.width / cv.width, r.height / cv.height);
  const ox = (r.width - cv.width * sc) / 2, oy = (r.height - cv.height * sc) / 2;
  return [(ev.clientX - r.left - ox) / sc, (ev.clientY - r.top - oy) / sc]; };
cv.onmousedown = ev => { if (sel < 0) { document.getElementById('msg').textContent = 'select or add an event first (A)'; return; }
  const [x, y] = pos(ev); drag = [x, y, x, y]; };
cv.onmousemove = ev => { if (drag) { const [x, y] = pos(ev); drag[2] = x; drag[3] = y; draw(); } };
cv.onmouseup = () => { if (!drag) return;
  const [x1, x2] = [drag[0], drag[2]].sort((a, b) => a - b), [y1, y2] = [drag[1], drag[3]].sort((a, b) => a - b);
  const c = v => Math.round(Math.max(0, Math.min(1, v)) * 1e4) / 1e4;
  if (x2 - x1 > 4 && y2 - y1 > 4) { events[sel].box = [c(x1 / cv.width), c(y1 / cv.height), c(x2 / cv.width), c(y2 / cv.height)]; touch(); }
  drag = null; draw(); };
window.onbeforeunload = () => dirty ? 'unsaved marks' : undefined;
</script></body></html>
"""

if __name__ == "__main__":
    main()
