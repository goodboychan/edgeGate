#!/usr/bin/env python3
"""Audit one candidate with Glimmer: rule decisions first, then veto-only extra checks.

Starts the MCP audit server (edgegate/audit_mcp.py) for the candidate, gives its
tools to Glimmer (vLLM, OpenAI-compatible tool calling) and lets it ask for
coverage and slice metrics. Accepted propose_hold calls become vetoes, which can
only turn a rule PASS into HOLD. If the audit does not finish (server error, no
final answer within --max-turns), every backend the rules passed is held with
check "audit": an unfinished audit never lets a candidate through unchecked.

Each run writes runs/audits/<label>_<time>_r<i>.json: rule and final verdicts,
vetoes with their evidence, rejected proposals, every tool call, Glimmer's final
explanation, wall time and tokens. --repeats N audits the same candidate N times
from scratch, for the repeat-consistency numbers.

Usage:  .venv-audit/bin/python scripts/audit.py POLICY RECORD.json... --model-sha SHA \
            --artifact trt=build/student.engine --artifact hailo8=build/student.hef \
            [--repeats 5] [--reasoning medium] [--label fp8] [--url http://127.0.0.1:8001/v1/chat/completions]
"""
import argparse
import asyncio
import json
import os
import sys
import time
import urllib.request
from datetime import datetime, timezone
from pathlib import Path

from mcp import Client, StdioServerParameters

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from edgegate.audit import artifact_sha  # noqa: E402
from edgegate.gate import PASS, _parse_time, apply_veto, decide, load_policy, load_records  # noqa: E402

SYSTEM = """Reasoning strength: {reasoning}

You audit a deployment decision for an edge perception skill.
Task: {instruction}
Deployment: {context}

A rule engine has already decided PASS or HOLD for each device from real measurements on an eval set. You cannot approve anything the rules held. Your only power is to propose a HOLD for a device the rules passed, when the audit tools show a risk that the summary metrics hide.

How to audit:
- Start with gate_decisions. Then check what the summary could hide: eval_coverage, and slice_metrics for each device the rules passed (by lighting, size, tag or session), compared with the reference on the same slices.
- Propose a HOLD only through propose_hold, citing the evidence ids of the tool results that show the problem. Do not hold on a hunch or for a difference within what a handful of events can show by chance.
- A slice that is weak on the reference too is a data or model gap, not a device gap; it still matters if the deployment depends on that slice.
- If nothing shows a problem, propose nothing. A HOLD on a good candidate is an error too.
- Finish with one short paragraph per device: the verdict after your audit and why, citing evidence ids."""


def chat(url, messages, tools, a):
    body = {"model": "muse-glimmer", "messages": messages, "tools": tools, "temperature": a.temperature,
            "top_p": 0.95, "top_k": 64, "max_tokens": a.max_tokens}
    headers = {"Content-Type": "application/json"}
    if os.environ.get("GLIMMER_API_KEY"):
        headers["Authorization"] = "Bearer " + os.environ["GLIMMER_API_KEY"]
    req = urllib.request.Request(url, json.dumps(body).encode(), headers)
    with urllib.request.urlopen(req, timeout=1800) as r:
        return json.load(r)


async def audit_once(a, policy, server_args):
    """One audit from a fresh server; returns the run dict (vetoes, trace, answer, error)."""
    run = {"vetoes": {}, "rejected": [], "tool_calls": [], "turns": [], "answer": None, "error": None,
           "usage": {"prompt_tokens": 0, "completion_tokens": 0}}
    params = StdioServerParameters(command=sys.executable, args=["-m", "edgegate.audit_mcp", *server_args],
                                   cwd=str(ROOT))
    messages = [{"role": "system", "content": SYSTEM.format(
                    reasoning=a.reasoning, instruction=policy["task"].get("instruction", policy["task"]["name"]),
                    context=policy["task"].get("context", "not given"))},
                {"role": "user", "content": "Audit this candidate."}]
    try:
        async with Client(params) as client:
            tools = [{"type": "function", "function": {"name": t.name, "description": t.description or "",
                                                       "parameters": t.input_schema}}
                     for t in (await client.list_tools()).tools]
            for _ in range(a.max_turns):
                t0 = time.perf_counter()
                resp = await asyncio.to_thread(chat, a.url, messages, tools, a)
                msg = resp["choices"][0]["message"]
                for k in run["usage"]:
                    run["usage"][k] += (resp.get("usage") or {}).get(k, 0)
                calls = msg.get("tool_calls") or []
                run["turns"].append({"s": round(time.perf_counter() - t0, 1),
                                     "reasoning": (msg.get("reasoning_content") or msg.get("reasoning") or "")[:4000],
                                     "content": msg.get("content"), "tool_calls": len(calls)})
                if not calls:
                    run["answer"] = msg.get("content")
                    break
                messages.append({"role": "assistant", "content": msg.get("content") or "", "tool_calls": calls})
                for c in calls:
                    name = c["function"]["name"]
                    try:
                        args = json.loads(c["function"]["arguments"] or "{}")
                        res = await client.call_tool(name, args)
                        text = "".join(getattr(b, "text", "") for b in res.content)
                    except Exception as e:  # noqa: BLE001 - a bad call goes back to Glimmer as an error
                        args, text = c["function"].get("arguments"), json.dumps({"error": f"{type(e).__name__}: {e}"})
                    run["tool_calls"].append({"name": name, "args": args, "result": text[:2000]})
                    if name == "propose_hold" and isinstance(args, dict):
                        out = json.loads(text) if text.startswith("{") else {}
                        if out.get("accepted"):
                            run["vetoes"][args["backend"]] = {"reason": args.get("reason"),
                                                              "evidence": args.get("evidence")}
                        else:
                            run["rejected"].append({**args, "problem": out.get("problem") or text[:300]})
                    messages.append({"role": "tool", "tool_call_id": c["id"], "content": text})
            else:
                run["error"] = f"no final answer within {a.max_turns} turns"
    except Exception as e:  # noqa: BLE001 - recorded, and the candidate is held below
        run["error"] = f"{type(e).__name__}: {e}"[:500]
    return run


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("policy")
    ap.add_argument("records", nargs="+")
    ap.add_argument("--model-sha", required=True)
    ap.add_argument("--artifact", action="append", default=[], metavar="BACKEND=PATH_OR_SHA")
    ap.add_argument("--now", help="ISO time to judge staleness against (replays)")
    ap.add_argument("--repeats", type=int, default=1)
    ap.add_argument("--reasoning", default="medium", choices=["low", "medium", "high", "xhigh"])
    ap.add_argument("--temperature", type=float, default=1.0, help="model card default")
    ap.add_argument("--max-tokens", type=int, default=8192, help="per turn, reasoning included")
    ap.add_argument("--max-turns", type=int, default=16)
    ap.add_argument("--url", default="http://127.0.0.1:8001/v1/chat/completions")
    ap.add_argument("--label", default="glimmer", help="e.g. fp8 or bf16, or the case name")
    ap.add_argument("--out", default=str(ROOT / "runs/audits"))
    a = ap.parse_args()

    policy = load_policy(a.policy)
    records = [str(Path(p).resolve()) for p in a.records]
    artifacts = {}
    for spec in a.artifact:
        backend, _, v = spec.partition("=")
        artifacts[backend] = artifact_sha(v)
    now = _parse_time(a.now) if a.now else None
    server_args = [str(Path(a.policy).resolve()), *records, "--model-sha", a.model_sha,
                   *[x for b, s in artifacts.items() for x in ("--artifact", f"{b}={s}")],
                   *(["--now", a.now] if a.now else [])]
    out_dir = Path(a.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    for i in range(a.repeats):
        t0 = time.perf_counter()
        run = asyncio.run(audit_once(a, policy, server_args))
        rules = decide(policy, load_records(records), a.model_sha, artifacts, now)
        rule_view = [{"backend": d.backend, "verdict": d.verdict, "reasons": list(d.reasons)} for d in rules]
        if run["error"]:
            final = apply_veto(rules, {d.backend: f"audit did not finish: {run['error']}"
                                       for d in rules if d.verdict == PASS}, by="audit-failure")
        else:
            final = apply_veto(rules, {b: v["reason"] for b, v in run["vetoes"].items()})
        rec = {"schema": 1, "label": a.label, "repeat": i, "model_sha256": a.model_sha, "artifacts": artifacts,
               "records": records, "reasoning": a.reasoning, "temperature": a.temperature, "url": a.url,
               "audited_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
               "wall_s": round(time.perf_counter() - t0, 1), "rules": rule_view,
               "final": [{"backend": d.backend, "verdict": d.verdict, "reasons": list(d.reasons),
                          "vetoed_by": d.vetoed_by} for d in final], **run}
        path = out_dir / f"{a.label}_{stamp}_r{i}.json"
        path.write_text(json.dumps(rec, indent=2) + "\n")
        print(f"{path}: {rec['wall_s']} s, {run['usage']['completion_tokens']} tokens out, "
              f"{len(run['tool_calls'])} tool calls" + (f", ERROR {run['error']}" if run["error"] else ""))
        for r, f in zip(rule_view, rec["final"]):
            print(f"  {f['backend']:12s} rules {r['verdict']:4s} -> {f['verdict']}"
                  + (f"  ({f['vetoed_by']}: {f['reasons'][-1][1]})" if f["vetoed_by"] else ""))


if __name__ == "__main__":
    main()
