"""MCP server (stdio) exposing the audit tools for one candidate.

One process serves one audit: it computes the rule decisions at start-up and
keeps the evidence and vetoes for that audit only. scripts/audit.py starts it,
hands its tools to Glimmer and reads the vetoes from the propose_hold results.

Usage (normally started by scripts/audit.py):
  .venv-audit/bin/python -m edgegate.audit_mcp POLICY RECORD.json... --model-sha SHA \
      --artifact trt=build/student.engine --artifact hailo8=build/student.hef [--now ISO]
"""
import argparse
import json
import sys

from mcp.server.mcpserver import MCPServer

from edgegate.audit import Audit, artifact_sha
from edgegate.gate import _parse_time, load_policy, load_records


def build(audit):
    server = MCPServer("edgegate-audit")

    @server.tool()
    def gate_decisions() -> str:
        """Rule verdict (PASS/HOLD) and reasons per device, the summary metrics each was
        judged on, the reference backend's metrics and the policy thresholds."""
        return json.dumps(audit.gate_decisions())

    @server.tool()
    def eval_coverage() -> str:
        """What the eval set covers: sessions, hours and marked events by lighting
        (day/evening/night), events by package size (small/medium/large) and by scene tag."""
        return json.dumps(audit.eval_coverage())

    @server.tool()
    def slice_metrics(backend: str, by: str) -> str:
        """On-time recall, late and missed arrivals per slice for one backend, next to the
        reference on the same slices. by: lighting | size | tag | session. Lighting and
        session slices also give hours and false alerts per hour."""
        return json.dumps(audit.slice_metrics(backend, by))

    @server.tool()
    def propose_hold(backend: str, reason: str, evidence: list[str]) -> str:
        """Hold a backend that the rules passed. Only for a risk the tool results show;
        cite their evidence ids (e.g. ["E3", "E5"]). Rejected for a backend the rules
        already hold, or without evidence from this audit about that backend or the eval set."""
        return json.dumps(audit.propose_hold(backend, reason, evidence))

    return server


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("policy")
    ap.add_argument("records", nargs="*")
    ap.add_argument("--model-sha", required=True)
    ap.add_argument("--artifact", action="append", default=[], metavar="BACKEND=PATH_OR_SHA")
    ap.add_argument("--now")
    a = ap.parse_args()
    artifacts = {}
    for spec in a.artifact:
        backend, _, v = spec.partition("=")
        artifacts[backend] = artifact_sha(v)
    now = _parse_time(a.now) if a.now else None
    audit = Audit(load_policy(a.policy), load_records(a.records), a.model_sha, artifacts, now)
    print(f"audit server: {', '.join(f'{b} {d.verdict}' for b, d in audit.decisions.items())}", file=sys.stderr)
    build(audit).run("stdio")


if __name__ == "__main__":
    main()
