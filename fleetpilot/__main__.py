"""Command line: build the data and models, ask the copilot, run the evaluation, check the audit log.

python -m fleetpilot build
python -m fleetpilot ask "Which customers in Portugal should I call this week?"
python -m fleetpilot eval [--live] [--out eval_report.json]
python -m fleetpilot audit-verify
python -m fleetpilot record          # save real runs for the app's demo mode (needs a key)
"""

import argparse
import json
import os
import sys

from . import config as C
from .logs import setup_logging


def main(argv=None) -> int:
    p = argparse.ArgumentParser(prog="fleetpilot")
    sub = p.add_subparsers(dest="cmd", required=True)
    sub.add_parser("build")
    a = sub.add_parser("ask")
    a.add_argument("question")
    a.add_argument("--scripted", action="store_true", help="use the scripted test model instead of Claude")
    e = sub.add_parser("eval")
    e.add_argument("--live", action="store_true", help="use real Claude (costs a few cents)")
    e.add_argument("--out", default=None)
    sub.add_parser("audit-verify")
    sub.add_parser("record", help="run the demo questions with real Claude and save them for the app's demo mode")
    args = p.parse_args(argv)
    setup_logging()
    s = C.settings()

    if args.cmd == "build":
        from .pipeline import build

        m = build(s)
        print(json.dumps({k: {m2: round(v, 3) for m2, v in d.items()} for k, d in m["test"].items()}, indent=1))
        return 0

    from .agent.scripted import ScriptedAnthropic
    from .service import copilot

    if args.cmd == "ask":
        cp = copilot(s, client=ScriptedAnthropic() if args.scripted else None, api_key=os.getenv("ANTHROPIC_API_KEY"))
        r = cp.ask(args.question)
        print(r.answer)
        print(
            f"\n[check: {'ok' if r.check.get('ok') else 'FAILED ' + str(r.check.get('unverified'))} | "
            f"tools: {', '.join(c['name'] for c in r.tool_calls) or '-'} | cost: {r.cost['eur']} EUR | {r.seconds}s]"
        )
        return 0

    if args.cmd == "eval":
        from .evaluation import run

        live = args.live
        cp = copilot(s, client=None if live else ScriptedAnthropic(), api_key=os.getenv("ANTHROPIC_API_KEY"))
        rep = run(cp, s.crm_db)
        summary = {k: v for k, v in rep.items() if k != "rows"}
        print(json.dumps(summary, indent=1))
        if args.out:
            with open(args.out, "w") as f:
                json.dump(rep, f, indent=1, default=str)
        return 0

    if args.cmd == "record":
        from .recordings import QUESTIONS, save

        cp = copilot(s, api_key=os.getenv("ANTHROPIC_API_KEY"))
        for q in QUESTIONS:
            r = cp.ask(q)
            print(f"saved {save(r)}  (check ok: {r.check.get('ok')}, cost: {r.cost['eur']} EUR)")
        return 0

    if args.cmd == "audit-verify":
        from .audit import AuditLog

        res = AuditLog(s.audit_db).verify()
        print(json.dumps(res))
        return 0 if res["ok"] else 1
    return 1


if __name__ == "__main__":
    sys.exit(main())
