"""Catalogue job CLI.

python -m ds_research_agent.catalogue --config config/local.yaml build
python -m ds_research_agent.catalogue --config config/local.yaml index
"""

from __future__ import annotations

import argparse
import sys
from collections import Counter
from pathlib import Path

from ds_research_agent.catalogue.index_job import run_index
from ds_research_agent.catalogue.profiler import build_catalogue
from ds_research_agent.config import load_settings


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(prog="python -m ds_research_agent.catalogue")
    ap.add_argument("--config", type=Path, required=True)
    ap.add_argument("command", choices=["build", "index"])
    args = ap.parse_args(argv)
    settings = load_settings(args.config)

    if args.command == "build":
        m = build_catalogue(settings.catalogue)
        status = Counter(e.parse_status for e in m.entries)
        print(f"{len(m.entries)} cards ({dict(sorted(status.items()))}); {len(m.skipped)} skipped")
        for e in m.entries:
            if e.parse_status == "error":
                print(f"  error: {e.file_path}: {e.parse_error}")
        for sk in m.skipped:
            print(f"  skipped: {sk.path}: {sk.reason}")
    else:
        g = run_index(settings.catalogue, settings.rag_service)
        print(g.model_dump_json(indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
