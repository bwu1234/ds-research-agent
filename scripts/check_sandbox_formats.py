"""Load every agent-visible KramaBench file inside the sandbox (local only).

    uv run python scripts/check_sandbox_formats.py --config config/local.yaml \
        --out data/measurements/sandbox_formats.json

One sandbox run per domain mounts ``visible/<domain>`` read-only and loads
each file with the reader for its extension, using only the image's
packages. Per-file status goes to ``--out`` (under the ignored ``data/``
tree: error messages can quote data); the console prints aggregate counts.
The run's read audit is compared with the set of files the loader opened.
"""

from __future__ import annotations

import argparse
import json
import time
from collections import Counter
from pathlib import Path

from ds_research_agent.config import load_settings
from ds_research_agent.sandbox import InputMount, SandboxRunner

LOADER = r"""
import json, os, time
import cdflib, geopandas, lxml.etree, numpy as np, pandas as pd, pyogrio
from sgp4.api import Satrec

def text(p):
    try:
        return open(p, encoding="utf-8").read(), "utf-8"
    except UnicodeDecodeError:
        return open(p, encoding="latin-1").read(), "latin-1"

def whitespace_table(p):
    df = pd.read_csv(p, sep=r"\s+", header=None, comment=None, engine="python")
    return f"{df.shape[0]}x{df.shape[1]}"

def load(p, ext):
    if ext == "csv":
        try:
            df = pd.read_csv(p, low_memory=False)
            return "ok", f"{df.shape[0]}x{df.shape[1]}"
        except UnicodeDecodeError:
            df = pd.read_csv(p, low_memory=False, encoding="latin-1")
            return "ok_latin1", f"{df.shape[0]}x{df.shape[1]}"
    if ext == "xlsx":
        sheets = pd.read_excel(p, sheet_name=None)
        return "ok", {k: f"{v.shape[0]}x{v.shape[1]}" for k, v in sheets.items()}
    if ext == "gpkg":
        layers = [l[0] for l in pyogrio.list_layers(p)]
        return "ok", {l: len(geopandas.read_file(p, layer=l)) for l in layers}
    if ext == "npz":
        with np.load(p, allow_pickle=False) as z:
            return "ok", {k: list(z[k].shape) for k in z.files}
    if ext == "cdf":
        cdf = cdflib.CDF(p)
        names = cdf.cdf_info().zVariables
        return "ok", {n: list(np.shape(cdf.varget(n))) for n in names}
    if ext == "json":
        return "ok", type(json.load(open(p))).__name__
    if ext == "html":
        try:
            return "ok", [t.shape[0] for t in pd.read_html(p)]
        except ValueError as e:
            if "No tables found" in str(e):
                return "ok_no_tables", len(text(p)[0])
            raise
    if ext == "hdr":
        return "ok", lxml.etree.parse(p).getroot().tag
    if ext == "tle":
        lines = [l.rstrip() for l in text(p)[0].splitlines() if l.strip()]
        sats = errors = 0
        for a, b in zip(lines, lines[1:]):
            if a.startswith("1 ") and b.startswith("2 "):
                s = Satrec.twoline2rv(a, b)
                e, _, _ = s.sgp4(s.jdsatepoch, s.jdsatepochF)
                sats += 1
                errors += e != 0
        return ("ok" if sats and not errors else "failed"), {"sats": sats, "errors": errors}
    if ext == "sp3":
        body = text(p)[0].splitlines()
        epochs = sum(l.startswith("*") for l in body)
        pos = sum(l.startswith("P") for l in body)
        ok = body and body[0].startswith("#") and epochs and pos
        return ("ok" if ok else "failed"), {"epochs": epochs, "positions": pos}
    if ext in ("dat", "lst", "txt", "text"):
        _, enc = text(p)
        try:
            return "ok" if enc == "utf-8" else "ok_latin1", whitespace_table(p)
        except Exception as e:
            return "ok_text_only", type(e).__name__
    _, enc = text(p)
    return "ok_text_only", enc

results = []
for root, _, files in os.walk("/data"):
    for f in sorted(files):
        p = os.path.join(root, f)
        ext = f.rsplit(".", 1)[-1].lower() if "." in f else "(none)"
        t0 = time.monotonic()
        try:
            status, detail = load(p, ext)
        except Exception as e:
            status, detail = "failed", f"{type(e).__name__}: {str(e)[:300]}"
        results.append({"path": p, "ext": ext, "status": status, "detail": detail,
                        "seconds": round(time.monotonic() - t0, 3)})
json.dump(results, open("/scratch/results.json", "w"))
print(len(results))
"""


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--wall-timeout-s", type=int, default=3600)
    args = ap.parse_args()
    settings = load_settings(args.config)
    sandbox = settings.sandbox.model_copy(
        update={"wall_timeout_s": args.wall_timeout_s, "cpu_time_s": args.wall_timeout_s}
    )
    runner = SandboxRunner(sandbox)
    image = runner.image_info()
    visible = settings.kramabench.visible_root
    report: dict[str, object] = {
        "started": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "image": image.model_dump(mode="json"),
        "domains": {},
    }
    domains: dict[str, object] = {}
    files: list[dict[str, object]] = []
    for domain in sorted(p.name for p in visible.iterdir() if p.is_dir()):
        mount = f"/data/{domain}"
        run = runner.run(LOADER, [InputMount(host_path=visible / domain, container_path=mount)])
        if run.exit_code != 0:
            raise SystemExit(f"{domain}: loader exited {run.exit_code}: {run.stderr[-2000:]}")
        results = json.loads((run.scratch_dir / "results.json").read_text())
        paths = {r["path"] for r in results}
        audited = set(run.audit.data_reads)
        domains[domain] = {
            "files": len(results),
            "wall_s": run.wall_s,
            "audit_complete": run.audit.complete,
            "audit_issues": run.audit.issues,
            "loaded_not_observed": sorted(paths - audited),
        }
        files += results
        print(
            f"{domain:12s} files={len(results):5d} wall={run.wall_s:7.1f}s "
            f"audit_complete={run.audit.complete} unobserved={len(paths - audited)}",
            flush=True,
        )
    by_ext: dict[str, Counter[str]] = {}
    for r in files:
        by_ext.setdefault(str(r["ext"]), Counter())[str(r["status"])] += 1
    report["domains"] = domains
    report["by_extension"] = {k: dict(v) for k, v in sorted(by_ext.items())}
    report["slowest"] = sorted(files, key=lambda r: -float(r["seconds"]))[:10]  # type: ignore[arg-type]
    report["files"] = files
    for ext, counts in report["by_extension"].items():  # type: ignore[attr-defined]
        print(f"{ext:8s} {counts}")
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, indent=1), encoding="utf-8")


if __name__ == "__main__":
    main()
