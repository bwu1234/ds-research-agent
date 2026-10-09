"""Derive the sandbox seccomp profile from Docker's default profile.

    gh api -H "Accept: application/vnd.github.raw" \
      "repos/moby/moby/contents/$PROFILE?ref=docker-v29.8.2" > /path/to/default.json
    uv run python scripts/build_seccomp_profile.py /path/to/default.json

where PROFILE is vendor/github.com/moby/profiles/seccomp/default.json.
The upstream file is checked against its Git blob hash, so the output is a
pure function of a pinned input. The one change: ``clone`` is allowed only
when its flags also exclude ``CLONE_UNTRACED``, which would otherwise let a
child escape the read auditor's ptrace (D0 spike). Docker's default already
answers ``clone3`` with ENOSYS for containers without ``CAP_SYS_ADMIN``, so
libc falls back to ``clone``, where the flag is visible.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

# moby docker-v29.8.2, vendor/github.com/moby/profiles/seccomp v0.2.3.
UPSTREAM_BLOB_SHA1 = "ea5a494afb8d64898fa0f4f47ae0c4f5ba9cbbc9"
NAMESPACE_MASK = 0x7E020000  # CLONE_NEW{NS,CGROUP,UTS,IPC,USER,PID,NET}
CLONE_UNTRACED = 0x00800000
OUT = Path(__file__).resolve().parents[1] / "ds_research_agent/sandbox/image/seccomp.json"


def git_blob_sha1(data: bytes) -> str:
    return hashlib.sha1(b"blob %d\0" % len(data) + data, usedforsecurity=False).hexdigest()


def derive(profile: dict[str, object]) -> dict[str, object]:
    patched = 0
    syscalls = profile["syscalls"]
    assert isinstance(syscalls, list)
    for rule in syscalls:
        if rule.get("names") != ["clone"] or rule.get("action") != "SCMP_ACT_ALLOW":
            continue
        for arg in rule.get("args", []):
            if arg["op"] == "SCMP_CMP_MASKED_EQ" and arg["value"] == NAMESPACE_MASK:
                arg["value"] = NAMESPACE_MASK | CLONE_UNTRACED
                patched += 1
    # One rule per clone argument order (generic and s390).
    if patched != 2:
        raise SystemExit(f"expected 2 clone rules to patch, found {patched}")
    return profile


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("upstream", type=Path)
    ap.add_argument("--out", type=Path, default=OUT)
    args = ap.parse_args()
    raw = args.upstream.read_bytes()
    if (sha := git_blob_sha1(raw)) != UPSTREAM_BLOB_SHA1:
        raise SystemExit(f"upstream blob {sha} != pinned {UPSTREAM_BLOB_SHA1}")
    out = derive(json.loads(raw))
    args.out.write_text(json.dumps(out, indent="\t") + "\n", encoding="utf-8")
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
