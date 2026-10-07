#!/bin/sh
# Trusted wrapper, runs as root. The program under audit runs as uid 1000
# under strace; its stdout/stderr go to /scratch. Only this wrapper writes
# the container's stdout, which carries the trace back to the runner.
#   entrypoint.sh trace|trace_ptrace_only|plain <case>
set -u
mode="$1"; shift
mkdir -p /audit && chmod 0700 /audit
mkdir -p /scratch/out && chown 1000:1000 /scratch /scratch/out 2>/dev/null || true
start=$(date +%s%N)
if [ "$mode" = trace ] || [ "$mode" = trace_ptrace_only ]; then
  bpf=--seccomp-bpf
  [ "$mode" = trace_ptrace_only ] && bpf=
  # shellcheck disable=SC2086
  timeout -s KILL 120 strace -ff -qq -y $bpf \
    -e trace=open,openat,openat2,creat,execve,execveat,clone,clone3,fork,vfork,io_uring_setup,open_by_handle_at \
    -o /audit/t -u agent -- python3 /opt/cases/cases.py "$@" \
    >/scratch/out/stdout 2>/scratch/out/stderr
else
  timeout -s KILL 120 setpriv --reuid=1000 --regid=1000 --clear-groups -- \
    python3 /opt/cases/cases.py "$@" >/scratch/out/stdout 2>/scratch/out/stderr
fi
rc=$?
end=$(date +%s%N)
echo "=== exit $rc"
echo "=== elapsed_ns $((end - start))"
echo "=== strace $(strace -V | head -1)"
for f in /audit/t.*; do
  [ -e "$f" ] || continue
  echo "=== file $f"
  cat "$f"
done
echo "=== end"
