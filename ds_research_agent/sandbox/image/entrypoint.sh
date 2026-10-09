#!/bin/sh
# Trusted wrapper; runs as root and is PID 1. The program runs as uid 1000
# under strace with its stdout/stderr redirected to /out. Only this wrapper
# writes the container's stdout, which carries the trace to the runner.
#   entrypoint.sh <wall_s> <cpu_s> <max_file_bytes>
set -u
wall="$1"; cpu="$2"; fsize="$3"
mkdir -p /audit && chmod 0700 /audit
start=$(date +%s%N)
# prlimit applies only to the program, not to strace writing the trace.
timeout -s KILL "$wall" strace -ff -qq -y --seccomp-bpf \
  -e trace=open,openat,openat2,creat,execve,execveat,clone,clone3,fork,vfork,io_uring_setup,open_by_handle_at \
  -o /audit/t -u agent -- \
  prlimit --cpu="$cpu" --fsize="$fsize" --core=0 -- \
  python3 /program/main.py >/out/stdout 2>/out/stderr
rc=$?
end=$(date +%s%N)
# Anything still running after strace is gone would read unobserved.
kill -9 -1 2>/dev/null
echo "=== exit $rc"
echo "=== elapsed_ns $((end - start))"
echo "=== strace $(strace -V | head -1)"
for f in /audit/t.*; do
  [ -e "$f" ] || continue
  echo "=== file $f"
  cat "$f"
done
echo "=== end"
