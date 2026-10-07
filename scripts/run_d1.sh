#!/usr/bin/env bash
# D1 model runs, in order: inline smoke, thinking-level sweep on the fixed
# development sample, the pre-registered level choice, both baselines on the
# development split at that level, then a replay check of every batch.
# Local model only (free, slow: roughly 12-14 h). Holdout untouched.
# Resumable: every batch runs with --resume, so rerunning the script skips
# finished tasks and redoes only an interrupted one.
#
#   scripts/run_d1.sh config/local.yaml > data/runs/d1.log 2>&1
set -euo pipefail
CONFIG=${1:-config/local.yaml}
run() { uv run python -m eval.kramabench.run --config "$CONFIG" "$@"; }

echo "== $(date -u +%FT%TZ) smoke"
run run --resume --condition inline --tasks smoke --think medium --batch d1-smoke-inline-medium

LEVELS=(off low medium xhigh)
SWEEP=()
for t in "${LEVELS[@]}"; do
  echo "== $(date -u +%FT%TZ) sweep $t"
  run run --resume --condition inline --tasks sample --think "$t" --batch "d1-sweep-inline-$t"
  SWEEP+=(--batch "d1-sweep-inline-$t")
done

echo "== $(date -u +%FT%TZ) choose thinking level"
run choose-think "${SWEEP[@]}" | tee data/runs/d1-choose-think.json
THINK=$(uv run python -c 'import json,sys; print(json.load(open(sys.argv[1]))["chosen"])' \
  data/runs/d1-choose-think.json)
[ "$THINK" = "False" ] && THINK=off

for c in no_tools inline; do
  echo "== $(date -u +%FT%TZ) dev $c $THINK"
  run run --resume --condition "$c" --tasks dev --think "$THINK" --batch "d1-dev-$c-$THINK"
done

echo "== $(date -u +%FT%TZ) replay checks"
BATCHES=(d1-smoke-inline-medium)
for t in "${LEVELS[@]}"; do BATCHES+=("d1-sweep-inline-$t"); done
BATCHES+=("d1-dev-no_tools-$THINK" "d1-dev-inline-$THINK")
for b in "${BATCHES[@]}"; do run replay --batch "$b"; done
echo "== $(date -u +%FT%TZ) done"
