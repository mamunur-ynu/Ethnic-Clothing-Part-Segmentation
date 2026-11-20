#!/usr/bin/env bash
# Run the pipeline step by step. Each switch runs one group of steps:
#   bash run_all.sh collect    search + download on Modal, copy to data/, load and screen
#   bash run_all.sh sam3       SAM 3 masks on Modal GPU, copy back, compose 512x512
#   bash run_all.sh review     make the two-person review plan (then each person runs: python -m ethnoseg review)
#   bash run_all.sh build      split, export, QC, statistics, review statistics, mask check
#   bash run_all.sh status     counts per group and status
# Steps that need a person (review, mask-edit) are never run automatically.
set -euo pipefail
cd "$(dirname "$0")"
[ -f .venv/bin/activate ] && source .venv/bin/activate
E="python -m ethnoseg"

case "${1:-status}" in
  collect)
    modal run modal_collect.py
    $E pull raw
    $E ingest
    $E reverify          # strict rule: evidence must name exactly this one group
    $E screen ;;
  sam3)
    modal run modal_segment.py
    $E pull seg
    $E compose ;;
  review)
    $E review-plan --reviewers "${REVIEWERS:-Mamun,Imamul}"
    echo "Now each reviewer runs:  python -m ethnoseg review --reviewer <name>" ;;
  build)
    $E split
    $E export
    $E qc
    $E stats
    $E review-stats
    $E mask-check
    $E mask-agreement ;;
  status)
    $E status ;;
  *)
    echo "usage: bash run_all.sh collect|sam3|review|build|status"; exit 1 ;;
esac
