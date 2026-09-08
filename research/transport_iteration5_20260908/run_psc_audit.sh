#!/usr/bin/env bash
# Run INSIDE an allocated PSC CPU job. This script does not submit any job.
# Usage: EXPECTED_COMMIT=<review-commit> PYTHON=/absolute/python OUTPUT_DIR=/new/path bash run_psc_audit.sh
set -euo pipefail
: "${EXPECTED_COMMIT:?Set the exact committed audit revision}"
: "${PYTHON:?Set an absolute Python interpreter path}"
: "${OUTPUT_DIR:?Set a NEW output directory}"
: "${SLURM_JOB_ID:?Run inside an allocated PSC job}"
[[ "$PYTHON" = /* && -x "$PYTHON" ]] || { echo 'PYTHON must be an executable absolute path' >&2; exit 2; }
cd "$(dirname "$0")"
[[ "$(git rev-parse HEAD)" = "$EXPECTED_COMMIT" ]] || { echo 'source revision mismatch' >&2; exit 2; }
[[ -z "$(git status --porcelain --untracked-files=no)" ]] || { echo 'tracked source is dirty' >&2; exit 2; }
[[ ! -e "$OUTPUT_DIR" ]] || { echo 'refusing existing output directory' >&2; exit 2; }
mkdir -p "$OUTPUT_DIR"
OUTPUT_DIR="$(cd "$OUTPUT_DIR" && pwd)"
export PYTHONDONTWRITEBYTECODE=1 OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
trap 'code=$?; if ((code)); then printf "failed_exit_code=%s\n" "$code" > "$OUTPUT_DIR/FAILED"; fi' EXIT
printf 'commit=%s\nslurm_job_id=%s\n' "$EXPECTED_COMMIT" "$SLURM_JOB_ID" > "$OUTPUT_DIR/provenance.txt"
"$PYTHON" --version >> "$OUTPUT_DIR/provenance.txt"
sha256sum audit_checks.py test_audit_checks.py AUDIT_AND_DECISION.md DEVELOPMENT_PROTOCOL.json run_psc_audit.sh > "$OUTPUT_DIR/SOURCE_SHA256SUMS"
"$PYTHON" -m unittest -v test_audit_checks > "$OUTPUT_DIR/tests.log" 2>&1
"$PYTHON" audit_checks.py > "$OUTPUT_DIR/deterministic_bound_report.json"
printf 'complete_deterministic_audit_only_no_training_or_data_access\n' > "$OUTPUT_DIR/COMPLETE"
