#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
RUN_ROOT="${RUN_ROOT:-$SCRIPT_DIR/final_training_run}"
LOG_FILE="${LOG_FILE:-$SCRIPT_DIR/final_training.log}"

if [[ "${1:-}" == "-h" || "${1:-}" == "--help" ]]; then
    echo "Usage: DATA_DIR=/path/to/geo_dataset ./run_training.sh"
    echo "Optional environment variables: RUN_ROOT, LOG_FILE, BATCH_SIZE,"
    echo "INFERENCE_BATCH_SIZE, CHIPOINT_ENCODE_BATCH, PLONK_BATCH_SIZE,"
    echo "WORKERS, and MIN_FREE_MIB."
    exit 0
fi
if (( $# > 0 )); then
    echo "Unknown argument: $1 (use --help for usage)" >&2
    exit 2
fi

if [[ "${FINAL_TRAINING_WORKER:-0}" != "1" ]]; then
    nohup env FINAL_TRAINING_WORKER=1 \
        RUN_ROOT="$RUN_ROOT" LOG_FILE="$LOG_FILE" DATA_DIR="${DATA_DIR:-}" \
        BATCH_SIZE="${BATCH_SIZE:-24}" INFERENCE_BATCH_SIZE="${INFERENCE_BATCH_SIZE:-48}" \
        CHIPOINT_ENCODE_BATCH="${CHIPOINT_ENCODE_BATCH:-12}" \
        PLONK_BATCH_SIZE="${PLONK_BATCH_SIZE:-8}" WORKERS="${WORKERS:-8}" \
        MIN_FREE_MIB="${MIN_FREE_MIB:-15000}" \
        bash "$0" >"$LOG_FILE" 2>&1 < /dev/null &
    PID=$!
    echo "Started clean final-training worker PID $PID"
    echo "Log: $LOG_FILE"
    echo "Follow progress with: tail -f '$LOG_FILE'"
    exit 0
fi

if [[ -n "${VIRTUAL_ENV:-}" && -x "$VIRTUAL_ENV/bin/python" ]]; then
    PYTHON="$VIRTUAL_ENV/bin/python"
elif [[ -x "$SCRIPT_DIR/.venv/bin/python" ]]; then
    PYTHON="$SCRIPT_DIR/.venv/bin/python"
elif [[ -x "$SCRIPT_DIR/../.venv/bin/python" ]]; then
    PYTHON="$SCRIPT_DIR/../.venv/bin/python"
else
    PYTHON="$(command -v python3)"
fi

DATA_DIR="${DATA_DIR:-$SCRIPT_DIR/../geo_dataset}"
BATCH_SIZE="${BATCH_SIZE:-24}"
INFERENCE_BATCH_SIZE="${INFERENCE_BATCH_SIZE:-48}"
CHIPOINT_ENCODE_BATCH="${CHIPOINT_ENCODE_BATCH:-12}"
PLONK_BATCH_SIZE="${PLONK_BATCH_SIZE:-8}"
WORKERS="${WORKERS:-8}"
MIN_FREE_MIB="${MIN_FREE_MIB:-15000}"

echo "Final all-data training"
echo "Host: $(hostname)"
echo "Python: $PYTHON"
echo "Data: $DATA_DIR"
echo "Run root: $RUN_ROOT"
echo "Started: $(date --iso-8601=seconds)"

"$PYTHON" - <<'PY'
import importlib
required = [
    "numpy", "pandas", "PIL", "scipy", "sklearn", "torch", "torchvision",
    "huggingface_hub", "open_clip", "plonk",
]
missing = []
for name in required:
    try:
        importlib.import_module(name)
    except Exception as error:
        missing.append(f"{name}: {error}")
if missing:
    raise SystemExit(
        "Missing/broken Python dependencies:\n  " + "\n  ".join(missing)
        + "\nInstall requirements.txt without replacing your CUDA torch build."
    )
PY

if ! command -v nvidia-smi >/dev/null 2>&1; then
    echo "nvidia-smi is unavailable; CUDA training cannot start." >&2
    exit 1
fi

echo "Waiting for at least ${MIN_FREE_MIB} MiB of free GPU memory..."
while true; do
    FREE_MIB="$(nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits | head -n 1 | tr -d ' ')"
    if [[ "$FREE_MIB" =~ ^[0-9]+$ ]] && (( FREE_MIB >= MIN_FREE_MIB )); then
        break
    fi
    echo "$(date --iso-8601=seconds): GPU free ${FREE_MIB:-unknown} MiB; waiting 30 s"
    sleep 30
done
echo "GPU is ready (${FREE_MIB} MiB free)."

"$PYTHON" "$SCRIPT_DIR/train.py" \
    --data-dir "$DATA_DIR" \
    --run-root "$RUN_ROOT" \
    --batch-size "$BATCH_SIZE" \
    --inference-batch-size "$INFERENCE_BATCH_SIZE" \
    --chipoint-encode-batch "$CHIPOINT_ENCODE_BATCH" \
    --plonk-batch-size "$PLONK_BATCH_SIZE" \
    --workers "$WORKERS"

echo "Finished: $(date --iso-8601=seconds)"
