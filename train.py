#!/usr/bin/env python3
"""Run teacher preparation, final training, prediction, and verification."""

from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import json
import subprocess
import sys
import time
import uuid
from pathlib import Path

import numpy as np
import pandas as pd


HERE = Path(__file__).resolve().parent
EXPECTED_TRAIN_ROWS = 11_758
EXPECTED_HOLDOUT_ROWS = 2_400

def utc_now() -> str:
    return dt.datetime.now(dt.timezone.utc).isoformat()

def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()

def atomic_json(path: Path, payload: dict) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    temporary.replace(path)

def dataset_fingerprint(data_dir: Path) -> dict:
    labels_path = data_dir / "train_labels.csv"
    frame = pd.read_csv(labels_path)
    train_names = frame.filename.astype(str).tolist()
    holdout_names = sorted(path.name for path in (data_dir / "holdout_public").glob("*.jpg"))
    listing = hashlib.sha256()
    for name in train_names:
        path = data_dir / "train" / name
        if not path.is_file():
            raise FileNotFoundError(path)
        listing.update(f"{name}\0{path.stat().st_size}\n".encode())
    for name in holdout_names:
        path = data_dir / "holdout_public" / name
        listing.update(f"{name}\0{path.stat().st_size}\n".encode())
    if len(frame) != EXPECTED_TRAIN_ROWS:
        raise RuntimeError(f"Expected {EXPECTED_TRAIN_ROWS:,} labels, found {len(frame):,}")
    if len(holdout_names) != EXPECTED_HOLDOUT_ROWS:
        raise RuntimeError(
            f"Expected {EXPECTED_HOLDOUT_ROWS:,} holdout images, "
            f"found {len(holdout_names):,}"
        )
    return {
        "train_rows": len(frame),
        "holdout_rows": len(holdout_names),
        "train_labels_sha256": sha256(labels_path),
        "ordered_filenames_and_sizes_sha256": listing.hexdigest(),
    }

def run_phase(name: str, command: list[str], timing_path: Path, complete) -> None:
    timing = json.loads(timing_path.read_text())
    if complete():
        print(f"=== {name}: already complete in this same run; skipping ===", flush=True)
        timing.setdefault("phases", {}).setdefault(name, {})["last_status"] = "already_complete"
        atomic_json(timing_path, timing)
        return
    print(f"=== {name} ===", flush=True)
    print(" ".join(command), flush=True)
    started_utc = utc_now()
    started = time.perf_counter()
    status = "failed"
    try:
        subprocess.run(command, cwd=HERE, check=True)
        if not complete():
            raise RuntimeError(f"{name} returned successfully but its output validation failed")
        status = "complete"
    finally:
        seconds = time.perf_counter() - started
        timing = json.loads(timing_path.read_text())
        record = timing.setdefault("phases", {}).setdefault(name, {"attempts": []})
        record.setdefault("attempts", []).append({
            "started_at_utc": started_utc,
            "ended_at_utc": utc_now(),
            "wall_seconds_this_invocation": seconds,
            "status": status,
            "command": command,
        })
        record["cumulative_wall_seconds"] = sum(
            attempt["wall_seconds_this_invocation"] for attempt in record["attempts"]
        )
        record["last_status"] = status
        atomic_json(timing_path, timing)
        print(f"=== {name}: {status}, {seconds / 60:.2f} min this invocation ===", flush=True)

def valid_chipoint(cache: Path, rows: int) -> bool:
    try:
        manifest = json.loads((cache / "manifest.json").read_text())
        return (
            len(manifest["filenames"]) == rows
            and np.load(cache / "queries_so400m256_r1.npy", mmap_mode="r").shape[0] == rows
            and np.load(cache / "queries_gopt_r1.npy", mmap_mode="r").shape[0] == rows
            and np.load(cache / "retrieval_so400m256_r1_m200.npz")["indices"].shape[0] == rows
            and np.load(cache / "retrieval_gopt_r1_m200.npz")["indices"].shape[0] == rows
            and np.load(cache.parent / "chipoint_gallery_coords.npy", mmap_mode="r").ndim == 2
        )
    except (FileNotFoundError, KeyError, ValueError, json.JSONDecodeError):
        return False

def valid_plonk(cache: Path, rows: int) -> bool:
    try:
        manifest = json.loads((cache / "manifest.json").read_text())
        return (
            len(manifest["filenames"]) == rows
            and tuple(np.load(cache / "plonk_all_samples.npy", mmap_mode="r").shape)
            == (rows, 4, 4, 2)
        )
    except (FileNotFoundError, KeyError, ValueError, json.JSONDecodeError):
        return False

def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument("--data-dir", type=Path, default=HERE.parent / "geo_dataset")
    parser.add_argument("--run-root", type=Path, default=HERE / "final_training_run")
    parser.add_argument("--batch-size", type=int, default=24)
    parser.add_argument("--inference-batch-size", type=int, default=48)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--chipoint-encode-batch", type=int, default=12)
    parser.add_argument("--plonk-batch-size", type=int, default=8)
    args = parser.parse_args()

    data_dir = args.data_dir.resolve()
    run_root = args.run_root.resolve()
    manifest_path = run_root / "run_manifest.json"
    timing_path = run_root / "pipeline_timing.json"
    if not data_dir.is_dir():
        raise FileNotFoundError(data_dir)
    if run_root.exists() and any(run_root.iterdir()) and not manifest_path.exists():
        raise RuntimeError(
            f"Refusing non-empty unregistered run directory {run_root}. "
            "Use a new empty RUN_ROOT for a demonstrably fresh run."
        )
    run_root.mkdir(parents=True, exist_ok=True)
    fingerprint = dataset_fingerprint(data_dir)
    source_files = [
        "train.py", "cache_chipoint.py", "cache_plonk.py",
        "prepare_teacher_targets.py", "model.py", "train_student.py",
        "predict.py", "verify.py",
    ]
    hashes = {name: sha256(HERE / name) for name in source_files}
    if manifest_path.exists():
        manifest = json.loads(manifest_path.read_text())
        if manifest["dataset"] != fingerprint or manifest["source_sha256"] != hashes:
            raise RuntimeError(
                "This interrupted run does not match the current dataset/scripts. "
                "Use a new empty RUN_ROOT instead of mixing run provenance."
            )
        print(f"Resuming registered clean run {manifest['run_id']}", flush=True)
    else:
        manifest = {
            "run_id": str(uuid.uuid4()),
            "created_at_utc": utc_now(),
            "purpose": "single final model trained on all labeled training images",
            "freshness_contract": {
                "student_checkpoint_initialization": "none; ImageNet initialization only",
                "teacher_query_embeddings": "generated inside this run directory",
                "teacher_retrieval_results": "generated inside this run directory",
                "plonk_predictions": "generated inside this run directory",
                "distillation_targets": "generated inside this run directory",
                "public_download_cache": (
                    "allowed only for immutable public model weights/gallery resources"
                ),
            },
            "fixed_epochs": 72,
            "dataset": fingerprint,
            "source_sha256": hashes,
            "command_line": sys.argv,
        }
        atomic_json(manifest_path, manifest)
    if not timing_path.exists():
        atomic_json(timing_path, {
            "run_id": manifest["run_id"], "pipeline_started_at_utc": utc_now(),
            "timing_definition": {
                "reported_training_time": (
                    "CUDA-synchronized wall time inside the 72 student epoch loops"
                ),
                "excluded": [
                    "public-model downloads", "fresh teacher inference", "geocell construction",
                    "student model construction/ImageNet download",
                    "checkpoint serialization", "holdout inference",
                ],
            },
            "phases": {},
        })

    python = sys.executable
    teacher_cache = run_root / "teacher_cache"
    chipoint_cache = teacher_cache / "chipoint"
    plonk_cache = teacher_cache / "plonk"
    targets = teacher_cache / "teacher_targets_all_cells1536.pt"
    model_dir = run_root / "model"
    final_model = model_dir / "final_model.pt"
    predictions = run_root / "submission" / "predictions.csv"
    verification = run_root / "verification"

    run_phase("1_chipoint_teacher", [
        python, str(HERE / "cache_chipoint.py"),
        "--data-dir", str(data_dir), "--cache-dir", str(chipoint_cache),
        "--batch-size", str(args.chipoint_encode_batch),
    ], timing_path, lambda: valid_chipoint(chipoint_cache, fingerprint["train_rows"]))

    run_phase("2_plonk_teacher", [
        python, str(HERE / "cache_plonk.py"),
        "--data-dir", str(data_dir), "--cache-dir", str(plonk_cache),
        "--batch-size", str(args.plonk_batch_size),
    ], timing_path, lambda: valid_plonk(plonk_cache, fingerprint["train_rows"]))

    run_phase("3_teacher_targets", [
        python, str(HERE / "prepare_teacher_targets.py"),
        "--data-dir", str(data_dir), "--teacher-cache", str(teacher_cache),
        "--output", str(targets),
    ], timing_path, lambda: targets.is_file())

    train_command = [
        python, str(HERE / "train_student.py"),
        "--data-dir", str(data_dir), "--teacher-target", str(targets),
        "--output-dir", str(model_dir), "--batch-size", str(args.batch_size),
        "--workers", str(args.workers),
    ]
    if (model_dir / "latest.pt").exists():
        train_command.append("--resume")
    training_timing = model_dir / "training_time.json"
    run_phase(
        "4_student_training", train_command, timing_path,
        lambda: final_model.is_file() and training_timing.is_file(),
    )

    def predictions_complete() -> bool:
        try:
            frame = pd.read_csv(predictions)
            return (
                frame.columns.tolist() == ["filename", "pred_lat", "pred_lng"]
                and len(frame) == EXPECTED_HOLDOUT_ROWS
                and frame.filename.nunique() == EXPECTED_HOLDOUT_ROWS
                and np.isfinite(frame[["pred_lat", "pred_lng"]].to_numpy()).all()
            )
        except (FileNotFoundError, ValueError, KeyError, pd.errors.ParserError):
            return False

    run_phase("5_holdout_inference", [
        python, str(HERE / "predict.py"),
        "--image-dir", str(data_dir / "holdout_public"),
        "--checkpoint", str(final_model),
        "--expected-count", str(EXPECTED_HOLDOUT_ROWS),
        "--output", str(predictions), "--batch-size", str(args.inference_batch_size),
        "--workers", str(args.workers),
    ], timing_path, predictions_complete)

    report = verification / "parameter_report.json"
    run_phase("6_artifact_verification", [
        python, str(HERE / "verify.py"),
        "--checkpoint", str(final_model), "--predictions", str(predictions),
        "--output-dir", str(verification),
        "--expected-rows", str(EXPECTED_HOLDOUT_ROWS),
    ], timing_path, lambda: (
        report.is_file()
        and (verification / "PARAMETER_REPORT.txt").is_file()
        and (verification / "parameter_breakdown_final.csv").is_file()
        and json.loads(report.read_text()).get("verification") == "PASS"
    ))

    timing = json.loads(timing_path.read_text())
    training = json.loads((model_dir / "training_time.json").read_text())
    timing["pipeline_completed_at_utc"] = utc_now()
    timing["authoritative_student_training"] = training
    atomic_json(timing_path, timing)
    print("\nFINAL PIPELINE COMPLETE", flush=True)
    print(f"Training time: {training['training_seconds'] / 3600:.4f} h ", flush=True)
    print(f"Final model: {final_model}", flush=True)
    print(f"Predictions: {predictions}", flush=True)
    print(f"Parameter report: {verification / 'PARAMETER_REPORT.txt'}", flush=True)
    print(f"Full timing/provenance: {timing_path}", flush=True)


if __name__ == "__main__":
    main()
