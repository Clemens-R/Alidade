#!/usr/bin/env python3
"""Train the fixed final student on all labeled images for 72 epochs."""

from __future__ import annotations

import argparse
import datetime as dt
import json
import time
from pathlib import Path

import pandas as pd
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

import model as geo


EXPECTED_PARAMETERS = 4_964_188
EPOCHS = 72
FEATURE_EPOCHS = 16

def learning_rate(epoch: int) -> float:
    if epoch <= 52:
        return 2e-4
    if epoch <= 68:
        return 8e-5
    return 3.2e-5

def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, required=True)
    parser.add_argument("--teacher-target", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--batch-size", type=int, default=24)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()

    if not torch.cuda.is_available():
        raise SystemExit("CUDA is required for training")

    data_dir = args.data_dir.resolve()
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    latest_path = output_dir / "latest.pt"
    final_path = output_dir / "final_model.pt"

    if final_path.exists() and (output_dir / "training_time.json").exists():
        print(f"Final model already complete: {final_path}", flush=True)
        return
    if latest_path.exists() and not args.resume:
        raise SystemExit(f"{latest_path} exists; pass --resume or use an empty directory")

    geo.set_seed(42)
    frame = pd.read_csv(data_dir / "train_labels.csv")
    country_map = {country: index for index, country in enumerate(geo.COUNTRIES)}
    centers_np, soft_cells, _ = geo.make_cells(frame)
    teacher = geo.load_teacher_targets(args.teacher_target.resolve(), frame, centers_np)
    distill_transform, train_transform = geo.make_transforms()

    # The ImageNet download and model construction are outside the reported
    # training interval.
    model = geo.AdapterMoE(imagenet_init=True)
    total_parameters = sum(parameter.numel() for parameter in model.parameters())
    trainable_parameters = sum(
        parameter.numel() for parameter in model.parameters() if parameter.requires_grad
    )
    if total_parameters != EXPECTED_PARAMETERS or trainable_parameters != EXPECTED_PARAMETERS:
        raise RuntimeError(
            f"Parameter audit failed: total={total_parameters:,}, "
            f"trainable={trainable_parameters:,}"
        )

    parameter_report = {
        "method": "sum(parameter.numel() for parameter in model.parameters())",
        "total_parameters": total_parameters,
        "trainable_parameters": trainable_parameters,
        "limit": 5_000_000,
        "headroom": 5_000_000 - total_parameters,
    }
    (output_dir / "parameter_count_pretrain.json").write_text(
        json.dumps(parameter_report, indent=2) + "\n"
    )
    print(f"Model parameters: {total_parameters:,} / 5,000,000", flush=True)

    device = torch.device("cuda")
    model = model.to(device)
    loader_options = {
        "num_workers": args.workers,
        "pin_memory": True,
        "persistent_workers": args.workers > 0,
    }
    image_dir = data_dir / "train"
    distill_loader = DataLoader(
        geo.GeoDataset(frame, image_dir, distill_transform, country_map, soft_cells),
        batch_size=args.batch_size, shuffle=True, **loader_options,
    )
    train_loader = DataLoader(
        geo.GeoDataset(frame, image_dir, train_transform, country_map, soft_cells),
        batch_size=args.batch_size, shuffle=True, **loader_options,
    )

    optimizer = torch.optim.AdamW(model.parameters(), lr=2e-4, weight_decay=0.025)
    scaler = torch.amp.GradScaler("cuda")
    centers = torch.from_numpy(centers_np).to(device)

    start_epoch = 0
    training_seconds = 0.0
    history = []
    started_at = dt.datetime.now(dt.timezone.utc).isoformat()
    resume_segments = 0
    if latest_path.exists():
        state = torch.load(latest_path, map_location=device, weights_only=True)
        if state.get("parameter_count") != EXPECTED_PARAMETERS:
            raise RuntimeError("Resume checkpoint has an unexpected parameter count")
        model.load_state_dict(state["model"])
        optimizer.load_state_dict(state["optimizer"])
        scaler.load_state_dict(state["scaler"])
        start_epoch = int(state["epoch"])
        training_seconds = float(state["completed_training_seconds"])
        history = list(state.get("history", []))
        started_at = state.get("run_started_at_utc", started_at)
        resume_segments = int(state.get("resume_segments", 0)) + 1
        torch.set_rng_state(state["torch_rng_state"].cpu())
        torch.cuda.set_rng_state_all(
            [value.cpu() for value in state["cuda_rng_state_all"]]
        )
        print(f"Resuming at epoch {start_epoch + 1}", flush=True)

    for epoch_index in range(start_epoch, EPOCHS):
        epoch = epoch_index + 1
        stage = "feature" if epoch <= FEATURE_EPOCHS else "geo"
        loader = distill_loader if stage == "feature" else train_loader
        lr = learning_rate(epoch)
        for group in optimizer.param_groups:
            group["lr"] = lr

        model.train()
        totals = {
            name: 0.0
            for name in ("loss", "cell", "country", "router", "feature", "geo", "kd")
        }
        correct_country = correct_router = seen = 0
        log_clock = time.perf_counter()
        torch.cuda.synchronize()
        epoch_started = time.perf_counter()

        for step, (images, truth, country, cell_target, indices) in enumerate(loader, 1):
            images = images.to(device, non_blocking=True)
            truth = truth.to(device)
            country = country.to(device)
            cell_target = cell_target.to(device)
            optimizer.zero_grad(set_to_none=True)

            with torch.amp.autocast("cuda", dtype=torch.float16):
                output = model(images)
                cell_loss = geo.conditional_cell_loss(output["cell"], cell_target)
                country_loss = F.cross_entropy(
                    output["country"], country, label_smoothing=0.01
                )
                router_loss = F.cross_entropy(
                    output["router"], country, label_smoothing=0.03
                )
                balance_loss = geo.load_balance_loss(output["gates"])

                so_target = teacher["chi_feature"][indices].to(device).float()
                gopt_target = teacher["gopt_feature"][indices].to(device).float()
                feature_loss = (
                    (1 - (output["so_feature"].float() * so_target).sum(1)).mean()
                    + (1 - (output["gopt_feature"].float() * gopt_target).sum(1)).mean()
                ) / 2
                cell_kd = geo.kd_loss(
                    output["cell"], teacher["fused_cell_prob"][indices].to(device)
                )
                country_kd = geo.kd_loss(
                    output["country"], teacher["fused_country_prob"][indices].to(device)
                )
                kd_loss = 0.5 * (cell_kd + country_kd)

                geo_loss = output["cell"].new_zeros(())
                if stage == "feature":
                    loss = (
                        feature_loss + country_loss + 0.25 * cell_loss + router_loss
                        + 0.01 * balance_loss + 0.25 * kd_loss
                    )
                else:
                    candidates = model.coordinates_by_country(output, centers)
                    rows = torch.arange(len(country), device=device)
                    prediction = candidates[rows, country]
                    geo_loss = geo.robust_angular_loss(prediction, truth)
                    loss = (
                        cell_loss + 10.0 * geo_loss + country_loss + router_loss
                        + 0.01 * balance_loss + 0.25 * kd_loss
                    )

            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5)
            scaler.step(optimizer)
            scaler.update()

            batch_size = len(country)
            seen += batch_size
            correct_country += int((output["country"].argmax(1) == country).sum())
            correct_router += int((output["router"].argmax(1) == country).sum())
            for name, value in (
                ("loss", loss), ("cell", cell_loss), ("country", country_loss),
                ("router", router_loss), ("feature", feature_loss),
                ("geo", geo_loss), ("kd", kd_loss),
            ):
                totals[name] += float(value.item())

            if step == len(loader) or time.perf_counter() - log_clock >= 60:
                print(
                    f"[epoch {epoch}/{EPOCHS} {stage}] {step}/{len(loader)} "
                    f"lr={lr:.2e} loss={totals['loss']/step:.4f} "
                    f"country={100*correct_country/seen:.1f}% "
                    f"router={100*correct_router/seen:.1f}%",
                    flush=True,
                )
                log_clock = time.perf_counter()

        torch.cuda.synchronize()
        epoch_seconds = time.perf_counter() - epoch_started
        training_seconds += epoch_seconds
        history.append({
            "epoch": epoch,
            "stage": stage,
            "learning_rate": lr,
            "seconds": epoch_seconds,
            "cumulative_training_seconds": training_seconds,
            **{name: value / len(loader) for name, value in totals.items()},
        })
        average_epoch = training_seconds / epoch
        print(
            f"Epoch {epoch} complete in {epoch_seconds:.1f}s; "
            f"training total={training_seconds/60:.1f} min; "
            f"ETA={average_epoch*(EPOCHS-epoch)/60:.1f} min",
            flush=True,
        )

        checkpoint = {
            "epoch": epoch,
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "scaler": scaler.state_dict(),
            "centers": torch.from_numpy(centers_np),
            "config": geo.FINAL_MODEL_CONFIG,
            "parameter_count": total_parameters,
            "completed_training_seconds": training_seconds,
            "history": history,
            "run_started_at_utc": started_at,
            "resume_segments": resume_segments,
            "torch_rng_state": torch.get_rng_state(),
            "cuda_rng_state_all": torch.cuda.get_rng_state_all(),
        }
        geo.atomic_save(checkpoint, latest_path)
        pd.DataFrame(history).to_csv(output_dir / "training_history.csv", index=False)

    finished_at = dt.datetime.now(dt.timezone.utc).isoformat()
    final_checkpoint = {
        "model": model.state_dict(),
        "centers": torch.from_numpy(centers_np),
        "epoch": EPOCHS,
        "config": geo.FINAL_MODEL_CONFIG,
        "parameters": total_parameters,
        "parameter_count": total_parameters,
        "completed_training_seconds": training_seconds,
        "run_started_at_utc": started_at,
        "run_finished_at_utc": finished_at,
        "resume_segments": resume_segments,
    }
    geo.atomic_save(final_checkpoint, final_path)

    timing = {
        "measurement": "student training-loop wall-clock time",
        "clock": "time.perf_counter with CUDA synchronization at epoch boundaries",
        "includes": [
            "training image loading", "augmentation", "forward pass",
            "loss", "backward pass", "optimizer update",
        ],
        "excludes": [
            "downloads", "teacher inference", "teacher-target construction",
            "model initialization", "geocell fitting", "checkpoint serialization",
            "holdout inference",
        ],
        "training_seconds": training_seconds,
        "training_minutes": training_seconds / 60,
        "training_hours": training_seconds / 3600,
        "epochs": EPOCHS,
        "training_images": len(frame),
        "image_presentations": len(frame) * EPOCHS,
        "batch_size": args.batch_size,
        "resume_segments": resume_segments,
        "started_at_utc": started_at,
        "finished_at_utc": finished_at,
        "gpu": torch.cuda.get_device_name(0),
        "torch": torch.__version__,
        "cuda": torch.version.cuda,
    }
    (output_dir / "training_time.json").write_text(json.dumps(timing, indent=2) + "\n")
    (output_dir / "TRAINING_TIME.txt").write_text(
        f"Student training time: {timing['training_hours']:.6f} hours "
        f"({timing['training_minutes']:.3f} minutes; {training_seconds:.3f} seconds)\n"
        "Measured with time.perf_counter and CUDA synchronization.\n"
        "Includes loading, augmentation, forward/backward passes, and optimizer steps.\n"
        "Excludes downloads, teacher preparation, checkpoints, and inference.\n"
    )
    print(f"Final model: {final_path}", flush=True)
    print(f"Measured training time: {training_seconds/60:.3f} minutes", flush=True)


if __name__ == "__main__":
    main()
