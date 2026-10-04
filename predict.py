#!/usr/bin/env python3
"""Predict European coordinates for every image in a directory."""

from __future__ import annotations

import argparse
import hashlib
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from PIL import Image, ImageFile, UnidentifiedImageError
from torch.utils.data import DataLoader, Dataset

import model as geo


ImageFile.LOAD_TRUNCATED_IMAGES = True
SUPPORTED_EXTENSIONS = {".jpg", ".jpeg", ".png", ".webp"}


class ImageFolderDataset(Dataset):
    def __init__(self, image_dir: Path, transform, recursive: bool = False):
        self.image_dir = image_dir
        candidates = image_dir.rglob("*") if recursive else image_dir.iterdir()
        self.paths = sorted(
            path for path in candidates
            if path.is_file() and path.suffix.lower() in SUPPORTED_EXTENSIONS
        )
        self.transform = transform

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, index):
        path = self.paths[index]
        last_error = None
        for attempt in range(4):
            try:
                with Image.open(path) as image:
                    tensor = self.transform(image.convert("RGB"))
                return tensor, path.relative_to(self.image_dir).as_posix()
            except (OSError, UnidentifiedImageError) as error:
                last_error = error
                if attempt < 3:
                    time.sleep(0.5 * (attempt + 1))
        raise RuntimeError(f"Could not decode {path} after four attempts") from last_error


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--image-dir", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, default=Path("final_model.pt"))
    parser.add_argument("--output", type=Path, default=Path("predictions.csv"))
    parser.add_argument("--batch-size", type=int, default=48)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--device", choices=("auto", "cuda", "cpu"), default="auto")
    parser.add_argument("--recursive", action="store_true")
    parser.add_argument(
        "--expected-count", type=int,
        help="Fail unless this exact number of images is found (useful for benchmarks).",
    )
    args = parser.parse_args()

    image_dir = args.image_dir.resolve()
    if not image_dir.is_dir():
        raise FileNotFoundError(image_dir)
    if args.device == "cuda" and not torch.cuda.is_available():
        raise SystemExit("CUDA was requested but is unavailable")
    device_name = (
        "cuda" if torch.cuda.is_available() else "cpu"
    ) if args.device == "auto" else args.device
    device = torch.device(device_name)

    checkpoint_path = args.checkpoint.resolve()
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
    model = geo.AdapterMoE(imagenet_init=False)
    model.load_state_dict(checkpoint["model"])
    model = model.to(device).eval()
    centers = torch.as_tensor(
        checkpoint["centers"], dtype=torch.float32, device=device
    )

    inference_transform, _ = geo.make_transforms()
    dataset = ImageFolderDataset(image_dir, inference_transform, args.recursive)
    if not dataset:
        raise RuntimeError(f"No supported images found in {image_dir}")
    if args.expected_count is not None and len(dataset) != args.expected_count:
        raise RuntimeError(
            f"Expected {args.expected_count:,} images, found {len(dataset):,}"
        )
    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.workers,
        pin_memory=device.type == "cuda",
        persistent_workers=args.workers > 0,
    )

    filenames, predictions = [], []
    if device.type == "cuda":
        torch.cuda.synchronize()
    started = time.perf_counter()
    with torch.inference_mode():
        for step, (images, names) in enumerate(loader, 1):
            images = images.to(device, non_blocking=device.type == "cuda")
            sums = None
            for rotation in range(4):
                view = torch.rot90(images, rotation, (-2, -1)) if rotation else images
                with torch.autocast(
                    device_type=device.type,
                    dtype=torch.float16,
                    enabled=device.type == "cuda",
                ):
                    output = model(view)
                if sums is None:
                    sums = {
                        key: output[key].float()
                        for key in ("cell", "country", "residual")
                    }
                else:
                    for key in sums:
                        sums[key] += output[key].float()
            averaged = {key: value / 4 for key, value in sums.items()}
            prediction = model.decode(averaged, centers)
            predictions.append(prediction.cpu().numpy())
            filenames.extend(names)
            completed = min(step * args.batch_size, len(dataset))
            print(f"[predict] {completed}/{len(dataset)}", flush=True)

    if device.type == "cuda":
        torch.cuda.synchronize()
    inference_seconds = time.perf_counter() - started
    latitude, longitude = geo.xyz2ll(np.concatenate(predictions))
    output = pd.DataFrame({
        "filename": filenames,
        "pred_lat": latitude,
        "pred_lng": longitude,
    })
    if output.filename.duplicated().any():
        raise RuntimeError("Input paths do not produce unique output filenames")
    coordinates = output[["pred_lat", "pred_lng"]]
    if not np.isfinite(coordinates.to_numpy()).all():
        raise RuntimeError("Predictions contain non-finite coordinates")
    if not output.pred_lat.between(-90, 90).all():
        raise RuntimeError("Predicted latitude is outside [-90, 90]")
    if not output.pred_lng.between(-180, 180).all():
        raise RuntimeError("Predicted longitude is outside [-180, 180]")

    output_path = args.output.resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    temporary_output = output_path.with_suffix(".tmp.csv")
    output.to_csv(temporary_output, index=False)
    temporary_output.replace(output_path)
    report = {
        "rows": len(output),
        "columns": output.columns.tolist(),
        "unique_filenames": int(output.filename.nunique()),
        "device": str(device),
        "tta_rotations": 4,
        "inference_seconds": inference_seconds,
        "checkpoint": str(checkpoint_path),
        "predictions_sha256": hashlib.sha256(output_path.read_bytes()).hexdigest(),
    }
    output_path.with_suffix(".report.json").write_text(
        json.dumps(report, indent=2) + "\n"
    )
    print(
        f"Predictions complete: {output_path} ({inference_seconds / 60:.2f} min)",
        flush=True,
    )


if __name__ == "__main__":
    main()
