#!/usr/bin/env python3
"""Cache PLONK coordinate samples for every labeled training image."""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from PIL import Image, ImageFile, UnidentifiedImageError


MODEL_NAME = "nicolas-dufour/PLONK_OSV_5M"
SAMPLES_PER_ROTATION = 4
SAMPLING_STEPS = 250
ROTATIONS = 4
RANDOM_SEED = 42
ImageFile.LOAD_TRUNCATED_IMAGES = True

def rotate(image, quarter_turns):
    if quarter_turns == 0:
        return image
    methods = (Image.Transpose.ROTATE_90, Image.Transpose.ROTATE_180, Image.Transpose.ROTATE_270)
    return image.transpose(methods[quarter_turns - 1])

def open_image(path: Path):
    """Open an image, retrying transient network-filesystem failures."""
    last_error = None
    for attempt in range(4):
        try:
            image = Image.open(path)
            image.load()
            return image
        except (OSError, UnidentifiedImageError) as error:
            last_error = error
            if attempt < 3:
                time.sleep(0.5 * (attempt + 1))
    raise RuntimeError(f"Could not decode {path} after four attempts") from last_error


@torch.inference_mode()
def sample_from_embeddings(pipe, embeddings, generator):
    """Use PLONK's official sampler while avoiding repeated image encoding."""
    expanded = embeddings.repeat_interleave(SAMPLES_PER_ROTATION, dim=0)
    noise = torch.randn(expanded.shape[0], 3, device=pipe.device, generator=generator)
    batch = {"y": noise, "emb": expanded}
    xyz = pipe.sampler(
        pipe.model,
        batch,
        conditioning_keys="emb",
        scheduler=pipe.scheduler,
        num_steps=SAMPLING_STEPS,
        cfg_rate=0,
        generator=generator,
    )
    gps = pipe.postprocessing(xyz)
    return np.degrees(gps.detach().cpu().numpy()).reshape(len(embeddings), SAMPLES_PER_ROTATION, 2)

def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, required=True)
    parser.add_argument("--cache-dir", type=Path, required=True)
    parser.add_argument("--batch-size", type=int, default=8)
    args = parser.parse_args()

    if not torch.cuda.is_available():
        raise SystemExit("CUDA is required for PLONK caching")

    data_dir = args.data_dir.resolve()
    cache_dir = args.cache_dir.resolve()
    batch_dir = cache_dir / "batches"
    batch_dir.mkdir(parents=True, exist_ok=True)
    frame = pd.read_csv(data_dir / "train_labels.csv")
    final_path = cache_dir / "plonk_all_samples.npy"
    expected_shape = (len(frame), ROTATIONS, SAMPLES_PER_ROTATION, 2)

    if final_path.exists():
        cached = np.load(final_path, mmap_mode="r")
        if cached.shape == expected_shape:
            print(f"Reusing complete PLONK cache {final_path}", flush=True)
            return
        raise RuntimeError(
            f"Stale PLONK cache has shape {cached.shape}, expected {expected_shape}"
        )

    from plonk import PlonkPipeline

    print(f"Loading {MODEL_NAME}...", flush=True)
    pipeline = PlonkPipeline(MODEL_NAME, device="cuda")
    started = time.monotonic()
    processed_this_run = 0

    for start in range(0, len(frame), args.batch_size):
        end = min(start + args.batch_size, len(frame))
        batch_path = batch_dir / f"{start:05d}_{end:05d}.npz"
        if batch_path.exists():
            continue

        image_paths = [data_dir / "train" / name for name in frame.filename.iloc[start:end]]
        rotation_samples = []
        for quarter in range(ROTATIONS):
            images = []
            for image_path in image_paths:
                with open_image(image_path) as image:
                    images.append(rotate(image.convert("RGB"), quarter))
            embeddings = pipeline.cond_preprocessing({"img": images})["emb"]
            generator = torch.Generator(device="cuda").manual_seed(
                RANDOM_SEED * 1_000_003 + start * 17 + quarter
            )
            rotation_samples.append(
                sample_from_embeddings(pipeline, embeddings, generator)
            )

        samples = np.stack(rotation_samples, axis=1).astype(np.float32)
        temporary = batch_path.with_suffix(".tmp.npz")
        np.savez_compressed(temporary, samples=samples)
        temporary.replace(batch_path)
        processed_this_run += end - start
        elapsed = time.monotonic() - started
        rate = processed_this_run / max(elapsed, 1e-6)
        eta_minutes = (len(frame) - end) / max(rate, 1e-6) / 60
        print(
            f"[PLONK] {end}/{len(frame)} | elapsed {elapsed/60:.1f} min "
            f"| ETA {eta_minutes:.1f} min",
            flush=True,
        )

    chunks = []
    for start in range(0, len(frame), args.batch_size):
        end = min(start + args.batch_size, len(frame))
        chunks.append(np.load(batch_dir / f"{start:05d}_{end:05d}.npz")["samples"])
    samples = np.concatenate(chunks, axis=0)
    if samples.shape != expected_shape:
        raise RuntimeError(f"Combined PLONK shape {samples.shape}, expected {expected_shape}")

    temporary = final_path.with_suffix(".tmp.npy")
    np.save(temporary, samples)
    temporary.replace(final_path)
    manifest = {
        "filenames": frame.filename.tolist(),
        "model": MODEL_NAME,
        "shape": list(samples.shape),
        "samples_per_rotation": SAMPLES_PER_ROTATION,
        "rotations": ROTATIONS,
        "steps": SAMPLING_STEPS,
    }
    (cache_dir / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print("PLONK cache complete", flush=True)


if __name__ == "__main__":
    main()
