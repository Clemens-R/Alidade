#!/usr/bin/env python3
"""Convert fresh ChiPoint and PLONK outputs into student training targets."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch

import model


COUNTRIES = model.COUNTRIES
EARTH_KM = model.EARTH_KM

CHI_SIGMA_KM = 50.0
PLONK_SIGMA_KM = 200.0
FEATURE_DIM = 128
RANDOM_SEED = 42


def distance_km(true_xyz, pred_xyz):
    dot = np.sum(true_xyz * pred_xyz, axis=-1)
    return EARTH_KM * np.arccos(np.clip(dot, -1, 1))

def metric_dict(errors):
    errors = np.asarray(errors)
    return {
        "median_km": float(np.median(errors)),
        "mean_km": float(np.mean(errors)),
        "within_100_pct": float(100 * np.mean(errors < 100)),
        "within_200_pct": float(100 * np.mean(errors < 200)),
        "within_750_pct": float(100 * np.mean(errors < 750)),
    }

def zscore(values):
    values = np.asarray(values, dtype=np.float64)
    return (values - values.mean()) / (values.std() + 1e-9)

def logsumexp(values, axis):
    maximum = np.max(values, axis=axis, keepdims=True)
    return np.squeeze(maximum, axis=axis) + np.log(np.exp(values - maximum).sum(axis=axis))

def fuse_lists(index_lists, similarity_lists):
    score = {}
    for indices, similarities in zip(index_lists, similarity_lists):
        for index, value in zip(indices, zscore(similarities)):
            index = int(index)
            score[index] = score.get(index, 0.0) + float(value)
    ranked = sorted(score, key=score.get, reverse=True)[:64]
    return np.asarray(ranked, np.int32), np.asarray([score[x] for x in ranked], np.float32)

def normalized_logits(logits):
    logits = logits - logits.max(axis=-1, keepdims=True)
    return np.clip(logits, -30, 0).astype(np.float16)

def country_logits(cell_logits, cell_country):
    values = []
    for country_index in range(len(COUNTRIES)):
        values.append(logsumexp(cell_logits[:, cell_country == country_index], axis=1))
    return normalized_logits(np.stack(values, axis=1))

def random_projection(seed):
    generator = torch.Generator().manual_seed(seed)
    matrix = torch.randn(512, FEATURE_DIM, generator=generator)
    return torch.linalg.qr(matrix, mode="reduced").Q.numpy().astype(np.float32)

def build_teacher_targets(frame, centers, cell_country, teacher_cache, cache_path):
    fingerprint = {
        "filenames": frame.filename.tolist(),
        "center_checksum": float(np.asarray(centers, dtype=np.float64).sum()),
        "chi_sigma": CHI_SIGMA_KM,
        "plonk_sigma": PLONK_SIGMA_KM,
        "feature_dim": FEATURE_DIM,
    }
    if cache_path.exists():
        payload = torch.load(cache_path, map_location="cpu", weights_only=True)
        if payload.get("fingerprint") == fingerprint:
            print(f"Loaded teacher targets from {cache_path}", flush=True)
            return payload
        raise RuntimeError(
            f"Stale teacher targets at {cache_path}; use a new output path"
        )

    chip_dir = teacher_cache / "chipoint"
    manifest = json.loads((chip_dir / "manifest.json").read_text())
    if manifest["filenames"] != frame.filename.tolist():
        raise RuntimeError("ChiPoint cache ordering does not match train_labels.csv")
    query = {}
    retrieval = {}
    for tag in ("so400m256", "gopt"):
        query[tag] = np.load(chip_dir / f"queries_{tag}_r1.npy", mmap_mode="r")[:, 0]
        packed = np.load(chip_dir / f"retrieval_{tag}_r1_m200.npz")
        retrieval[tag] = (packed["indices"], packed["similarities"])

    coord_path = teacher_cache / "chipoint_gallery_coords.npy"
    if not coord_path.exists():
        raise RuntimeError(
            f"Missing {coord_path}. The run script should download this small coordinate table."
        )
    gallery_coords = np.load(coord_path, mmap_mode="r")
    plonk_manifest = json.loads((teacher_cache / "plonk/manifest.json").read_text())
    if plonk_manifest["filenames"] != frame.filename.tolist():
        raise RuntimeError("PLONK cache ordering does not match train_labels.csv")
    plonk_samples = np.load(teacher_cache / "plonk/plonk_all_samples.npy", mmap_mode="r")

    count = len(frame)
    cells = len(centers)
    chi_cell = np.empty((count, cells), dtype=np.float16)
    plonk_cell = np.empty((count, cells), dtype=np.float16)
    chi_error = np.empty(count, dtype=np.float32)
    plonk_error = np.empty(count, dtype=np.float32)
    truth = model.ll2xyz(frame.lat, frame.lng)
    print(f"Building teacher distributions for {count} training images...", flush=True)
    for index in range(count):
        candidate_indices, candidate_score = fuse_lists(
            [retrieval[tag][0][index] for tag in ("so400m256", "gopt")],
            [retrieval[tag][1][index] for tag in ("so400m256", "gopt")],
        )
        candidate_coords = np.asarray(gallery_coords[candidate_indices])
        candidate_xyz = model.ll2xyz(candidate_coords[:, 0], candidate_coords[:, 1])
        distances = EARTH_KM * np.arccos(np.clip(candidate_xyz @ centers.T, -1, 1))
        logits = logsumexp(
            zscore(candidate_score)[:, None] - 0.5 * (distances / CHI_SIGMA_KM) ** 2,
            axis=0,
        )
        chi_cell[index] = normalized_logits(logits[None])[0]
        chi_prediction = centers[int(np.argmax(logits))]
        chi_error[index] = distance_km(truth[index], chi_prediction)

        samples = np.asarray(plonk_samples[index]).reshape(-1, 2)
        sample_xyz = model.ll2xyz(samples[:, 0], samples[:, 1])
        sample_distances = EARTH_KM * np.arccos(np.clip(sample_xyz @ centers.T, -1, 1))
        logits = logsumexp(-0.5 * (sample_distances / PLONK_SIGMA_KM) ** 2, axis=0)
        plonk_cell[index] = normalized_logits(logits[None])[0]
        plonk_prediction = centers[int(np.argmax(logits))]
        plonk_error[index] = distance_km(truth[index], plonk_prediction)
        if (index + 1) % 500 == 0 or index + 1 == count:
            print(f"[teacher targets] {index + 1}/{count}", flush=True)

    so_projection = random_projection(RANDOM_SEED + 101)
    gopt_projection = random_projection(RANDOM_SEED + 202)
    so_feature = np.asarray(query["so400m256"], dtype=np.float32) @ so_projection
    gopt_feature = np.asarray(query["gopt"], dtype=np.float32) @ gopt_projection
    so_feature /= np.clip(np.linalg.norm(so_feature, axis=1, keepdims=True), 1e-8, None)
    gopt_feature /= np.clip(np.linalg.norm(gopt_feature, axis=1, keepdims=True), 1e-8, None)
    payload = {
        "fingerprint": fingerprint,
        "chi_cell_logits": torch.from_numpy(chi_cell),
        "plonk_cell_logits": torch.from_numpy(plonk_cell),
        "chi_country_logits": torch.from_numpy(
            country_logits(chi_cell.astype(np.float32), cell_country)
        ),
        "plonk_country_logits": torch.from_numpy(
            country_logits(plonk_cell.astype(np.float32), cell_country)
        ),
        "chi_feature": torch.from_numpy(so_feature.astype(np.float16)),
        "gopt_feature": torch.from_numpy(gopt_feature.astype(np.float16)),
        "chi_quality": torch.from_numpy(
            np.maximum(0.1, np.exp(-chi_error / 500)).astype(np.float32)
        ),
        "plonk_quality": torch.from_numpy(
            np.maximum(0.1, np.exp(-plonk_error / 500)).astype(np.float32)
        ),
        "diagnostics": {
            "chi": metric_dict(chi_error),
            "plonk": metric_dict(plonk_error),
            "oracle": metric_dict(np.minimum(chi_error, plonk_error)),
        },
    }
    temporary = cache_path.with_suffix(".tmp.pt")
    torch.save(payload, temporary)
    temporary.replace(cache_path)
    print(
        "Teacher diagnostics:",
        json.dumps(payload["diagnostics"], indent=2),
        flush=True,
    )
    return payload

def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, required=True)
    parser.add_argument("--teacher-cache", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    data_dir = args.data_dir.resolve()
    teacher_cache = args.teacher_cache.resolve()
    output = args.output.resolve()
    output.parent.mkdir(parents=True, exist_ok=True)

    frame = pd.read_csv(data_dir / "train_labels.csv")
    centers, _, cell_country = model.make_cells(frame)
    payload = build_teacher_targets(
        frame, centers, cell_country, teacher_cache, output
    )

    expected_shape = (len(frame), len(centers))
    for name in ("chi_cell_logits", "plonk_cell_logits"):
        if tuple(payload[name].shape) != expected_shape:
            raise RuntimeError(
                f"{name} has shape {tuple(payload[name].shape)}, expected {expected_shape}"
            )
    print(f"Teacher targets ready: {output}", flush=True)


if __name__ == "__main__":
    main()
