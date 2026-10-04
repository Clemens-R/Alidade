#!/usr/bin/env python3
"""Cache the two ChiPoint retrieval teachers used by the final student."""

from __future__ import annotations

import argparse
import gc
import json
import os
import shutil
import time
from pathlib import Path

os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image, ImageFile, UnidentifiedImageError


MODEL_REPO = "chiikabu-labs/chipoint-2"
TOWER_SPECS = {
    "so400m256": {
        "model": "ViT-SO400M-16-SigLIP2-256",
        "input_dim": 1152,
        "gallery": "street_gallery/proj_head_gallery_512_so400m256.npy",
    },
    "gopt": {
        "model": "ViT-gopt-16-SigLIP2-256",
        "input_dim": 1536,
        "gallery": "street_gallery/proj_head_gallery_512_gopt.npy",
    },
}
RETRIEVAL_POOL = 200
GALLERY_CHUNK = 250_000
RETRIEVAL_QUERY_BATCH = 64
ImageFile.LOAD_TRUNCATED_IMAGES = True

class ProjHead(nn.Module):
    def __init__(self, input_dim: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 1024), nn.GELU(), nn.Dropout(0.1), nn.Linear(1024, 512)
        )

    def forward(self, x):
        return self.net(x)

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

def atomic_json(path: Path, value):
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, default=str))
    temporary.replace(path)

def load_projection_head(tag, input_dim, device):
    from huggingface_hub import hf_hub_download
    path = hf_hub_download(MODEL_REPO, f"weights/proj_head_{tag}.pt")
    checkpoint = torch.load(path, map_location="cpu", weights_only=True)
    head = ProjHead(input_dim)
    head.load_state_dict({k: v for k, v in checkpoint.items() if k.startswith("net.")})
    return head.to(device).half().eval()

def load_tower(tag, device):
    import open_clip

    spec = TOWER_SPECS[tag]
    print(f"[{tag}] Loading encoder and projection head...", flush=True)
    encoder, _, preprocess = open_clip.create_model_and_transforms(
        spec["model"], pretrained="webli"
    )
    encoder = encoder.to(device).half().eval()
    head = load_projection_head(tag, spec["input_dim"], device)

    @torch.inference_mode()
    def project(pixels):
        pixels = pixels.to(device, non_blocking=True).half()
        embedding = F.normalize(encoder.encode_image(pixels).float(), dim=-1)
        return F.normalize(head(embedding.half()).float(), dim=-1)

    return encoder, preprocess, head, project

def encode_queries(tag, frame, image_dir, batch_size, cache_dir, device):
    output_path = cache_dir / f"queries_{tag}_r1.npy"
    state_path = cache_dir / f"queries_{tag}_r1.json"
    expected_shape = (len(frame), 1, 512)
    if output_path.exists():
        query_map = np.lib.format.open_memmap(output_path, mode="r+")
        if query_map.shape != expected_shape:
            raise RuntimeError(f"Stale {output_path}: {query_map.shape} != {expected_shape}")
        start_at = (
            json.loads(state_path.read_text()).get("next_image", 0)
            if state_path.exists() else 0
        )
    else:
        query_map = np.lib.format.open_memmap(
            output_path, mode="w+", dtype=np.float16, shape=expected_shape
        )
        query_map[:] = np.nan
        query_map.flush()
        start_at = 0

    if start_at >= len(frame) and np.isfinite(query_map).all():
        print(f"[{tag}] Reusing complete query cache {output_path.name}", flush=True)
        return output_path

    model, preprocess, head, projected = load_tower(tag, device)
    started = time.monotonic()
    for start in range(start_at, len(frame), batch_size):
        end = min(start + batch_size, len(frame))
        paths = [image_dir / name for name in frame.filename.iloc[start:end]]
        for quarter in range(1):
            tensors = []
            for path in paths:
                with open_image(path) as image:
                    tensors.append(preprocess(rotate(image.convert("RGB"), quarter)))
            pixels = torch.stack(tensors)
            query = projected(pixels).cpu().numpy().astype(np.float16)
            if not np.isfinite(query).all():
                raise RuntimeError(f"{tag} produced non-finite query embeddings")
            query_map[start:end, quarter] = query
        query_map.flush()
        atomic_json(state_path, {"next_image": end, "shape": expected_shape})
        elapsed = time.monotonic() - started
        rate = (end - start_at) / max(elapsed, 1e-6)
        eta = (len(frame) - end) / max(rate, 1e-6) / 60
        print(
            f"[{tag} encode] {end}/{len(frame)} | "
            f"elapsed {elapsed/60:.1f} min | ETA {eta:.1f} min",
            flush=True,
        )

    del projected, head, model
    gc.collect()
    torch.cuda.empty_cache()
    return output_path

def save_retrieval_checkpoint(path, next_row, indices, similarities, settings):
    temporary = path.with_suffix(".tmp.npz")
    np.savez(
        temporary,
        next_gallery_row=np.asarray(next_row, dtype=np.int64),
        indices=indices,
        similarities=similarities,
        settings=np.asarray(json.dumps(settings)),
    )
    temporary.replace(path)


@torch.inference_mode()
def retrieve(tag, query_path, cache_dir, device):
    from huggingface_hub import hf_hub_download
    spec = TOWER_SPECS[tag]
    print(f"[{tag}] Downloading/opening 5 GB gallery...", flush=True)
    gallery_path = hf_hub_download(MODEL_REPO, spec["gallery"])
    gallery = np.load(gallery_path, mmap_mode="r")
    queries = np.load(query_path, mmap_mode="r").reshape(-1, 512)
    total_queries = len(queries)
    output_path = cache_dir / f"retrieval_{tag}_r1_m{RETRIEVAL_POOL}.npz"
    settings = {
        "tag": tag,
        "queries": total_queries,
        "gallery_rows": len(gallery),
        "pool_m": RETRIEVAL_POOL,
        "gallery_chunk": GALLERY_CHUNK,
    }
    if output_path.exists():
        old = np.load(output_path)
        old_settings = json.loads(str(old["settings"]))
        if old_settings != settings:
            raise RuntimeError(
                f"Stale retrieval cache {output_path}; use a new cache directory"
            )
        next_row = int(old["next_gallery_row"])
        best_idx = old["indices"].copy()
        best_sim = old["similarities"].copy()
        if next_row >= len(gallery):
            print(f"[{tag}] Reusing complete retrieval cache {output_path.name}", flush=True)
            return output_path
    else:
        next_row = 0
        best_idx = np.zeros((total_queries, RETRIEVAL_POOL), dtype=np.int32)
        best_sim = np.full((total_queries, RETRIEVAL_POOL), -np.inf, dtype=np.float32)

    started = time.monotonic()
    for gallery_start in range(next_row, len(gallery), GALLERY_CHUNK):
        gallery_end = min(gallery_start + GALLERY_CHUNK, len(gallery))
        block_np = gallery[gallery_start:gallery_end]
        if not block_np.flags["C_CONTIGUOUS"]:
            block_np = np.ascontiguousarray(block_np)
        block = torch.from_numpy(np.asarray(block_np)).to(device, non_blocking=True)
        for query_start in range(0, total_queries, RETRIEVAL_QUERY_BATCH):
            query_end = min(query_start + RETRIEVAL_QUERY_BATCH, total_queries)
            query = torch.from_numpy(
                np.asarray(queries[query_start:query_end])
            ).to(device, non_blocking=True)
            similarities = query @ block.T
            k = min(RETRIEVAL_POOL, similarities.shape[1])
            chunk_sim, chunk_idx = torch.topk(similarities, k, dim=1)
            chunk_idx += gallery_start
            previous_sim = torch.from_numpy(best_sim[query_start:query_end]).to(device)
            previous_idx = torch.from_numpy(
                best_idx[query_start:query_end].astype(np.int64)
            ).to(device)
            combined_sim = torch.cat((previous_sim, chunk_sim.float()), dim=1)
            combined_idx = torch.cat((previous_idx, chunk_idx), dim=1)
            keep_sim, keep_order = torch.topk(combined_sim, RETRIEVAL_POOL, dim=1)
            keep_idx = torch.gather(combined_idx, 1, keep_order)
            best_sim[query_start:query_end] = keep_sim.cpu().numpy()
            best_idx[query_start:query_end] = keep_idx.cpu().numpy().astype(np.int32)
            del query, similarities, chunk_sim, chunk_idx, previous_sim, previous_idx
            del combined_sim, combined_idx, keep_sim, keep_order, keep_idx
        del block
        save_retrieval_checkpoint(output_path, gallery_end, best_idx, best_sim, settings)
        elapsed = time.monotonic() - started
        done = gallery_end - next_row
        eta = elapsed * (len(gallery) - gallery_end) / max(done, 1) / 60
        print(
            f"[{tag} retrieve] {gallery_end:,}/{len(gallery):,} rows "
            f"({100*gallery_end/len(gallery):.1f}%) | elapsed {elapsed/60:.1f} min | ETA {eta:.1f} min",
            flush=True,
        )
    return output_path

def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, required=True)
    parser.add_argument("--cache-dir", type=Path, required=True)
    parser.add_argument("--batch-size", type=int, default=12)
    args = parser.parse_args()

    if not torch.cuda.is_available():
        raise SystemExit("CUDA is required for ChiPoint caching")

    data_dir = args.data_dir.resolve()
    cache_dir = args.cache_dir.resolve()
    cache_dir.mkdir(parents=True, exist_ok=True)
    frame = pd.read_csv(data_dir / "train_labels.csv")
    device = torch.device("cuda")
    torch.backends.cuda.matmul.allow_tf32 = True

    outputs = {}
    print(f"Caching ChiPoint teachers for {len(frame)} images", flush=True)
    for tag in TOWER_SPECS:
        query_path = encode_queries(
            tag, frame, data_dir / "train", args.batch_size, cache_dir, device
        )
        gc.collect()
        torch.cuda.empty_cache()
        retrieval_path = retrieve(tag, query_path, cache_dir, device)
        outputs[tag] = {
            "queries": str(query_path),
            "retrieval": str(retrieval_path),
        }
        gc.collect()
        torch.cuda.empty_cache()

    from huggingface_hub import hf_hub_download

    coordinates_source = hf_hub_download(
        MODEL_REPO, "street_gallery/gallery_all_C_so400m256.npy"
    )
    coordinates_path = cache_dir.parent / "chipoint_gallery_coords.npy"
    temporary = coordinates_path.with_suffix(".tmp.npy")
    shutil.copyfile(coordinates_source, temporary)
    temporary.replace(coordinates_path)

    manifest = {
        "filenames": frame.filename.tolist(),
        "towers": list(TOWER_SPECS),
        "rotations": 1,
        "pool_m": RETRIEVAL_POOL,
        "outputs": outputs,
    }
    atomic_json(cache_dir / "manifest.json", manifest)
    print("ChiPoint cache complete", flush=True)


if __name__ == "__main__":
    main()
