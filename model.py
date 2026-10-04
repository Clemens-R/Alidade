"""Model, data loading, decoding, and losses for the final student."""

from __future__ import annotations

import math
import random
import time

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image, ImageFile, UnidentifiedImageError
from torch.utils.data import Dataset
from torchvision import transforms
from torchvision.models import MobileNet_V3_Large_Weights, mobilenet_v3_large


COUNTRIES = [
    "Belarus", "Finland", "France", "Germany", "Iceland", "Italy",
    "Norway", "Poland", "Spain", "Sweden", "Turkey", "United_Kingdom",
]
EARTH_KM = 6371.0088
IMAGE_READ_ATTEMPTS = 4
ImageFile.LOAD_TRUNCATED_IMAGES = True

FINAL_MODEL_CONFIG = {
    "cells": 1536,
    "experts": 12,
    "adapter_rank": 24,
    "head_rank": 24,
    "top_experts": 3,
    "residual_scale": 0.05,
    "feature_dim": 128,
    "image_size": 512,
    "decode_top_k": 3,
    "decode_radius_km": 300.0,
}

def set_seed(value: int) -> None:
    random.seed(value)
    np.random.seed(value)
    torch.manual_seed(value)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(value)

def ll2xyz(lat, lon):
    lat = np.radians(np.asarray(lat, dtype=np.float64))
    lon = np.radians(np.asarray(lon, dtype=np.float64))
    return np.stack(
        (np.cos(lat) * np.cos(lon), np.cos(lat) * np.sin(lon), np.sin(lat)), axis=-1
    ).astype(np.float32)

def xyz2ll(xyz):
    xyz = np.asarray(xyz)
    xyz = xyz / np.clip(np.linalg.norm(xyz, axis=-1, keepdims=True), 1e-8, None)
    return (
        np.degrees(np.arcsin(np.clip(xyz[..., 2], -1, 1))),
        np.degrees(np.arctan2(xyz[..., 1], xyz[..., 0])),
    )

class RandomQuarterTurn:
    def __call__(self, image):
        quarter = random.randrange(4)
        if quarter == 0:
            return image
        methods = (
            Image.Transpose.ROTATE_90,
            Image.Transpose.ROTATE_180,
            Image.Transpose.ROTATE_270,
        )
        return image.transpose(methods[quarter - 1])

def make_transforms():
    image_size = FINAL_MODEL_CONFIG["image_size"]
    normalize = transforms.Normalize(
        [0.485, 0.456, 0.406], [0.229, 0.224, 0.225]
    )
    teacher_transform = transforms.Compose([
        transforms.Resize((image_size, image_size)),
        transforms.ToTensor(),
        normalize,
    ])
    training_transform = transforms.Compose([
        RandomQuarterTurn(),
        transforms.Resize((image_size, image_size)),
        transforms.RandomHorizontalFlip(),
        transforms.ColorJitter(0.12, 0.12, 0.08, 0.025),
        transforms.ToTensor(),
        normalize,
    ])
    return teacher_transform, training_transform

class GeoDataset(Dataset):
    def __init__(self, frame, image_dir, transform, country_map, soft_cells=None):
        self.frame = frame.reset_index(drop=True)
        self.image_dir = image_dir
        self.transform = transform
        self.xyz = ll2xyz(self.frame.lat, self.frame.lng)
        self.country = np.asarray(
            [country_map[x] for x in self.frame.country], dtype=np.int64
        )
        self.soft_cells = soft_cells

    def __len__(self):
        return len(self.frame)

    def __getitem__(self, index):
        path = self.image_dir / self.frame.filename.iloc[index]
        last_error = None
        for attempt in range(IMAGE_READ_ATTEMPTS):
            try:
                with Image.open(path) as source:
                    image = self.transform(source.convert("RGB"))
                break
            except (OSError, UnidentifiedImageError) as error:
                last_error = error
                if attempt + 1 < IMAGE_READ_ATTEMPTS:
                    time.sleep(0.5 * (attempt + 1))
        else:
            raise RuntimeError(
                f"Could not decode {path} after {IMAGE_READ_ATTEMPTS} attempts"
            ) from last_error

        soft = (
            np.empty(0, dtype=np.float32)
            if self.soft_cells is None else self.soft_cells[index]
        )
        return image, self.xyz[index], self.country[index], soft, index

class ConvAdapterBank(nn.Module):
    """Per-expert low-rank residual adapters for a feature map."""

    def __init__(self, channels, experts, rank):
        super().__init__()
        self.experts = experts
        self.down = nn.ModuleList([
            nn.Conv2d(channels, rank, 1, bias=False) for _ in range(experts)
        ])
        self.up = nn.ModuleList([
            nn.Conv2d(rank, channels, 1, bias=False) for _ in range(experts)
        ])
        # Zero-initialized up-projections make each adapter an identity at
        # startup; a unit scale still lets the up-projections receive gradients.
        self.scale = nn.Parameter(torch.ones(experts))
        for layer in self.up:
            nn.init.zeros_(layer.weight)

    def forward(self, x, gates):
        if self.experts == 1:
            update = self.up[0](F.hardswish(self.down[0](x)))
            return x + self.scale[0] * update
        update = None
        for expert, (down, up) in enumerate(zip(self.down, self.up)):
            value = up(F.hardswish(down(x)))
            weight = gates[:, expert, None, None, None] * self.scale[expert]
            value = value * weight
            update = value if update is None else update + value
        return x + update

class ExpertLinearDelta(nn.Module):
    """Gated low-rank deltas added to a shared linear prediction head."""

    def __init__(self, input_dim, output_dim, experts, rank):
        super().__init__()
        self.experts = experts
        self.down = nn.ModuleList([
            nn.Linear(input_dim, rank, bias=False) for _ in range(experts)
        ])
        self.up = nn.ModuleList([
            nn.Linear(rank, output_dim, bias=False) for _ in range(experts)
        ])
        for layer in self.up:
            nn.init.zeros_(layer.weight)

    def forward(self, hidden, gates):
        result = None
        for expert, (down, up) in enumerate(zip(self.down, self.up)):
            value = up(F.hardswish(down(hidden))) * gates[:, expert, None]
            result = value if result is None else result + value
        return result

def sparse_gates(logits, top_experts):
    probabilities = F.softmax(logits, dim=1)
    if top_experts >= logits.shape[1]:
        return probabilities
    _, indices = probabilities.topk(top_experts, dim=1)
    mask = torch.zeros_like(probabilities).scatter_(1, indices, 1)
    probabilities = probabilities * mask
    return probabilities / probabilities.sum(1, keepdim=True).clamp_min(1e-8)

class AdapterMoE(nn.Module):
    def __init__(self, imagenet_init=False):
        super().__init__()
        config = FINAL_MODEL_CONFIG
        cells = config["cells"]
        experts = config["experts"]
        adapter_rank = config["adapter_rank"]
        head_rank = config["head_rank"]
        top_experts = config["top_experts"]
        residual_scale = config["residual_scale"]
        feature_dim = config["feature_dim"]
        if cells % len(COUNTRIES):
            raise ValueError("cells must be divisible by 12")
        weights = MobileNet_V3_Large_Weights.DEFAULT if imagenet_init else None
        backbone = mobilenet_v3_large(weights=weights)
        blocks = list(backbone.features.children())
        self.blocks_early = nn.ModuleList(blocks[:11])
        self.blocks_middle = nn.ModuleList(blocks[11:13])
        self.blocks_late = nn.ModuleList(blocks[13:16])
        self.final_conv = blocks[16]
        self.experts = experts
        self.top_experts = min(top_experts, experts)
        early_channels = self._infer_channels(self.blocks_early)
        middle_channels = self._infer_channels(self.blocks_middle)
        late_channels = self._infer_channels(self.blocks_late)
        final_channels = backbone.classifier[0].in_features
        self.router = nn.Linear(early_channels, experts)
        self.adapter_early = ConvAdapterBank(early_channels, experts, adapter_rank)
        self.adapter_middle = ConvAdapterBank(middle_channels, experts, adapter_rank)
        self.adapter_late = ConvAdapterBank(late_channels, experts, adapter_rank)
        self.adapter_final = ConvAdapterBank(final_channels, experts, adapter_rank)
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.neck = nn.Sequential(
            nn.Linear(final_channels, 256),
            nn.LayerNorm(256),
            nn.Hardswish(),
            nn.Dropout(0.12),
        )
        self.cell_shared = nn.Linear(256, cells)
        self.cell_delta = ExpertLinearDelta(256, cells, experts, head_rank)
        self.country = nn.Linear(256, len(COUNTRIES))
        self.residual = nn.Linear(256, len(COUNTRIES) * 3)
        self.so_feature = nn.Linear(256, feature_dim)
        self.gopt_feature = nn.Linear(256, feature_dim)
        self.per_country = cells // len(COUNTRIES)
        self.residual_scale = residual_scale

    @staticmethod
    def _infer_channels(blocks):
        # The last BatchNorm in the final Conv2dNormActivation or inverted
        # residual block exposes the output channel count without a dummy pass.
        for module in reversed(list(blocks.modules())):
            if isinstance(module, nn.BatchNorm2d):
                return module.num_features
        raise RuntimeError("Could not infer MobileNet block channels")

    @staticmethod
    def _run(blocks, x):
        for block in blocks:
            x = block(x)
        return x

    def forward(self, images):
        x = self._run(self.blocks_early, images)
        router_logits = self.router(self.pool(x).flatten(1))
        gates = sparse_gates(router_logits, self.top_experts)
        x = self.adapter_early(x, gates)
        x = self._run(self.blocks_middle, x)
        x = self.adapter_middle(x, gates)
        x = self._run(self.blocks_late, x)
        x = self.adapter_late(x, gates)
        x = self.final_conv(x)
        x = self.adapter_final(x, gates)
        hidden = self.neck(self.pool(x).flatten(1))
        cell_logits = self.cell_shared(hidden) + self.cell_delta(hidden, gates)
        country_logits = self.country(hidden)
        residual = self.residual(hidden).view(-1, len(COUNTRIES), 3)
        return {
            "cell": cell_logits,
            "country": country_logits,
            "residual": residual,
            "router": router_logits,
            "gates": gates,
            "hidden": hidden,
            "so_feature": F.normalize(self.so_feature(hidden), dim=1),
            "gopt_feature": F.normalize(self.gopt_feature(hidden), dim=1),
        }

    def coordinates_by_country(self, output, centers):
        batch = output["cell"].shape[0]
        logits = output["cell"].view(batch, len(COUNTRIES), self.per_country)
        country_centers = centers.view(len(COUNTRIES), self.per_country, 3)
        take = min(FINAL_MODEL_CONFIG["decode_top_k"], self.per_country)
        values, indices = logits.topk(take, dim=2)
        chosen = country_centers[None].expand(batch, -1, -1, -1).gather(
            2, indices[..., None].expand(-1, -1, -1, 3)
        )
        base = (values.softmax(2)[..., None] * chosen).sum(2)
        prediction = F.normalize(
            base + self.residual_scale * torch.tanh(output["residual"]), dim=2
        )
        return prediction

    def decode(self, output, centers):
        """Choose the strongest p(country) p(cell | country) geographic mode."""
        batch = output["cell"].shape[0]
        joint_log_prob = hierarchical_log_prob(output, self.per_country)
        anchor = joint_log_prob.flatten(1).argmax(1)
        chosen_country = torch.div(anchor, self.per_country, rounding_mode="floor")
        anchor_local = anchor.remainder(self.per_country)

        row = torch.arange(batch, device=centers.device)
        conditional = joint_log_prob[row, chosen_country]
        country_centers = centers.view(len(COUNTRIES), self.per_country, 3)
        selected_centers = country_centers[chosen_country]
        local_cos = math.cos(FINAL_MODEL_CONFIG["decode_radius_km"] / EARTH_KM)
        local = torch.einsum(
            "bd,bkd->bk", selected_centers[row, anchor_local], selected_centers
        ) >= local_cos
        coherent = conditional.masked_fill(~local, -torch.inf)
        take = min(FINAL_MODEL_CONFIG["decode_top_k"], self.per_country)
        values, indices = coherent.topk(take, dim=1)
        chosen = selected_centers.gather(
            1, indices[..., None].expand(-1, -1, 3)
        )
        base = (values.softmax(1)[..., None] * chosen).sum(1)
        residual = output["residual"][row, chosen_country]
        prediction = F.normalize(
            base + self.residual_scale * torch.tanh(residual), dim=1
        )
        return prediction

def make_cells(frame):
    from sklearn.cluster import MiniBatchKMeans

    count = FINAL_MODEL_CONFIG["cells"]
    seed = 43
    sigma_km = 50.0
    if count % len(COUNTRIES):
        raise ValueError("--cells must be divisible by 12")
    per_country = count // len(COUNTRIES)
    centers, cell_country = [], []
    soft = np.zeros((len(frame), count), dtype=np.float32)
    for country_index, country in enumerate(COUNTRIES):
        rows = np.flatnonzero(frame.country.to_numpy() == country)
        xyz = ll2xyz(frame.lat.to_numpy()[rows], frame.lng.to_numpy()[rows])
        kmeans = MiniBatchKMeans(
            per_country, batch_size=512, n_init=10, random_state=seed + country_index
        ).fit(xyz)
        country_centers = kmeans.cluster_centers_.astype(np.float32)
        country_centers /= np.linalg.norm(country_centers, axis=1, keepdims=True)
        distances = EARTH_KM * np.arccos(np.clip(xyz @ country_centers.T, -1, 1))
        take = min(8, per_country)
        nearest = np.argpartition(distances, take - 1, axis=1)[:, :take]
        local = np.zeros_like(distances, dtype=np.float32)
        rr = np.arange(len(rows))[:, None]
        local[rr, nearest] = np.exp(-0.5 * (distances[rr, nearest] / sigma_km) ** 2)
        local /= np.clip(local.sum(1, keepdims=True), 1e-8, None)
        start = country_index * per_country
        soft[rows, start:start + per_country] = local
        centers.append(country_centers)
        cell_country.extend([country_index] * per_country)
    return np.concatenate(centers), soft, np.asarray(cell_country, dtype=np.int64)

def robust_angular_loss(prediction, truth):
    dot = (prediction.float() * truth.float()).sum(1).clamp(-1 + 1e-6, 1 - 1e-6)
    cross = torch.linalg.vector_norm(
        torch.linalg.cross(prediction.float(), truth.float()), dim=1
    )
    angle = torch.atan2(cross, dot)
    return F.smooth_l1_loss(angle, torch.zeros_like(angle), beta=0.02)

def kd_loss(student_logits, teacher_prob):
    temperature = 2.0
    log_prob = F.log_softmax(student_logits.float() / temperature, dim=1)
    return -(teacher_prob.float() * log_prob).sum(1).mean() * temperature**2

def load_teacher_targets(path, train_frame, centers):
    """Load and fuse the two teachers using their per-image training quality."""
    temperature = 2.0
    if not path.is_file():
        raise FileNotFoundError(path)
    payload = torch.load(path, map_location="cpu", weights_only=True)
    fingerprint = payload.get("fingerprint", {})
    if fingerprint.get("filenames") != train_frame.filename.tolist():
        raise RuntimeError("Teacher targets do not match train_labels.csv")
    checksum = fingerprint.get("center_checksum")
    expected_checksum = float(np.asarray(centers, np.float64).sum())
    if checksum is None or abs(float(checksum) - expected_checksum) > 1e-5:
        raise RuntimeError("Teacher targets use a different geocell layout")

    chi_score = 0.5 * payload["chi_quality"].float().clamp_min(1e-6).square()
    plonk_score = 0.5 * payload["plonk_quality"].float().clamp_min(1e-6).square()
    chi_weight = chi_score / (chi_score + plonk_score).clamp_min(1e-8)
    plonk_weight = 1 - chi_weight

    chi_cell = F.softmax(payload["chi_cell_logits"].float() / temperature, dim=1)
    plonk_cell = F.softmax(payload["plonk_cell_logits"].float() / temperature, dim=1)
    payload["fused_cell_prob"] = (
        chi_weight[:, None] * chi_cell + plonk_weight[:, None] * plonk_cell
    ).to(torch.float16)

    chi_country = F.softmax(payload["chi_country_logits"].float() / temperature, dim=1)
    plonk_country = F.softmax(payload["plonk_country_logits"].float() / temperature, dim=1)
    payload["fused_country_prob"] = (
        chi_weight[:, None] * chi_country + plonk_weight[:, None] * plonk_country
    ).to(torch.float16)
    print(
        f"Teacher mixture: ChiPoint={chi_weight.mean():.3f}, "
        f"PLONK={plonk_weight.mean():.3f}",
        flush=True,
    )
    return payload

def load_balance_loss(gates):
    if gates.shape[1] == 1:
        return gates.new_zeros(())
    usage = gates.mean(0)
    target = torch.full_like(usage, 1 / gates.shape[1])
    return F.mse_loss(usage, target) * gates.shape[1]

def hierarchical_log_prob(output, per_country):
    """Return log p(country, cell) and log p(country)."""
    batch = output["cell"].shape[0]
    cell_log_prob = F.log_softmax(
        output["cell"].float().view(batch, len(COUNTRIES), per_country), dim=2
    )
    country_score = F.log_softmax(output["country"].float(), dim=1)
    country_log_prob = country_score - torch.logsumexp(
        country_score, dim=1, keepdim=True
    )
    return country_log_prob[:, :, None] + cell_log_prob

def conditional_cell_loss(cell_logits, soft_cells):
    """Cross-entropy for p(cell | country), without competing countries."""
    batch = cell_logits.shape[0]
    per_country = cell_logits.shape[1] // len(COUNTRIES)
    log_prob = F.log_softmax(
        cell_logits.float().view(batch, len(COUNTRIES), per_country), dim=2
    )
    target = soft_cells.float().view(batch, len(COUNTRIES), per_country)
    return -(target * log_prob).sum((1, 2)).mean()

def atomic_save(payload, path):
    temporary = path.with_suffix(path.suffix + ".tmp")
    torch.save(payload, temporary)
    temporary.replace(path)
