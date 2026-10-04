#!/usr/bin/env python3
"""Verify the model parameter count and, optionally, a predictions file."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch

import model as geo


EXPECTED = 4_964_188

def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--predictions", type=Path)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--expected-rows", type=int)
    args = parser.parse_args()
    checkpoint = torch.load(
        args.checkpoint.resolve(), map_location="cpu", weights_only=True
    )
    model = geo.AdapterMoE(imagenet_init=False)
    parameter_names = {name for name, _ in model.named_parameters()}
    model_count = sum(parameter.numel() for parameter in model.parameters())
    checkpoint_count = sum(checkpoint["model"][name].numel() for name in parameter_names)
    declared_count = int(checkpoint.get("parameter_count", checkpoint.get("parameters", -1)))
    state_count = sum(value.numel() for value in checkpoint["model"].values())
    center_count = int(np.asarray(checkpoint["centers"]).size)
    complete_inference_state = state_count + center_count

    groups = {
        "backbone": ("blocks_", "final_conv."),
        "router": ("router.",),
        "convolutional_expert_adapters": ("adapter_",),
        "shared_neck": ("neck.",),
        "shared_geocell_head": ("cell_shared.",),
        "expert_geocell_deltas": ("cell_delta.",),
        "country_head": ("country.",),
        "coordinate_residual_head": ("residual.",),
        "training_only_distillation_projections": ("so_feature.", "gopt_feature."),
    }
    counts = {name: 0 for name in groups}
    for name, parameter in model.named_parameters():
        matches = [group for group, prefixes in groups.items() if name.startswith(prefixes)]
        if len(matches) != 1:
            raise RuntimeError(f"Parameter {name} matched {matches}")
        counts[matches[0]] += parameter.numel()
    if not (model_count == checkpoint_count == declared_count == sum(counts.values()) == EXPECTED):
        raise RuntimeError(
            f"Parameter counts disagree: reconstructed={model_count}, "
            f"checkpoint tensors={checkpoint_count}, "
            f"declared={declared_count}, grouped={sum(counts.values())}, expected={EXPECTED}"
        )
    if model_count > 5_000_000:
        raise RuntimeError("Five-million-parameter limit exceeded")
    if complete_inference_state > 5_000_000:
        raise RuntimeError("Complete inference state exceeds five million values")

    predictions = None
    if args.predictions is not None:
        predictions = pd.read_csv(args.predictions.resolve())
        if predictions.columns.tolist() != ["filename", "pred_lat", "pred_lng"]:
            raise RuntimeError(
                f"Wrong predictions.csv columns: {predictions.columns.tolist()}"
            )
        if predictions.filename.nunique() != len(predictions):
            raise RuntimeError("Prediction filenames must be unique")
        if args.expected_rows is not None and len(predictions) != args.expected_rows:
            raise RuntimeError(
                f"Expected {args.expected_rows:,} predictions, found {len(predictions):,}"
            )
    elif args.expected_rows is not None:
        parser.error("--expected-rows requires --predictions")

    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    table = pd.DataFrame([
        {"component": name, "parameters": value, "percent_of_total": 100 * value / model_count}
        for name, value in counts.items()
    ])
    table.to_csv(output_dir / "parameter_breakdown_final.csv", index=False)
    report = {
        "verification": "PASS",
        "reconstructed_model_parameters": model_count,
        "checkpoint_parameter_tensor_elements": checkpoint_count,
        "checkpoint_declared_parameters": declared_count,
        "model_state_dict_values": state_count,
        "geocell_center_values": center_count,
        "complete_inference_state_values": complete_inference_state,
        "parameter_limit": 5_000_000,
        "headroom": 5_000_000 - model_count,
        "component_sum": int(table.parameters.sum()),
    }
    if predictions is not None:
        report.update({
            "predictions_rows": len(predictions),
            "predictions_columns": predictions.columns.tolist(),
        })
    (output_dir / "parameter_report.json").write_text(json.dumps(report, indent=2) + "\n")
    text = (
        "FINAL ARTIFACT VERIFICATION: PASS\n"
        f"Parameters: {model_count:,} / 5,000,000\n"
        f"Parameter headroom: {5_000_000-model_count:,}\n"
        f"Complete inference state: {complete_inference_state:,} values "
        f"({5_000_000-complete_inference_state:,} below the limit)\n"
        "Count independently agrees across reconstructed model, named checkpoint tensors, "
        "checkpoint declaration, and component sum.\n"
    )
    if predictions is not None:
        text += (
            f"predictions.csv: {len(predictions):,} unique rows with exact required header.\n"
        )
    text += "\n" + table.to_string(index=False) + "\n"
    (output_dir / "PARAMETER_REPORT.txt").write_text(text)
    print(text, flush=True)


if __name__ == "__main__":
    main()
