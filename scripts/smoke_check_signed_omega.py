#!/usr/bin/env python3
"""Smoke checks for SC-INR-Adaptive-Signed.

This is a fast pre-training gate:
  1. signed omega initialization matches the log-polar reference;
  2. omega is signed and bounded;
  3. forward/backward are finite;
  4. an existing softplus SC-INR-Adaptive checkpoint still loads.
"""

from __future__ import annotations

import argparse
import math
import sys
from pathlib import Path

import torch
import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import models
from models.sc_inr_adaptive import init_log_polar_freqs
from utils import make_coord


def load_model_from_config(config_path: Path):
    with config_path.open("r") as f:
        config = yaml.load(f, Loader=yaml.FullLoader)
    return models.make(config["model"]), config


def check_signed_init(model) -> None:
    ref = init_log_polar_freqs(
        model.num_freqs,
        model.num_angles,
        model.freq_min,
        model.freq_max,
    ).flatten()

    raw_bias = model.omega_conv.bias.detach().cpu()
    omega = model._parameterize_omega(raw_bias).detach().cpu()
    max_abs_err = float((omega - ref).abs().max())
    neg_frac = float((omega < 0).float().mean())
    pos_frac = float((omega > 0).float().mean())
    bound = float(model.omega_bound)

    print(f"omega_init min={float(omega.min()):.6f} max={float(omega.max()):.6f} "
          f"neg_frac={neg_frac:.3f} pos_frac={pos_frac:.3f} max_abs_err={max_abs_err:.6e}")

    if max_abs_err > 1e-5:
        raise AssertionError(f"signed omega init does not match reference: {max_abs_err}")
    if neg_frac <= 0 or pos_frac <= 0:
        raise AssertionError("signed omega should contain both negative and positive components")
    if float(omega.abs().max()) > bound + 1e-6:
        raise AssertionError("signed omega exceeds configured bound")


def check_forward_backward(model, device: torch.device) -> None:
    model = model.to(device)
    model.train()
    inp = torch.rand(1, 3, 48, 48, device=device)
    coord = make_coord([96, 96]).unsqueeze(0).to(device)
    coord = coord[:, :256, :].contiguous()
    cell = torch.ones_like(coord)
    cell[:, :, 0] *= 2 / 96
    cell[:, :, 1] *= 2 / 96
    target = torch.rand(1, coord.shape[1], 3, device=device)

    pred = model(inp, coord, cell)
    loss = torch.nn.functional.l1_loss(pred, target)
    loss.backward()

    if not torch.isfinite(pred).all():
        raise AssertionError("non-finite prediction")
    if not torch.isfinite(loss):
        raise AssertionError("non-finite loss")

    finite_grads = []
    for p in model.parameters():
        if p.grad is not None:
            finite_grads.append(torch.isfinite(p.grad).all())
    if not finite_grads or not all(bool(x) for x in finite_grads):
        raise AssertionError("non-finite gradient")
    pred_detached = pred.detach()
    print(
        f"forward_backward pred_range=({float(pred_detached.min()):.6f},"
        f"{float(pred_detached.max()):.6f}) loss={float(loss.detach()):.6f}"
    )


def check_scale_cells(model, device: torch.device) -> None:
    model.eval()
    inp = torch.rand(1, 3, 48, 48, device=device)
    coord = make_coord([96, 96]).unsqueeze(0).to(device)[:, :128, :].contiguous()
    with torch.no_grad():
        for scale in [2, 4, 8, 16]:
            cell = torch.ones_like(coord)
            size = 48 * scale
            cell[:, :, 0] *= 2 / size
            cell[:, :, 1] *= 2 / size
            pred = model(inp, coord, cell)
            if not torch.isfinite(pred).all():
                raise AssertionError(f"non-finite prediction for x{scale}")
            print(f"scale_cell x{scale}: pred_mean={float(pred.mean()):.6f} pred_std={float(pred.std()):.6f}")


def check_old_checkpoint(path: Path, device: torch.device) -> None:
    if not path.exists():
        print(f"old_checkpoint skipped: {path} not found")
        return
    ckpt = torch.load(path, map_location="cpu")
    model = models.make(ckpt["model"], load_sd=True, strict=False).to(device)
    omega_param = getattr(model, "omega_param", "unknown")
    if omega_param != "softplus":
        raise AssertionError(f"old sc_inr_adaptive checkpoint should load with softplus semantics, got {omega_param}")
    print(f"old_checkpoint loaded: {path} omega_param={omega_param}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=ROOT / "configs" / "train-div2k" / "train-sc-inr-adaptive-signed.yaml")
    parser.add_argument("--old_checkpoint", type=Path, default=ROOT / "save" / "sc-inr-adaptive" / "epoch-best.pth")
    parser.add_argument("--device", default="cuda:0")
    args = parser.parse_args()

    if args.device.startswith("cuda") and not torch.cuda.is_available():
        device = torch.device("cpu")
    else:
        device = torch.device(args.device)

    model, config = load_model_from_config(args.config)
    if config["model"]["name"] != "sc_inr_adaptive_signed":
        raise AssertionError("smoke config must use sc_inr_adaptive_signed")

    check_signed_init(model)
    check_forward_backward(model, device)
    check_scale_cells(model, device)
    check_old_checkpoint(args.old_checkpoint, device)
    print("signed omega smoke checks passed")


if __name__ == "__main__":
    main()
