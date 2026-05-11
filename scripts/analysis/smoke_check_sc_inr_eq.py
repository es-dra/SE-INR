#!/usr/bin/env python3
"""Smoke checks for the exploratory SC-INR-EQ model.

This is a fast pre-training gate:
  1. config builds the intended sc_inr_eq model;
  2. signed omega initialization matches the per-transform log-polar reference;
  3. forward/backward are finite;
  4. different cell sizes produce finite outputs;
  5. the analytic response is non-constant when sinc is enabled.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch
import yaml

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

import models
from models.sc_inr_adaptive import init_log_polar_freqs
from utils import make_coord


def load_model_from_config(config_path: Path):
    with config_path.open("r") as f:
        config = yaml.load(f, Loader=yaml.FullLoader)
    return models.make(config["model"]), config


def check_omega_init(model) -> None:
    ref = init_log_polar_freqs(
        model.num_freqs_per_tran,
        model.num_angles,
        model.freq_min,
        model.freq_max,
    )
    ref_flat = ref.t().contiguous().view(-1)
    raw_bias = model.omega_conv.c.detach().cpu().view(-1)
    omega = model._parameterize_omega(raw_bias).detach().cpu()
    max_abs_err = float((omega - ref_flat).abs().max())
    neg_frac = float((omega < 0).float().mean())
    pos_frac = float((omega > 0).float().mean())

    print(
        f"omega_init per_tran={model.num_freqs_per_tran} "
        f"min={float(omega.min()):.6f} max={float(omega.max()):.6f} "
        f"neg_frac={neg_frac:.3f} pos_frac={pos_frac:.3f} "
        f"max_abs_err={max_abs_err:.6e}"
    )
    if max_abs_err > 1e-5:
        raise AssertionError(f"omega init mismatch: {max_abs_err}")
    if neg_frac <= 0 or pos_frac <= 0:
        raise AssertionError("omega init should contain positive and negative components")


def make_batch(device: torch.device, q: int = 256):
    inp = torch.rand(1, 3, 48, 48, device=device)
    coord = make_coord([96, 96]).unsqueeze(0).to(device)
    coord = coord[:, :q, :].contiguous()
    cell = torch.ones_like(coord)
    cell[:, :, 0] *= 2 / 96
    cell[:, :, 1] *= 2 / 96
    target = torch.rand(1, coord.shape[1], 3, device=device)
    return inp, coord, cell, target


def check_forward_backward(model, device: torch.device) -> None:
    model = model.to(device)
    model.train()
    inp, coord, cell, target = make_batch(device)
    pred = model(inp, coord, cell)
    loss = torch.nn.functional.l1_loss(pred, target)
    loss.backward()

    if pred.shape != target.shape:
        raise AssertionError(f"unexpected pred shape: {tuple(pred.shape)}")
    if not torch.isfinite(pred).all():
        raise AssertionError("non-finite prediction")
    if not torch.isfinite(loss):
        raise AssertionError("non-finite loss")

    finite_grads = [
        torch.isfinite(p.grad).all()
        for p in model.parameters()
        if p.grad is not None
    ]
    if not finite_grads or not all(bool(x) for x in finite_grads):
        raise AssertionError("non-finite gradient")

    pred_detached = pred.detach()
    print(
        f"forward_backward shape={tuple(pred.shape)} "
        f"pred_range=({float(pred_detached.min()):.6f},"
        f"{float(pred_detached.max()):.6f}) loss={float(loss.detach()):.6f}"
    )


def check_scale_cells(model, device: torch.device) -> None:
    model.eval()
    inp = torch.rand(1, 3, 48, 48, device=device)
    coord = make_coord([96, 96]).unsqueeze(0).to(device)[:, :128, :].contiguous()
    with torch.no_grad():
        for scale in [2, 4, 8, 16, 30]:
            cell = torch.ones_like(coord)
            size = 48 * scale
            cell[:, :, 0] *= 2 / size
            cell[:, :, 1] *= 2 / size
            pred = model(inp, coord, cell)
            if not torch.isfinite(pred).all():
                raise AssertionError(f"non-finite prediction for x{scale}")
            print(
                f"scale_cell x{scale}: pred_mean={float(pred.mean()):.6f} "
                f"pred_std={float(pred.std()):.6f}"
            )


def check_response_not_constant(model, device: torch.device) -> None:
    model.eval()
    inp, coord, cell, _ = make_batch(device, q=128)
    with torch.no_grad():
        model.gen_feat(inp)
        q_omega = model._grid_fetch(model.omega_map, coord)
        q_coord = model._grid_fetch(model.feat_coord, coord)
        rel_coord = coord - q_coord
        rel_coord[:, :, 0] *= model.feat.shape[-2]
        rel_coord[:, :, 1] *= model.feat.shape[-1]

        cell_small = cell.clone()
        cell_small[:, :, 0] *= model.feat.shape[-2]
        cell_small[:, :, 1] *= model.feat.shape[-1]
        cell_large = cell_small * 8

        q_phi = model._grid_fetch(model.phase_map, coord) if model.phase_map is not None else None
        feats_small = model._basis_and_response(q_omega, q_phi, rel_coord, cell_small)
        feats_large = model._basis_and_response(q_omega, q_phi, rel_coord, cell_large)
        delta = float((feats_small - feats_large).abs().mean())
    print(f"response_cell_delta={delta:.6e}")
    if model.use_sinc_response and delta <= 1e-7:
        raise AssertionError("sinc response appears insensitive to cell size")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config",
        type=Path,
        default=ROOT / "configs" / "train-div2k" / "train-sc-inr-eq.yaml",
    )
    parser.add_argument("--device", default="cuda:0")
    args = parser.parse_args()

    if args.device.startswith("cuda") and not torch.cuda.is_available():
        device = torch.device("cpu")
    else:
        device = torch.device(args.device)

    model, config = load_model_from_config(args.config)
    if config["model"]["name"] != "sc_inr_eq":
        raise AssertionError("smoke config must use sc_inr_eq")

    check_omega_init(model)
    check_forward_backward(model, device)
    check_scale_cells(model, device)
    check_response_not_constant(model, device)
    print("sc_inr_eq smoke checks passed")


if __name__ == "__main__":
    main()
