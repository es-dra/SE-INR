#!/usr/bin/env python3
"""
V0: Gradient flow check for SC-INR Phase 2.
Verifies that omega_conv receives gradients of normal magnitude.

Checks:
  1. Forward pass succeeds (no NaN, shapes correct)
  2. omega_conv.weight.grad is non-zero after backward
  3. Gradient magnitude is comparable to coef
  4. No NaN or Inf in gradients

Usage:
  python verify_v0_gradient.py
"""

import sys
sys.path.insert(0, '.')
import torch
import models

device = 'cuda:0'

# Build Phase 2 model
model_spec = {
    'name': 'sc_inr_phase2',
    'args': {
        'encoder_spec': {
            'name': 'edsr-baseline',
            'args': {'no_upsampling': True},
        },
        'imnet_spec': {
            'name': 'mlp',
            'args': {'out_dim': 3, 'hidden_list': [256, 256, 256]},
        },
        'hidden_dim': 256,
        'num_freqs': 128,
        'num_angles': 8,
        'freq_min': 0.1,
        'freq_max': 2.0,
    },
}
model = models.make(model_spec).to(device)
model.train()

# Count params
total = sum(p.numel() for p in model.parameters())
trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
print(f"Total params: {total:,}  Trainable: {trainable:,}")
omega_params = sum(p.numel() for p in model.omega_conv.parameters())
print(f"omega_conv params: {omega_params:,}")
print()

# Synthetic batch
B, C, H, W = 2, 3, 48, 48
Q = 100
inp = torch.randn(B, C, H, W, device=device)
coord = torch.rand(B, Q, 2, device=device) * 2 - 1
cell = torch.ones(B, Q, 2, device=device) * (2 / 4)  # scale=4 cell

# === Forward pass ===
print("Forward pass...")
pred = model(inp, coord, cell)
print(f"  pred shape: {pred.shape}  (expected: [{B}, {Q}, 3])")
print(f"  pred range: [{pred.min().item():.4f}, {pred.max().item():.4f}]")
print(f"  any NaN: {torch.isnan(pred).any().item()}")
print(f"  any Inf: {torch.isinf(pred).any().item()}")
print()

# Check omega_map stats
print("omega_map statistics:")
omega_map = model.omega_map
print(f"  shape: {omega_map.shape}  (expected: [{B}, 256, {H}, {W}])")
print(f"  mean: {omega_map.mean().item():.4f}")
print(f"  std: {omega_map.std().item():.4f}")
print(f"  min: {omega_map.min().item():.4f}")
print(f"  max: {omega_map.max().item():.4f}")
print()

# === Backward pass ===
print("Backward pass...")
loss = pred.mean()
loss.backward()

# Check omega_conv gradients
omega_w_grad = model.omega_conv.weight.grad
omega_b_grad = model.omega_conv.bias.grad
coef_w_grad = model.coef.weight.grad

print(f"  omega_conv.weight.grad:")
print(f"    mean abs: {omega_w_grad.abs().mean().item():.8f}")
print(f"    max  abs: {omega_w_grad.abs().max().item():.8f}")
print(f"    any NaN:  {torch.isnan(omega_w_grad).any().item()}")
print(f"    any Inf:  {torch.isinf(omega_w_grad).any().item()}")

print(f"  omega_conv.bias.grad:")
print(f"    mean abs: {omega_b_grad.abs().mean().item():.8f}")
print(f"    max  abs: {omega_b_grad.abs().max().item():.8f}")

print(f"  coef.weight.grad (reference):")
print(f"    mean abs: {coef_w_grad.abs().mean().item():.8f}")

# Compare gradient magnitudes
omega_grad_mag = omega_w_grad.abs().mean().item()
coef_grad_mag = coef_w_grad.abs().mean().item()
ratio = omega_grad_mag / coef_grad_mag if coef_grad_mag > 0 else float('inf')

print(f"\n  omega_conv / coef grad ratio: {ratio:.4f}")
print()

# === Init verification ===
print("Initialization verification:")
# Re-create to check init
model2 = models.make(model_spec).to(device)
with torch.no_grad():
    model2.gen_feat(torch.randn(1, 3, 48, 48, device=device))
    omega_init = model2.omega_map
    # Phase 1 reference frequencies for comparison

# Quick check: omega_map values should be in a reasonable range
omega_init_vals = omega_init.flatten().cpu()
print(f"  omega_map at init: mean={omega_init_vals.mean():.4f}, "
      f"std={omega_init_vals.std():.4f}, "
      f"min={omega_init_vals.min():.4f}, "
      f"max={omega_init_vals.max():.4f}")
print(f"  Expected range: [{model.freq_min}, {model.freq_max}]")

# === VERDICT ===
print()
print("=" * 50)
checks = [
    ("Forward no NaN", not torch.isnan(pred).any().item()),
    ("Forward no Inf", not torch.isinf(pred).any().item()),
    ("omega_conv.w has grad", omega_w_grad.abs().sum() > 0),
    ("omega_conv.b has grad", omega_b_grad.abs().sum() > 0),
    ("Grad no NaN", not torch.isnan(omega_w_grad).any().item()),
    ("Grad no Inf", not torch.isinf(omega_w_grad).any().item()),
    ("Grad ratio reasonable", 0.001 < ratio < 100),
    ("omega positive (softplus)", omega_map.min().item() > 0),
]
all_pass = True
for name, passed in checks:
    status = "PASS" if passed else "FAIL"
    if not passed:
        all_pass = False
    print(f"  [{status}] {name}")

print()
if all_pass:
    print("V0: ALL CHECKS PASSED - Gradient flow is healthy.")
else:
    print("V0: SOME CHECKS FAILED - Investigate before proceeding.")
