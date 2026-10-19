"""E1 geometry and matrix-free linear GP; no experiment runs on import.

Adapted from claude-20261004-e1/e1gp.py. The new task uses matched A,
independent calibration, un-clamped outputs, and explicit numerical records.
"""
from __future__ import annotations
import dataclasses
import math
import time
import zlib
import numpy as np
import torch
import torch.nn.functional as F

ARMS = ('gauss_3', 'gauss_10', 'gauss_30', 'gauss_60', 'gauss_128',
        'power_2', 'power_2.5', 'power_3')


@dataclasses.dataclass(frozen=True)
class Geometry:
    hr: int = 256
    scale: int = 4
    frames: int = 16
    subsamples: int = 8
    field: int = 512
    shave: int = 8

    @property
    def lr(self):
        return self.hr // self.scale


def frame_params(name, g):
    rng = np.random.default_rng(zlib.crc32(name.encode()))
    d = rng.uniform(-1, 1, (g.frames, 2)) * g.scale
    a = rng.uniform(-1, 1, g.frames) * math.pi / 180
    d[0] = 0
    a[0] = 0
    return d, a


def sampling_points(d, a, g, device='cpu', dtype=torch.float32):
    ii, jj = torch.meshgrid(torch.arange(g.lr, dtype=dtype), torch.arange(g.lr, dtype=dtype), indexing='ij')
    c = torch.stack([(jj + .5) * g.scale, (ii + .5) * g.scale], -1).view(-1, 1, 2)
    off1 = (torch.arange(g.subsamples, dtype=dtype) + .5) / g.subsamples * g.scale - g.scale / 2
    oy, ox = torch.meshgrid(off1, off1, indexing='ij')
    off = torch.stack([ox, oy], -1).view(1, -1, 2)
    p = (c + off).view(-1, 2) - g.hr / 2
    points = []
    for displacement, angle in zip(d, a):
        rot = torch.tensor([[math.cos(angle), -math.sin(angle)],
                            [math.sin(angle), math.cos(angle)]], dtype=dtype)
        points.append(p @ rot.T + g.hr / 2 + torch.tensor(displacement, dtype=dtype))
    return torch.stack(points).to(device=device, dtype=dtype) / g.hr


class Operator:
    def __init__(self, g, points):
        self.g, self.points = g, points
        self.a_calls = self.at_calls = 0
        self.sample_evals = 0

    def forward(self, x, indices=None):
        p = self.points if indices is None else self.points[indices]
        self.a_calls += 1
        self.sample_evals += p.numel() // 2
        grid = (p.reshape(1, 1, -1, 2) * 2 - 1)
        val = F.grid_sample(x[None, :, :self.g.hr, :self.g.hr], grid,
                            mode='bicubic', padding_mode='border', align_corners=False)[0, :, 0].T
        return val.reshape(len(p), self.g.lr ** 2, self.g.subsamples ** 2, x.shape[0]).mean(2)

    def adjoint(self, y, indices=None):
        self.at_calls += 1
        z = torch.zeros(y.shape[-1], self.g.field, self.g.field,
                        device=y.device, dtype=y.dtype, requires_grad=True)
        with torch.enable_grad():
            result, = torch.autograd.grad(self.forward(z, indices), z, grad_outputs=y)
        return result.detach()

    def counts(self):
        return dict(a_calls=self.a_calls, at_calls=self.at_calls,
                    interpolated_points=self.sample_evals)


def covariance_half(x, spectrum):
    return torch.fft.ifft2(torch.fft.fft2(x) * spectrum.sqrt()).real


def spectrum(arm, variance, g, device, dtype):
    f = torch.fft.fftfreq(g.field, device=device, dtype=dtype)
    if arm.startswith('gauss_'):
        # Original E1: cycles per HR image, L = 2 HR.
        nu = f * g.hr
        k2 = nu[:, None] ** 2 + nu[None] ** 2
        p = torch.exp(-k2 / (2 * float(arm.split('_')[1]) ** 2))
    else:
        k = (f[:, None] ** 2 + f[None] ** 2).sqrt()
        p = (k + 1. / g.field) ** (-float(arm.split('_')[1]))
    p[0, 0] = 0
    return p * variance * g.field ** 2 / p.sum() + 1e-10


class NumericalFailure(RuntimeError):
    pass


def fft_preconditioner_diagonal(pspec, g, frames, nv):
    """Fixed SPD Fourier approximation to I+C^1/2 A' A C^1/2/nv.

    Uniform sample density times observed area and squared box response.
    Fixed quarter-power damping avoids over-scaling directions made weakly
    observed by the spatial crop; it does not change the exact posterior.
    Cropping, subpixel motions and bicubic sampling are NOT replaced in M:
    this approximation is used only for the PCG search direction.
    """
    f = torch.fft.fftfreq(g.field, device=pspec.device, dtype=pspec.dtype)
    box = torch.sinc(g.scale*f).square()
    density = frames / g.scale**2 * (g.hr / g.field)**2
    diagonal = 1 + pspec * density * box[:,None] * box[None,:] / nv
    if not torch.isfinite(diagonal).all() or float(diagonal.min()) < 1:
        raise NumericalFailure('Invalid FFT preconditioner; must be finite and >=1')
    return diagonal.pow(.25)


def gp_mean(y, op, pspec, physical_noise, indices=None, max_iters=4000,
            tol=1e-5, budget_check=lambda: None, preconditioner='fft'):
    """Linear map for fixed spectrum. Physical noise and floor are distinct.

    Includes E1's sample-mean intercept. Never clamp. Nonconvergence is an
    explicit failure rather than a usable partially solved candidate.
    """
    if y.dtype != torch.float64 or pspec.dtype != torch.float64 or op.points.dtype != torch.float64:
        raise NumericalFailure('v4 requires float64 y, spectrum and operator geometry')
    nv = max(float(physical_noise) ** 2, 1e-5)
    mu = y.mean((0, 1))
    b = y - mu
    sqrt_spectrum = pspec.sqrt()
    def half(x):
        return torch.fft.ifft2(torch.fft.fft2(x)*sqrt_spectrum).real
    def matrix(x):
        return half(op.adjoint(op.forward(half(x), indices), indices)) / nv + x
    rhs = half(op.adjoint(b, indices)) / nv
    if preconditioner not in ('fft', 'none'):
        raise ValueError('Unknown preconditioner')
    diag = fft_preconditioner_diagonal(pspec, op.g,
        op.g.frames if indices is None else len(indices), nv) if preconditioner=='fft' else None
    preconditioner_calls = 0
    def inv_precondition(x):
        nonlocal preconditioner_calls
        preconditioner_calls += 1
        return x if diag is None else torch.fft.ifft2(torch.fft.fft2(x)/diag).real
    pcg_details = dict(kind=preconditioner, spectral_power=.25 if diag is not None else 0.,
        diagonal_min=float(diag.min()) if diag is not None else 1.,
        diagonal_max=float(diag.max()) if diag is not None else 1.)
    x = torch.zeros_like(rhs)
    residual = rhs.clone()
    z = inv_precondition(residual)
    direction = z.clone()
    rz = (residual*z).sum()
    rhs_norm = rhs.norm()
    history = []
    true_checks = 0
    residual_restarts = 0
    if not torch.isfinite(rhs_norm):
        raise NumericalFailure('nonfinite right hand side')
    if float(rhs_norm) == 0:
        return x[:, :op.g.hr, :op.g.hr] + mu[:, None, None], dict(
            iterations=0, relative_residual=0., true_relative_residual=0.,
            recursive_relative_residual=0., true_residual_checks=0,
            residual_restarts=0, converged=True, solver_variance=nv, dtype='float64',
            preconditioner=pcg_details, preconditioner_calls=preconditioner_calls)
    for iteration in range(1, max_iters + 1):
        budget_check()
        md = matrix(direction)
        den = (direction * md).sum()
        if not torch.isfinite(den) or float(den) <= 0:
            raise NumericalFailure('CG non-positive/nonfinite curvature')
        if not torch.isfinite(rz) or float(rz) <= 0:
            raise NumericalFailure('PCG non-positive/nonfinite preconditioned residual')
        alpha = rz / den
        x = x + alpha * direction
        residual = residual - alpha * md
        new_rr = (residual * residual).sum()
        rel = float(new_rr.sqrt() / rhs_norm)
        if not math.isfinite(rel):
            raise NumericalFailure('CG nonfinite residual')
        if iteration % 25 == 0:
            history.append([iteration, rel])
        if rel < tol or iteration == max_iters:
            # Recursive CG residual can drift; only the freshly evaluated
            # rhs-M(x) may certify convergence. A premature recursive trigger
            # replaces the residual and restarts the search direction.
            budget_check()
            actual_residual = rhs - matrix(x)
            actual_relative = float(actual_residual.norm() / rhs_norm)
            true_checks += 1
            if not math.isfinite(actual_relative):
                raise NumericalFailure('nonfinite true residual')
            if actual_relative < tol:
                field = half(x)[:, :op.g.hr, :op.g.hr] + mu[:, None, None]
                if not torch.isfinite(field).all():
                    raise NumericalFailure('nonfinite reconstruction')
                return field, dict(iterations=iteration, relative_residual=actual_relative,
                    true_relative_residual=actual_relative, recursive_relative_residual=rel,
                    true_residual_checks=true_checks, residual_restarts=residual_restarts,
                    converged=True, solver_variance=nv, checkpoints=history, dtype='float64',
                    preconditioner=pcg_details, preconditioner_calls=preconditioner_calls)
            if iteration < max_iters:
                residual = actual_residual
                z = inv_precondition(actual_residual)
                direction = z.clone()
                rz = (actual_residual*z).sum()
                residual_restarts += 1
                continue
        z = inv_precondition(residual)
        new_rz = (residual*z).sum()
        direction = z + new_rz / rz * direction
        rz = new_rz
    raise NumericalFailure(f'CG did not converge: {max_iters} iterations, '
                           f'recursive residual {rel:.8g}, true residual {actual_relative:.8g}')


def cropped_mse(a, b, g):
    s = g.shave
    return float(((a - b)[:, s:g.hr-s, s:g.hr-s] ** 2).mean())


def calibrated_scale(y, unit_observations, noise):
    """Only y and calibration probes. Each color has its own intercept.

    Reference spectral shape is independent; pooled scalar contrast is plug-in.
    Four common reference probes estimate tr(J A C0 A' J)/m.
    """
    n = y.shape[0] * y.shape[1]
    observed = float(((y - y.mean((0, 1))) ** 2).mean())
    expected = sum(float(((p - p.mean((0, 1))) ** 2).mean())
                   for p in unit_observations) / len(unit_observations)
    if expected <= 0 or not math.isfinite(expected):
        raise NumericalFailure('invalid calibration observation energy')
    scale = max((observed - noise ** 2 * (1 - 1. / n)) / expected, 1e-8)
    return scale, dict(observed_variance=observed, calibration_observed_variance=expected,
                       noise_correction=noise ** 2 * (1 - 1. / n), floor=1e-8)


def random_field(unit_pspec, seed, device, dtype):
    generator = torch.Generator(device=device).manual_seed(seed)
    z = torch.randn((3,) + tuple(unit_pspec.shape), generator=generator, device=device, dtype=dtype)
    return covariance_half(z, unit_pspec)


def noise_like(y, seed):
    generator = torch.Generator(device=y.device).manual_seed(seed)
    return torch.randn(y.shape, generator=generator, device=y.device, dtype=y.dtype)


def sync(device):
    if str(device).startswith('cuda'):
        torch.cuda.synchronize(device)


class Measurement:
    def __init__(self, op, device):
        self.op, self.device = op, device
    def __enter__(self):
        sync(self.device)
        self.counts = self.op.counts()
        self.start = time.monotonic()
        if str(self.device).startswith('cuda'):
            torch.cuda.reset_peak_memory_stats(self.device)
        return self
    def __exit__(self, *_):
        sync(self.device)
        self.record = dict(seconds=time.monotonic()-self.start,
                           **{k: v-self.counts[k] for k,v in self.op.counts().items()},
                           peak_gpu_bytes=torch.cuda.max_memory_allocated(self.device)
                           if str(self.device).startswith('cuda') else 0)
