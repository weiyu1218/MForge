"""Learnable-curvature Lorentz manifold operations."""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as functional
from torch import Tensor


class LearnableLorentzManifold(nn.Module):
    def __init__(self, curvature: float = 1.0, eps: float = 1e-8):
        super().__init__()
        if curvature <= 0:
            raise ValueError("curvature must be positive")
        self.eps = eps
        initial = torch.log(torch.expm1(torch.tensor(float(curvature - eps))).clamp_min(eps))
        self.raw_curvature = nn.Parameter(initial)

    @property
    def k(self) -> Tensor:
        return functional.softplus(self.raw_curvature) + self.eps

    @property
    def c(self) -> Tensor:
        return self.k

    @c.setter
    def c(self, value: float):
        if value <= 0:
            raise ValueError("curvature must be positive")
        with torch.no_grad():
            raw = torch.log(torch.expm1(torch.tensor(float(value - self.eps))).clamp_min(self.eps))
            self.raw_curvature.copy_(raw.to(self.raw_curvature.device))

    def origin(self, dim: int) -> Tensor:
        x = torch.zeros(
            dim + 1,
            dtype=self.raw_curvature.dtype,
            device=self.raw_curvature.device,
        )
        x[0] = torch.rsqrt(self.k)
        return x

    def project_tangent(self, base: Tensor, v: Tensor) -> Tensor:
        return self._project_to_tangent(base, v)

    def _project(self, x: Tensor) -> Tensor:
        x_s = x[..., 1:]
        norm_sq = (x_s**2).sum(dim=-1, keepdim=True)
        x_t = torch.sqrt(norm_sq + 1.0 / self.k + self.eps)
        return torch.cat([x_t, x_s], dim=-1)

    def _project_to_tangent(self, x: Tensor, v: Tensor) -> Tensor:
        inner_xv = self.inner(x, v)
        return v + self.k * inner_xv * x

    def inner(self, x: Tensor, y: Tensor, keepdim: bool = True) -> Tensor:
        return -x[..., 0:1] * y[..., 0:1] + (x[..., 1:] * y[..., 1:]).sum(
            dim=-1,
            keepdim=keepdim,
        )

    def distance(self, x: Tensor, y: Tensor) -> Tensor:
        arg = -self.k * self.inner(x, y)
        return torch.arccosh(torch.clamp(arg, min=1.0 + self.eps))

    def expmap(self, x: Tensor, v: Tensor) -> Tensor:
        v_tangent = self._project_to_tangent(x, v)
        v_norm_sq = torch.clamp(self.inner(v_tangent, v_tangent), min=0.0) * self.k
        v_norm = torch.sqrt(v_norm_sq + self.eps)
        small = v_norm < 1e-6
        sinh_over_norm = torch.where(
            small,
            torch.ones_like(v_norm),
            torch.sinh(v_norm) / v_norm,
        )
        result = torch.cosh(v_norm) * x + sinh_over_norm * v_tangent
        return self._project(result)

    def logmap(self, x: Tensor, y: Tensor) -> Tensor:
        c_inner = self.k * self.inner(x, y)
        dist = torch.acosh(torch.clamp(-c_inner, min=1.0 + self.eps))
        u = y + c_inner * x
        u_norm_sq = torch.clamp(self.inner(u, u), min=self.eps)
        u_norm = torch.sqrt(u_norm_sq * self.k)
        return dist * u / u_norm
