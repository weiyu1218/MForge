"""Lorentz manifold operations on the hyperboloid model.

The Lorentz manifold ℍ^d (hyperboloid model) is defined as:
  ℍ^d = {x ∈ ℝ^{d+1} : <x, x>_L = -1, x_0 > 0}
where <x, y>_L = -x_0*y_0 + Σ_{i=1}^d x_i*y_i is the Lorentz inner product.

For a point x ∈ ℍ^d, the tangent space T_x ℍ^d consists of vectors v ∈ ℝ^{d+1}
satisfying <x, v>_L = 0.
"""
import torch
from torch import Tensor


class LorentzManifold:
    def __init__(self, curvature: float = 1.0, eps: float = 1e-8):
        self.k = curvature
        self.eps = eps

    @property
    def c(self) -> float:
        return self.k

    @c.setter
    def c(self, value: float):
        self.k = value

    def origin(self, dim: int) -> "torch.Tensor":
        x = torch.zeros(dim + 1, dtype=torch.float32)
        x[0] = 1.0 / (self.k ** 0.5)
        return x

    def project_tangent(self, base: "torch.Tensor", v: "torch.Tensor") -> "torch.Tensor":
        return self._project_to_tangent(base, v)

    def _project(self, x: Tensor) -> Tensor:
        """Project an arbitrary point onto the Lorentz hyperboloid <x,x>_L = -1/k.

        Sets x_0 = sqrt(1/k + ||x_s||^2) so that -x_0^2 + ||x_s||^2 = -1/k.
        """
        x_t = x[..., 0:1]
        x_s = x[..., 1:]
        norm_sq = (x_s**2).sum(dim=-1, keepdim=True)
        x_t = torch.sqrt(norm_sq + 1.0 / self.k + self.eps)
        return torch.cat([x_t, x_s], dim=-1)

    def _project_to_tangent(self, x: Tensor, v: Tensor) -> Tensor:
        """Project vector v onto the tangent space T_x ℍ^d.

        For the Lorentz manifold: proj_{T_x}(v) = v + k * <x, v>_L * x
        (since <x, x>_L = -1/k, the orthogonal projection onto T_x
        is v - <x,v>_L / <x,x>_L * x = v + k * <x,v>_L * x)
        """
        inner_xv = self.inner(x, v)
        return v + self.k * inner_xv * x

    def inner(self, x: Tensor, y: Tensor, keepdim: bool = True) -> Tensor:
        """Lorentzian inner product: <x, y>_L = -x_0*y_0 + Σ x_i*y_i."""
        return -x[..., 0:1] * y[..., 0:1] + (x[..., 1:] * y[..., 1:]).sum(
            dim=-1, keepdim=keepdim
        )

    def distance(self, x: Tensor, y: Tensor) -> Tensor:
        """Geodesic distance on ℍ^d: d(x,y) = arccosh(-<x,y>_L).

        Uses the identity cosh(d) = -<x,y>_L for unit curvature.
        """
        inner_val = self.inner(x, y)
        arg = -inner_val  # cosh(d) = -<x,y>_L
        arg = torch.clamp(arg, min=1.0 + self.eps)
        return torch.arccosh(arg)

    def expmap(self, x: Tensor, v: Tensor) -> Tensor:
        """Exponential map exp_x: T_x ℍ^d → ℍ^d.

        exp_x(v) = cosh(|v|) * x + sinh(|v|) * (v / |v|)
        where |v| = sqrt(<v, v>_L * c).

        First projects v to T_x ℍ^d to ensure the result lies on the manifold.
        Numerically stable: when |v| → 0, sinh(|v|)/|v| ≈ 1.
        """
        v_tangent = self._project_to_tangent(x, v)
        v_norm_sq = torch.clamp(self.inner(v_tangent, v_tangent), min=0.0) * self.k
        v_norm = torch.sqrt(v_norm_sq + self.eps)

        small = v_norm < 1e-6
        sinh_over_norm = torch.where(
            small,
            torch.ones_like(v_norm),
            torch.sinh(v_norm) / v_norm,
        )
        cosh_v = torch.cosh(v_norm)

        result = cosh_v * x + sinh_over_norm * v_tangent
        return self._project(result)

    def logmap(self, x: Tensor, y: Tensor) -> Tensor:
        """Logarithmic map log_x: ℍ^d → T_x ℍ^d.

        log_x(y) = arccosh(-c·<x,y>_L) · (y + c·<x,y>_L · x) / sqrt((c·<x,y>_L)^2 - 1)
        """
        c_inner = self.k * self.inner(x, y)
        c_inner_clamped = torch.clamp(-c_inner, min=1.0 + self.eps)
        dist = torch.acosh(c_inner_clamped)

        u = y + c_inner * x
        u_norm_sq = torch.clamp(self.inner(u, u), min=self.eps)
        u_norm = torch.sqrt(u_norm_sq * self.k)

        return dist * u / u_norm
