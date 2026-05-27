"""Sparse Variational Gaussian Process for Lorentz manifold.

Implements a sparse variational GP (SVGP) that uses Lorentz manifold kernels
and inducing points for scalable Bayesian optimization in high-dimensional
chemical latent space.

Reference:
    Hensman et al., "Gaussian Processes for Big Data", UAI 2013.
    Titsias, "Variational Learning of Inducing Variables in Sparse Gaussian
    Processes", AISTATS 2009.
"""
import torch
from torch import Tensor, nn
from mf_humu.gp.kernels import LorentzMaternKernel
from mf_humu.manifold.lorentz import LorentzManifold


class SparseVariationalGP(nn.Module):
    """Sparse Variational Gaussian Process with Lorentz manifold kernel.

    Uses inducing points on the Lorentz manifold for scalable inference.
    """

    def __init__(
        self,
        inducing_points: Tensor,
        n_outputs: int = 1,
        length_scale: float = 1.0,
        nu: float = 2.5,
        noise_variance: float = 1e-4,
        manifold: LorentzManifold | None = None,
    ):
        """Initialize SVGP.

        Args:
            inducing_points: Initial inducing points on Lorentz manifold,
                shape (m, d+1).
            n_outputs: Number of output dimensions (objectives).
            length_scale: Kernel length scale.
            nu: Matern smoothness parameter (0.5, 1.5, or 2.5).
            noise_variance: Observation noise variance (jitter).
            manifold: Lorentz manifold instance.
        """
        super().__init__()
        self.manifold = manifold or LorentzManifold()

        # Variational parameters
        self.inducing_points = nn.Parameter(inducing_points.clone())
        n_inducing = inducing_points.shape[0]

        # Variational distribution parameters q(u) = N(m, S)
        # Mean of inducing outputs
        self.q_mu = nn.Parameter(torch.zeros(n_inducing, n_outputs))
        # Lower triangular Cholesky factor of covariance
        self.q_sqrt = nn.Parameter(
            torch.eye(n_inducing * n_outputs).reshape(
                n_outputs, n_inducing, n_inducing
            )
        )

        self.kernel = LorentzMaternKernel(
            length_scale=length_scale, nu=nu, manifold=self.manifold
        )
        self.noise_variance = noise_variance
        self.n_outputs = n_outputs

    def forward(self, x: Tensor) -> tuple[Tensor, Tensor]:
        """Compute predictive posterior at test points.

        Args:
            x: Test points on Lorentz manifold, shape (n, d+1).

        Returns:
            mean: Predictive mean, shape (n, n_outputs).
            variance: Predictive variance, shape (n, n_outputs).
        """
        # Kernel matrices
        K_uu = self.kernel(self.inducing_points, self.inducing_points)
        K_uf = self.kernel(self.inducing_points, x)  # (m, n)
        K_ff_diag = self._kernel_diag(x)

        # Add jitter to K_uu for numerical stability
        K_uu = K_uu + torch.eye(
            self.inducing_points.shape[0], device=K_uu.device
        ) * self.noise_variance

        # Cholesky of K_uu
        L_uu = torch.linalg.cholesky(K_uu)

        # Compute K_uu^{-1} K_uf via triangular solve
        L_inv_K_uf = torch.linalg.solve_triangular(
            L_uu, K_uf, upper=False
        )

        # Predictive mean
        mean = K_uf.transpose(-2, -1) @ torch.linalg.solve_triangular(
            L_uu, L_inv_K_uf, upper=False
        ).transpose(-2, -1) @ self.q_mu

        # Predictive variance (per output)
        # var = k_ff - k_uf^T K_uu^{-1} k_uf + k_uf^T K_uu^{-1} S K_uu^{-1} k_uf
        var = K_ff_diag.unsqueeze(-1).expand(-1, self.n_outputs)
        quad_term = (L_inv_K_uf**2).sum(dim=0)  # (n,)

        for d in range(self.n_outputs):
            var[:, d] = var[:, d] - quad_term
            # Add variational correction: diag(k_uf^T L^{-T} S L^{-1} k_uf)
            S_d = self.q_sqrt[d] @ self.q_sqrt[d].transpose(-2, -1)
            L_S = torch.linalg.cholesky(
                S_d
                + torch.eye(S_d.shape[0], device=S_d.device) * self.noise_variance
            )
            if L_S.device != L_inv_K_uf.device:
                L_S = L_S.to(L_inv_K_uf.device)
            L_S_L_inv_K = L_S @ L_inv_K_uf
            var[:, d] = var[:, d] + (L_S_L_inv_K**2).sum(dim=0)

        return mean, torch.clamp(var, min=self.noise_variance)

    def _kernel_diag(self, x: Tensor) -> Tensor:
        """Compute diagonal of kernel matrix K_ff.

        For stationary kernels, k(x, x) = 1.0 (with unit variance).

        Args:
            x: Input points, shape (n, d+1).

        Returns:
            Diagonal values, shape (n,).
        """
        return torch.ones(x.shape[0], device=x.device)

    def elbo(
        self, x: Tensor, y: Tensor, n_data: int | None = None
    ) -> Tensor:
        """Evidence Lower Bound for variational inference.

        Args:
            x: Training inputs, shape (n, d+1).
            y: Training targets, shape (n, n_outputs).
            n_data: Total number of data points (for minibatch scaling).

        Returns:
            ELBO scalar value.
        """
        if n_data is None:
            n_data = x.shape[0]

        mean, var = self.forward(x)

        # Data likelihood term (Gaussian)
        quad_term = ((y - mean) ** 2) / var
        log_det_term = torch.log(var)
        likelihood = -0.5 * (quad_term + log_det_term + torch.log(
            torch.tensor(2.0 * torch.pi, device=x.device)
        ))

        # Scale by data ratio for minibatching
        scale = n_data / x.shape[0]
        data_term = likelihood.sum() * scale

        # KL divergence from variational posterior to prior
        kl = self._kl_divergence()

        return data_term - kl

    def _kl_divergence(self) -> Tensor:
        """Compute KL(q(u) || p(u)).

        KL between variational Gaussian and prior Gaussian on inducing outputs.

        Returns:
            KL divergence scalar.
        """
        m = self.inducing_points.shape[0]
        device = self.q_mu.device

        K_uu = self.kernel(self.inducing_points, self.inducing_points)
        K_uu = K_uu + torch.eye(m, device=device) * self.noise_variance
        L_uu = torch.linalg.cholesky(K_uu)

        kl = torch.tensor(0.0, device=device)

        for d in range(self.n_outputs):
            mu_d = self.q_mu[:, d : d + 1]
            S_d = self.q_sqrt[d] @ self.q_sqrt[d].transpose(-2, -1)

            # Trace term: Tr(K_uu^{-1} S)
            L_inv_S = torch.linalg.solve_triangular(
                L_uu, S_d, upper=False
            )
            L_inv_S_L_inv_T = torch.linalg.solve_triangular(
                L_uu, L_inv_S.transpose(-2, -1), upper=False
            )
            trace_term = torch.trace(L_inv_S_L_inv_T)

            # Quadratic term: mu^T K_uu^{-1} mu
            L_inv_mu = torch.linalg.solve_triangular(
                L_uu, mu_d, upper=False
            )
            quad_term = (L_inv_mu**2).sum()

            # Log determinant: log(|K_uu|) - log(|S|)
            log_det_K = 2.0 * torch.log(torch.diag(L_uu)).sum()
            log_det_S = torch.log(
                torch.clamp(
                    torch.diag(self.q_sqrt[d]), min=1e-8
                )
            ).sum() * 2.0

            kl = kl + 0.5 * (
                trace_term + quad_term - m + log_det_K - log_det_S
            )

        return kl
