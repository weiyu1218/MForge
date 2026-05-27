from __future__ import annotations

import torch


def test_learnable_lorentz_curvature_is_trainable_and_positive() -> None:
    from mf_humu.manifold.learnable_lorentz import LearnableLorentzManifold

    manifold = LearnableLorentzManifold(curvature=1.0)
    origin = manifold.origin(3)
    tangent = torch.tensor([0.0, 0.2, -0.1, 0.3])

    point = manifold.expmap(origin, tangent)
    loss = manifold.distance(origin, point).sum() + manifold.k
    loss.backward()

    assert manifold.k.item() > 0.0
    assert "raw_curvature" in dict(manifold.named_parameters())
    assert manifold.raw_curvature.grad is not None


def test_humu_encoders_expose_learnable_curvature_parameters() -> None:
    from mf_encoders.humu_mol.encoder import HUMUMoleculeEncoder

    encoder = HUMUMoleculeEncoder(dim=8, curvature=1.0, learnable_curvature=True)

    names = [name for name, _ in encoder.named_parameters()]

    assert any(name.endswith("raw_curvature") for name in names)


def test_humu_pretrain_builds_learnable_curvature_encoders() -> None:
    from humu_pretrain.pipeline import _build_encoders

    encoders = _build_encoders(
        {"embed_dim": 9, "curvature": 1.0, "learnable_curvature": True, "encoders": {}},
        torch.device("cpu"),
    )

    parameter_names = [
        name
        for encoder in encoders.values()
        for name, _ in encoder.named_parameters()
    ]

    assert any(name.endswith("raw_curvature") for name in parameter_names)
