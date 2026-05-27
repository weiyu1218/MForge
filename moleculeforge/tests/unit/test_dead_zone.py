from __future__ import annotations

import torch


def test_patent_dead_zone_uses_batched_lorentz_distance() -> None:
    from mf_humu.manifold.lorentz import LorentzManifold
    from mf_humu.operations.dead_zone import patent_dead_zone_potential

    manifold = LorentzManifold(curvature=1.0)
    origin = manifold.origin(2)
    tangent_near = torch.tensor([0.0, 0.2, 0.0])
    tangent_far = torch.tensor([0.0, 1.5, 0.0])
    query = torch.stack(
        [
            origin,
            manifold.expmap(origin, tangent_far),
        ]
    )
    dead_zone = manifold.expmap(origin, tangent_near).unsqueeze(0)

    potential = patent_dead_zone_potential(query, [dead_zone], radius=0.5, manifold=manifold)

    assert potential.shape == (2,)
    assert potential[0] > potential[1]


def test_patent_dead_zone_empty_zones_returns_zero() -> None:
    from mf_humu.operations.dead_zone import patent_dead_zone_potential

    z = torch.ones(3, 4)

    potential = patent_dead_zone_potential(z, [])

    assert torch.equal(potential, torch.zeros(3))
