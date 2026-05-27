# mf-humu

Hyperbolic-manifold math used by every layer that touches HUMU embeddings.

## Layout

```
src/mf_humu/
├── manifold/        Lorentz model: distance, exp/log, parallel transport
├── encoders/        Tangent-space → manifold projection layers, hyperbolic attention
├── operations/      Intent cone, dead zone potentials, cliff detection, OOD
├── gp/              Hyperbolic Gaussian processes (SVGP, EHVI)
└── utils/           Numerical stability, Poincaré-disk visualisation
```

## Why both Lorentz and Poincaré?

We use the **Lorentz** model in production (closed-form exp/log, stable
gradients near the boundary). Poincaré conversions are kept only for
visualisation (`utils/visualization.py`).

ADR: see `docs/adr/0001-use-lorentz-not-poincare.md`.
