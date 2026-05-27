# mf-eval

All evaluation metrics live here. Used by:

- pipelines/ (e.g. `boltz2_eval`, `pareto_bo`)
- tests/benchmark/
- the Critic agent (rules in `agents/critic`)

## Layout

```
src/mf_eval/
├── molecule/   Validity / Uniqueness / Novelty, MOSES, GuacaMol, PMO
├── humu/       Tree-distortion, cliff separation, retrieval EF1%
├── pareto/     Hypervolume, spread, convergence
└── agent/      Task completion, audit completeness
```
