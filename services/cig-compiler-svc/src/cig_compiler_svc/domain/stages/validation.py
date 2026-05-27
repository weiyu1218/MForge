"""Stage 2b: CIG validation."""
from __future__ import annotations

from dataclasses import dataclass, field

from mf_core.types.cig import ChemicalIntentGraph


@dataclass
class ValidationReport:
    is_valid: bool = True
    errors: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)

    def __str__(self) -> str:
        parts = []
        if self.errors:
            parts.append("Errors: " + "; ".join(self.errors))
        if self.warnings:
            parts.append("Warnings: " + "; ".join(self.warnings))
        return "\n".join(parts) if parts else "OK"


def validate_cig(cig: ChemicalIntentGraph) -> ValidationReport:
    report = ValidationReport()
    if not cig.objective_nodes:
        report.is_valid = False
        report.errors.append("No objective nodes defined in CIG")
    total_weight = sum(o.weight for o in cig.objective_nodes)
    if cig.objective_nodes and abs(total_weight - 1.0) > 0.1:
        report.warnings.append(f"Objective weights sum to {total_weight:.2f}, expected ~1.0")
    return report
