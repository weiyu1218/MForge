"""PMO (Practical Molecular Optimization) benchmark.

PMO evaluates multi-property optimization capability with 23 tasks
spanning: logP, QED, DRD2, JNK3, GSK3B, and multi-objective combinations.

Reference: https://github.com/wenhao-gao/mol_opt
"""

import pytest


@pytest.mark.benchmark
@pytest.mark.slow
class TestPMOBenchmark:
    """PMO 23-task multi-objective optimization benchmark."""

    def test_logp_optimization(self):
        """Maximize LogP (simple single-objective baseline)."""
        pytest.skip("Requires trained generators + PMO dataset")

    def test_qed_optimization(self):
        """Maximize QED (drug-likeness)."""
        pytest.skip("Requires trained generators + PMO dataset")

    def test_drd2_optimization(self):
        """Maximize DRD2 activity score."""
        pytest.skip("Requires trained generators + PMO dataset")

    def test_multi_objective_logp_qed(self):
        """Simultaneously maximize LogP and QED."""
        pytest.skip("Requires trained generators + PMO dataset")

    def test_multi_objective_jnk3_gsk3b(self):
        """Dual kinase selectivity optimization."""
        pytest.skip("Requires trained generators + PMO dataset")
