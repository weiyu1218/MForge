from __future__ import annotations

import builtins
import importlib.util
import sys
from pathlib import Path
from types import SimpleNamespace

ROOT = Path(__file__).resolve().parents[2]
WRAPPER_PATH = ROOT / "tools" / "oracles" / "pyscf_quantum_oracle_wrapper.py"


spec = importlib.util.spec_from_file_location("pyscf_quantum_oracle_wrapper", WRAPPER_PATH)
assert spec is not None
assert spec.loader is not None
wrapper = importlib.util.module_from_spec(spec)
spec.loader.exec_module(wrapper)


def test_gpu4pyscf_rhf_uses_narrow_patch_import(monkeypatch):
    patch_loaded = False
    original_import = builtins.__import__

    def guarded_import(name, globals=None, locals=None, fromlist=(), level=0):
        if name == "gpu4pyscf" and not fromlist:
            raise AssertionError("GPU4PySCF wrapper must not use top-level import")
        if name == "pyscf":
            return SimpleNamespace(
                gto=SimpleNamespace(M=lambda **kwargs: SimpleNamespace(nelectron=2)),
                scf=SimpleNamespace(RHF=lambda mol: _FakeCpuRHF()),
            )
        if name == "gpu4pyscf.scf":
            return SimpleNamespace(hf=SimpleNamespace(RHF=lambda mol: _FakeRHF()))
        return original_import(name, globals, locals, fromlist, level)

    def load_patch():
        nonlocal patch_loaded
        patch_loaded = True

    monkeypatch.setattr(builtins, "__import__", guarded_import)
    monkeypatch.setattr(wrapper, "_load_gpu4pyscf_patch", load_patch)

    energy, converged = wrapper._run_gpu4pyscf_rhf(
        [("H", (0.0, 0.0, 0.0)), ("H", (0.0, 0.0, 0.74))],
        0,
        "sto-3g",
    )

    assert energy == -1.0
    assert converged is True
    assert patch_loaded is True


def test_gpu4pyscf_rhf_avoids_pyscf_to_gpu_conversion(monkeypatch):
    original_import = builtins.__import__

    def guarded_import(name, globals=None, locals=None, fromlist=(), level=0):
        if name == "pyscf":
            return SimpleNamespace(
                gto=SimpleNamespace(M=lambda **kwargs: SimpleNamespace(nelectron=2)),
                scf=SimpleNamespace(RHF=lambda mol: _FakeCpuRHF()),
            )
        if name == "gpu4pyscf.scf":
            return SimpleNamespace(hf=SimpleNamespace(RHF=lambda mol: _FakeRHF()))
        return original_import(name, globals, locals, fromlist, level)

    monkeypatch.setattr(builtins, "__import__", guarded_import)
    monkeypatch.setattr(wrapper, "_load_gpu4pyscf_patch", lambda: None)

    energy, converged = wrapper._run_gpu4pyscf_rhf(
        [("H", (0.0, 0.0, 0.0)), ("H", (0.0, 0.0, 0.74))],
        0,
        "sto-3g",
    )

    assert energy == -1.0
    assert converged is True


def test_gpu4pyscf_patch_loader_bypasses_parent_init(monkeypatch, tmp_path):
    package_dir = tmp_path / "gpu4pyscf"
    package_dir.mkdir()
    (package_dir / "__init__.py").write_text(
        "raise AssertionError('top-level gpu4pyscf import executed')\n"
    )
    (package_dir / "_patch_pyscf.py").write_text("PATCH_LOADED = True\n")
    monkeypatch.syspath_prepend(str(tmp_path))
    monkeypatch.delitem(sys.modules, "gpu4pyscf", raising=False)
    monkeypatch.delitem(sys.modules, "gpu4pyscf._patch_pyscf", raising=False)

    module = wrapper._load_gpu4pyscf_patch()

    assert module.PATCH_LOADED is True
    assert sys.modules["gpu4pyscf"].__path__ == [str(package_dir)]


class _FakeRHF:
    converged = True

    def kernel(self):
        return -1.0


class _FakeCpuRHF:
    def to_gpu(self):
        raise AssertionError("GPU4PySCF wrapper must construct GPU RHF directly")
