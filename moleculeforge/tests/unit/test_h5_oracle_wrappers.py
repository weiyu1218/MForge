from __future__ import annotations

import json
import os
import runpy
import subprocess
import sys
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
ORACLE_DIR = ROOT / "tools" / "oracles"
FAST_BOLTZ_CLI = ORACLE_DIR / "boltz2_fast_cli.py"


def _run_wrapper(
    script_name: str,
    payload: dict,
    env: dict[str, str],
) -> subprocess.CompletedProcess[str]:
    run_env = os.environ.copy()
    run_env.update(env)
    return subprocess.run(  # noqa: S603
        [sys.executable, str(ORACLE_DIR / script_name)],
        input=json.dumps(payload, sort_keys=True),
        capture_output=True,
        check=False,
        cwd=ROOT,
        env=run_env,
        text=True,
    )


def _json_stdout(completed: subprocess.CompletedProcess[str]) -> dict:
    assert completed.returncode == 0, completed.stderr
    return json.loads(completed.stdout)


def _write_executable(path: Path, source: str) -> Path:
    path.write_text(source, encoding="utf-8")
    path.chmod(0o755)
    return path


def _write_openfe_transformation(path: Path, *, protocol_repeats: int = 1) -> Path:
    path.write_text(
        json.dumps(
            {
                "protocol": {
                    "settings": {
                        "protocol_repeats": protocol_repeats,
                    }
                }
            }
        ),
        encoding="utf-8",
    )
    return path


def test_dock_wrapper_runs_gnina_score_only(tmp_path: Path) -> None:
    receptor = tmp_path / "protein.pdb"
    receptor.write_text(
        "ATOM      1  N   ALA A   1      11.104  13.207   2.345  1.00 20.00           N\n",
        encoding="utf-8",
    )
    gnina = _write_executable(
        tmp_path / "gnina_fake.py",
        "\n".join(
            [
                "#!/usr/bin/env python3",
                "import pathlib, sys",
                "args = sys.argv",
                "assert '--score_only' in args",
                "assert '-r' in args",
                "assert '-l' in args",
                "assert pathlib.Path(args[args.index('-r') + 1]).is_file()",
                "assert pathlib.Path(args[args.index('-l') + 1]).is_file()",
                "print('CNNaffinity: -8.5')",
                "print('CNNscore: 0.71')",
            ]
        )
        + "\n",
    )

    completed = _run_wrapper(
        "dock_oracle_wrapper.py",
        {"engine": "gnina", "smiles": "CCO", "protein_pdb": str(receptor)},
        {"GNINA_BINARY": str(gnina)},
    )

    payload = _json_stdout(completed)
    assert payload["engine"] == "gnina"
    assert payload["scores"]["docking_score"] == pytest.approx(-8.5)
    assert payload["score"] == pytest.approx(-8.5)
    assert payload["elapsed_ms"] >= 0


def test_boltz2_wrapper_uses_boltz_cli_artifacts(tmp_path: Path) -> None:
    model_path = tmp_path / "boltz-2"
    model_path.mkdir()
    (model_path / "boltz2_conf.ckpt").write_bytes(b"conf")
    (model_path / "boltz2_aff.ckpt").write_bytes(b"aff")
    template_dir = tmp_path / "templates"
    template_dir.mkdir()
    (template_dir / "6OIM.yaml").write_text(
        """
version: 1
sequences:
  - protein:
      id: A
      sequence: MTEYKLVVVGAGGVGKSALTIQLIQNHFVDEYDPTIEDSYRKQV
      msa: empty
  - ligand:
      id: L
      smiles: "__LIGAND_SMILES__"
properties:
  - affinity:
      binder: L
""".strip(),
        encoding="utf-8",
    )
    boltz = _write_executable(
        tmp_path / "boltz_fake.py",
        "\n".join(
            [
                "#!/usr/bin/env python3",
                "import json, pathlib, sys",
                "args = sys.argv",
                "assert args[1] == 'predict'",
                "assert args[args.index('--num_workers') + 1] == '0'",
                "input_path = pathlib.Path(args[2])",
                "assert 'CCO' in input_path.read_text()",
                "out_dir = pathlib.Path(args[args.index('--out_dir') + 1])",
                "prediction_dir = out_dir / 'predictions' / input_path.stem",
                "prediction_dir.mkdir(parents=True)",
                "output = prediction_dir / f'affinity_{input_path.stem}.json'",
                "output.write_text(json.dumps({",
                "    'affinity_pred_value': -3.0,",
                "    'affinity_pred_value1': -3.1,",
                "    'affinity_pred_value2': -2.9,",
                "}), encoding='utf-8')",
            ]
        )
        + "\n",
    )

    completed = _run_wrapper(
        "boltz2_oracle_wrapper.py",
        {"protein_pdb_id": "6OIM", "ligand_smiles": ["CCO"], "ensemble_size": 2},
        {
            "BOLTZ_BINARY": str(boltz),
            "BOLTZ_MODEL_PATH": str(model_path),
            "BOLTZ_INPUT_TEMPLATE_DIR": str(template_dir),
            "BOLTZ_WORK_DIR": str(tmp_path / "work"),
            "BOLTZ_ACCELERATOR": "cpu",
            "BOLTZ_NUM_WORKERS": "0",
        },
    )

    payload = _json_stdout(completed)
    row = payload["affinities"][0]
    assert row["protein_pdb_id"] == "6OIM"
    assert row["ligand_smiles"] == "CCO"
    assert row["delta_g_kcal_mol"] == pytest.approx(-12.276)
    assert row["uncertainty"] == pytest.approx(0.1364)
    assert row["ki_nm"] == pytest.approx(1.0)
    assert row["ensemble_size"] == 2


def test_boltz2_fast_cli_entrypoint_exists_and_is_executable() -> None:
    assert FAST_BOLTZ_CLI.is_file()
    assert os.access(FAST_BOLTZ_CLI, os.X_OK)


def test_fep_wrapper_passes_service_contract_to_openfe_runner(tmp_path: Path) -> None:
    openfe_runner = _write_executable(
        tmp_path / "openfe_fake.py",
        "\n".join(
            [
                "#!/usr/bin/env python3",
                "import json, sys",
                "payload = json.load(sys.stdin)",
                "assert payload['project_id'] == 'project-1'",
                "assert payload['protein_pdb_id'] == '7abc'",
                "assert payload['reference_ligand_smiles'] == 'CCO'",
                "assert payload['test_ligand_smiles'] == ['CCN']",
                "assert payload['method'] == 'openfe'",
                "assert payload['n_repeats'] == 2",
                "print(json.dumps({",
                "    **payload,",
                "    'results': [{",
                "        'ligand_a_smiles': 'CCO',",
                "        'ligand_b_smiles': 'CCN',",
                "        'ddg_kcal_mol': -1.2,",
                "        'ddg_uncertainty': 0.3,",
                "        'n_repeats': 2,",
                "        'method': 'openfe',",
                "        'per_repeat_ddg': {'repeat_1': -1.1, 'repeat_2': -1.3},",
                "        'converged': True,",
                "    }],",
                "    'total_elapsed_ms': 33,",
                "}))",
            ]
        )
        + "\n",
    )

    completed = _run_wrapper(
        "fep_oracle_wrapper.py",
        {
            "project_id": "project-1",
            "request_id": "request-1",
            "batch_id": "batch-1",
            "protein_pdb_id": "7abc",
            "reference_ligand_smiles": "CCO",
            "test_ligand_smiles": ["CCN"],
            "method": "openfe",
            "n_repeats": 2,
        },
        {"OPENFE_RUNNER_PATH": str(openfe_runner)},
    )

    payload = _json_stdout(completed)
    assert payload["request_id"] == "request-1"
    assert payload["batch_id"] == "batch-1"
    assert payload["total_elapsed_ms"] == 33
    assert payload["results"][0]["ddg_kcal_mol"] == pytest.approx(-1.2)
    assert payload["results"][0]["converged"] is True


def test_openfe_json_runner_replays_configured_result(tmp_path: Path) -> None:
    replay = tmp_path / "openfe-result.json"
    replay.write_text(
        json.dumps(
            {
                "project_id": "project-1",
                "protein_pdb_id": "7abc",
                "reference_ligand_smiles": "CCO",
                "test_ligand_smiles": ["CCN"],
                "method": "openfe",
                "n_repeats": 1,
                "results": [
                    {
                        "ligand_a_smiles": "CCO",
                        "ligand_b_smiles": "CCN",
                        "ddg_kcal_mol": -1.2,
                        "ddg_uncertainty": 0.3,
                        "n_repeats": 1,
                        "method": "openfe",
                        "per_repeat_ddg": {"repeat_1": -1.2},
                        "converged": True,
                    }
                ],
                "total_elapsed_ms": 12,
            }
        ),
        encoding="utf-8",
    )

    completed = _run_wrapper(
        "openfe_json_runner.py",
        {
            "project_id": "project-1",
            "request_id": "request-1",
            "batch_id": "batch-1",
            "protein_pdb_id": "7abc",
            "reference_ligand_smiles": "CCO",
            "test_ligand_smiles": ["CCN"],
            "method": "openfe",
            "n_repeats": 1,
        },
        {"OPENFE_RESULT_REPLAY_PATH": str(replay)},
    )

    payload = _json_stdout(completed)
    assert payload["request_id"] == "request-1"
    assert payload["batch_id"] == "batch-1"
    assert payload["total_elapsed_ms"] == 12
    assert payload["results"][0]["ligand_a_smiles"] == "CCO"
    assert payload["results"][0]["ligand_b_smiles"] == "CCN"
    assert payload["results"][0]["ddg_kcal_mol"] == pytest.approx(-1.2)


def test_openfe_registry_builder_writes_transformation_and_result_registries(
    tmp_path: Path,
) -> None:
    from rdkit import Chem

    ligands_sdf = tmp_path / "ligands.sdf"
    writer = Chem.SDWriter(str(ligands_sdf))
    for name, smiles in (("lig_a", "CCO"), ("lig_b", "CCN")):
        mol = Chem.MolFromSmiles(smiles)
        assert mol is not None
        mol.SetProp("_Name", name)
        writer.write(mol)
    writer.close()

    transformations_dir = tmp_path / "transformations"
    transformations_dir.mkdir()
    complex_transformation = transformations_dir / "rbfe_lig_a_complex_lig_b_complex.json"
    solvent_transformation = transformations_dir / "rbfe_lig_a_solvent_lig_b_solvent.json"
    _write_openfe_transformation(complex_transformation)
    _write_openfe_transformation(solvent_transformation)
    ddg_tsv = tmp_path / "ddg.tsv"
    ddg_tsv.write_text(
        "\n".join(
            [
                (
                    "ligand_i\tligand_j\tDDG(i->j) (kcal/mol)"
                    "\tuncertainty (kcal/mol)\tn_repeats"
                    "\tper_repeat_ddg\tconverged"
                ),
                'lig_a\tlig_b\t1.25\t0.4\t2\t{"repeat_1":1.0,"repeat_2":1.5}\ttrue',
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    transformation_registry = tmp_path / "transformation-registry.json"
    result_registry = tmp_path / "result-registry.json"

    completed = subprocess.run(  # noqa: S603
        [
            sys.executable,
            str(ORACLE_DIR / "openfe_registry_builder.py"),
            "--protein-id",
            "tyk2",
            "--ligands-sdf",
            str(ligands_sdf),
            "--transformations-dir",
            str(transformations_dir),
            "--transformation-registry-output",
            str(transformation_registry),
            "--ddg-tsv",
            str(ddg_tsv),
            "--result-registry-output",
            str(result_registry),
        ],
        capture_output=True,
        check=False,
        cwd=ROOT,
        text=True,
    )

    assert completed.returncode == 0, completed.stderr
    transformation_payload = json.loads(transformation_registry.read_text(encoding="utf-8"))
    result_payload = json.loads(result_registry.read_text(encoding="utf-8"))
    assert transformation_payload["tyk2"]["CCO>>CCN"] == {
        "complex": "transformations/rbfe_lig_a_complex_lig_b_complex.json",
        "solvent": "transformations/rbfe_lig_a_solvent_lig_b_solvent.json",
        "ligand_a_name": "lig_a",
        "ligand_b_name": "lig_b",
        "ligand_a_smiles": "CCO",
        "ligand_b_smiles": "CCN",
    }
    assert result_payload["tyk2"]["CCO>>CCN"]["ddg_kcal_mol"] == pytest.approx(1.25)
    assert result_payload["tyk2"]["CCO>>CCN"]["ddg_uncertainty"] == pytest.approx(0.4)
    assert result_payload["tyk2"]["CCO>>CCN"]["n_repeats"] == 2
    assert result_payload["tyk2"]["CCO>>CCN"]["per_repeat_ddg"] == {
        "repeat_1": pytest.approx(1.0),
        "repeat_2": pytest.approx(1.5),
    }


def test_openfe_registry_builder_rejects_gather_tsv_without_repeat_evidence(
    tmp_path: Path,
) -> None:
    from rdkit import Chem

    ligands_sdf = tmp_path / "ligands.sdf"
    writer = Chem.SDWriter(str(ligands_sdf))
    for name, smiles in (("lig_a", "CCO"), ("lig_b", "CCN")):
        mol = Chem.MolFromSmiles(smiles)
        assert mol is not None
        mol.SetProp("_Name", name)
        writer.write(mol)
    writer.close()
    ddg_tsv = tmp_path / "gather-ddg.tsv"
    ddg_tsv.write_text(
        "ligand_i\tligand_j\tDDG(i->j) (kcal/mol)"
        "\tuncertainty (kcal/mol)\tconverged\n"
        "lig_a\tlig_b\t1.25\t0.4\ttrue\n",
        encoding="utf-8",
    )

    completed = subprocess.run(  # noqa: S603
        [
            sys.executable,
            str(ORACLE_DIR / "openfe_registry_builder.py"),
            "--protein-id",
            "tyk2",
            "--ligands-sdf",
            str(ligands_sdf),
            "--ddg-tsv",
            str(ddg_tsv),
            "--result-registry-output",
            str(tmp_path / "result-registry.json"),
        ],
        capture_output=True,
        check=False,
        cwd=ROOT,
        text=True,
    )

    assert completed.returncode == 1
    assert "n_repeats" in completed.stderr


def test_openfe_registry_builder_writes_experimental_binding_registry(
    tmp_path: Path,
) -> None:
    from rdkit import Chem

    ligands_sdf = tmp_path / "ligands.sdf"
    writer = Chem.SDWriter(str(ligands_sdf))
    for name, smiles in (("lig_a", "CCO"), ("lig_b", "CCN")):
        mol = Chem.MolFromSmiles(smiles)
        assert mol is not None
        mol.SetProp("_Name", name)
        writer.write(mol)
    writer.close()
    experimental_json = tmp_path / "experimental_binding_data.json"
    experimental_json.write_text(
        json.dumps(
            {
                "lig_a": {
                    "dg": {
                        "magnitude": -8.0,
                        "uncertainty": 0.2,
                        "unit": "kilocalories_per_mole",
                    },
                    "converged": True,
                    "reference": "https://example.test/source",
                },
                "lig_b": {
                    "dg": {
                        "magnitude": -6.5,
                        "uncertainty": 0.3,
                        "unit": "kilocalories_per_mole",
                    },
                    "converged": True,
                    "reference": "https://example.test/source",
                },
            }
        ),
        encoding="utf-8",
    )
    output = tmp_path / "experimental-registry.json"

    completed = subprocess.run(  # noqa: S603
        [
            sys.executable,
            str(ORACLE_DIR / "openfe_registry_builder.py"),
            "--protein-id",
            "benchmark",
            "--ligands-sdf",
            str(ligands_sdf),
            "--experimental-binding-json",
            str(experimental_json),
            "--experimental-registry-output",
            str(output),
        ],
        capture_output=True,
        check=False,
        cwd=ROOT,
        text=True,
    )

    assert completed.returncode == 0, completed.stderr
    payload = json.loads(output.read_text(encoding="utf-8"))
    assert payload["benchmark"]["CCO>>CCN"]["ddg_kcal_mol"] == pytest.approx(1.5)
    assert payload["benchmark"]["CCN>>CCO"]["ddg_kcal_mol"] == pytest.approx(-1.5)
    assert payload["benchmark"]["CCO>>CCN"]["ddg_uncertainty"] == pytest.approx(
        (0.2**2 + 0.3**2) ** 0.5
    )
    assert payload["benchmark"]["CCO>>CCN"]["method"] == "experimental_binding_free_energy"
    assert payload["benchmark"]["CCO>>CCN"]["per_repeat_ddg"] == {
        "repeat_1": pytest.approx(1.5)
    }


def test_openfe_json_runner_uses_result_registry_with_canonical_smiles(
    tmp_path: Path,
) -> None:
    registry = tmp_path / "openfe-result-registry.json"
    registry.write_text(
        json.dumps(
            {
                "tyk2": {
                    "CCO>>CCN": {
                        "ligand_a_smiles": "CCO",
                        "ligand_b_smiles": "CCN",
                        "ddg_kcal_mol": 0.8,
                        "ddg_uncertainty": 0.1,
                        "n_repeats": 1,
                        "method": "openfe",
                        "per_repeat_ddg": {"repeat_1": 0.8},
                        "converged": True,
                    }
                }
            }
        ),
        encoding="utf-8",
    )

    completed = _run_wrapper(
        "openfe_json_runner.py",
        {
            "project_id": "project-1",
            "request_id": "request-1",
            "batch_id": "batch-1",
            "protein_pdb_id": "tyk2",
            "reference_ligand_smiles": "OCC",
            "test_ligand_smiles": ["NCC"],
            "method": "openfe",
            "n_repeats": 1,
        },
        {"OPENFE_RESULT_REGISTRY": str(registry)},
    )

    payload = _json_stdout(completed)
    assert payload["request_id"] == "request-1"
    assert payload["batch_id"] == "batch-1"
    assert payload["results"][0]["ligand_a_smiles"] == "OCC"
    assert payload["results"][0]["ligand_b_smiles"] == "NCC"
    assert payload["results"][0]["ddg_kcal_mol"] == pytest.approx(0.8)
    assert payload["results"][0]["ddg_uncertainty"] == pytest.approx(0.1)


def test_openfe_json_runner_uses_registry_and_gathered_ddg(tmp_path: Path) -> None:
    complex_transformation = tmp_path / "edge-ccn-complex.json"
    solvent_transformation = tmp_path / "edge-ccn-solvent.json"
    _write_openfe_transformation(complex_transformation)
    _write_openfe_transformation(solvent_transformation)
    registry = tmp_path / "openfe-registry.json"
    registry.write_text(
        json.dumps(
            {
                "7abc": {
                    "CCO>>CCN": {
                        "complex": complex_transformation.name,
                        "solvent": solvent_transformation.name,
                    }
                }
            }
        ),
        encoding="utf-8",
    )
    result_registry = tmp_path / "openfe-result-registry.json"
    result_registry.write_text(json.dumps({"7abc": {}}), encoding="utf-8")
    log_path = tmp_path / "openfe-calls.jsonl"
    openfe = _write_executable(
        tmp_path / "openfe_fake.py",
        "\n".join(
            [
                "#!/usr/bin/env python3",
                "import json, os, pathlib, sys",
                "args = sys.argv[1:]",
                "expected_path_dir = os.environ.get('OPENFE_EXPECTED_PATH_DIR')",
                "if expected_path_dir:",
                "    assert expected_path_dir in os.environ['PATH'].split(os.pathsep)",
                "log_path = pathlib.Path(os.environ['OPENFE_FAKE_LOG'])",
                "with log_path.open('a', encoding='utf-8') as handle:",
                "    handle.write(json.dumps(args) + '\\n')",
                "if args[0] == 'quickrun':",
                "    expected = {",
                "        pathlib.Path(os.environ['COMPLEX_TRANSFORMATION_PATH']),",
                "        pathlib.Path(os.environ['SOLVENT_TRANSFORMATION_PATH']),",
                "    }",
                "    assert pathlib.Path(args[1]) in expected",
                "    is_complex = pathlib.Path(args[1]) == pathlib.Path(",
                "        os.environ['COMPLEX_TRANSFORMATION_PATH']",
                "    )",
                "    estimate = -2.291077 if is_complex else -1.4",
                "    uncertainty = 0.04 if is_complex else 0.05101255360203015",
                "    work_dir = pathlib.Path(args[args.index('-d') + 1])",
                "    assert work_dir.is_dir()",
                "    output = pathlib.Path(args[args.index('-o') + 1])",
                "    output.parent.mkdir(parents=True, exist_ok=True)",
                "    output.write_text(",
                "        json.dumps({",
                "            'estimate': {",
                "                'magnitude': estimate, 'unit': 'kilocalorie / mole'",
                "            },",
                "            'uncertainty': {",
                "                'magnitude': 0.0, 'unit': 'kilocalorie / mole'",
                "            },",
                "            'protocol_result': {'data': {'repeat-1': [{",
                "                'outputs': {",
                "                    'unit_estimate': {",
                "                        'magnitude': estimate,",
                "                        'unit': 'kilocalorie / mole',",
                "                    },",
                "                    'unit_estimate_error': {",
                "                        'magnitude': uncertainty,",
                "                        'unit': 'kilocalorie / mole',",
                "                    },",
                "                }",
                "            }]}},",
                "            'unit_results': {'unit-1': {}},",
                "        }),",
                "        encoding='utf-8',",
                "    )",
                "elif args[0] == 'gather':",
                "    assert args[args.index('--report') + 1] == 'ddg'",
                "    assert '--tsv' in args",
                "    output = pathlib.Path(args[args.index('-o') + 1])",
                "    output.write_text(",
                "        'ligand_i\\tligand_j\\tDDG(i->j) (kcal/mol)\\tuncertainty (kcal/mol)\\n'",
                "        'CCO\\tCCN\\t-0.89\\t0.06\\n',",
                "        encoding='utf-8',",
                "    )",
                "else:",
                "    raise SystemExit(f'unexpected command: {args}')",
            ]
        )
        + "\n",
    )

    completed = _run_wrapper(
        "openfe_json_runner.py",
        {
            "project_id": "project-1",
            "request_id": "request-1",
            "batch_id": "batch-1",
            "protein_pdb_id": "7abc",
            "reference_ligand_smiles": "CCO",
            "test_ligand_smiles": ["CCN"],
            "method": "openfe",
            "n_repeats": 1,
        },
        {
            "OPENFE_CLI_PATH": str(openfe),
            "OPENFE_FAKE_LOG": str(log_path),
            "OPENFE_TRANSFORMATION_REGISTRY": str(registry),
            "OPENFE_RESULT_REGISTRY": str(result_registry),
            "OPENFE_WORK_DIR": str(tmp_path / "work"),
            "OPENFE_EXPECTED_PATH_DIR": str(tmp_path),
            "COMPLEX_TRANSFORMATION_PATH": str(complex_transformation),
            "SOLVENT_TRANSFORMATION_PATH": str(solvent_transformation),
        },
    )

    payload = _json_stdout(completed)
    assert payload["request_id"] == "request-1"
    assert payload["batch_id"] == "batch-1"
    assert payload["results"][0]["ligand_a_smiles"] == "CCO"
    assert payload["results"][0]["ligand_b_smiles"] == "CCN"
    assert payload["results"][0]["ddg_kcal_mol"] == pytest.approx(-0.891077)
    assert payload["results"][0]["ddg_uncertainty"] == pytest.approx(0.064825)
    assert payload["results"][0]["per_repeat_ddg"] == {
        "repeat_1": pytest.approx(-0.891077)
    }
    calls = [json.loads(line) for line in log_path.read_text(encoding="utf-8").splitlines()]
    assert calls[0][0] == "quickrun"
    assert calls[1][0] == "quickrun"
    assert calls[2][0] == "gather"


def test_openfe_leg_requires_a_success_for_every_protocol_unit() -> None:
    openfe_leg_result = runpy.run_path(
        str(ORACLE_DIR / "openfe_json_runner.py")
    )["_openfe_leg_result"]
    quantity = {
        "magnitude": -2.0,
        "unit": "kilocalorie / mole",
    }
    result_payload = {
        "estimate": quantity,
        "protocol_result": {
            "data": {
                "repeat-1": [
                    {
                        "outputs": {
                            "unit_estimate": quantity,
                            "unit_estimate_error": {
                                "magnitude": 0.2,
                                "unit": "kilocalorie / mole",
                            },
                        }
                    }
                ]
            }
        },
        "unit_results": {
            "ProtocolUnitFailure-retry": {
                "name": "simulation",
                "source_key": "ProtocolUnit-simulation",
                "inputs": {},
                "outputs": {},
                "stderr": {},
                "stdout": {},
                "start_time": None,
                "end_time": None,
                "exception": ["RuntimeError", ["first attempt failed"]],
                "traceback": "first attempt failed",
            },
            "ProtocolUnitResult-retry": {
                "name": "simulation",
                "source_key": "ProtocolUnit-simulation",
                "inputs": {},
                "outputs": {},
                "stderr": {},
                "stdout": {},
                "start_time": None,
                "end_time": None,
            },
        },
    }

    assert openfe_leg_result(result_payload)[2] is True

    result_payload["unit_results"]["ProtocolUnitFailure-analysis"] = {
        "name": "analysis",
        "source_key": "ProtocolUnit-analysis",
        "inputs": {},
        "outputs": {},
        "stderr": {},
        "stdout": {},
        "start_time": None,
        "end_time": None,
        "exception": ["RuntimeError", ["all attempts failed"]],
        "traceback": "all attempts failed",
    }

    assert openfe_leg_result(result_payload)[2] is False


def test_openfe_json_runner_executes_and_aggregates_each_repeat(tmp_path: Path) -> None:
    complex_transformation = tmp_path / "edge-ccn-complex.json"
    solvent_transformation = tmp_path / "edge-ccn-solvent.json"
    _write_openfe_transformation(complex_transformation)
    _write_openfe_transformation(solvent_transformation)
    transformation_registry = tmp_path / "openfe-registry.json"
    transformation_registry.write_text(
        json.dumps(
            {
                "7abc": {
                    "CCO>>CCN": {
                        "complex": str(complex_transformation),
                        "solvent": str(solvent_transformation),
                    }
                }
            }
        ),
        encoding="utf-8",
    )
    result_registry = tmp_path / "openfe-result-registry.json"
    result_registry.write_text(
        json.dumps(
            {
                "7abc": {
                    "CCO>>CCN": {
                        "ligand_a_smiles": "CCO",
                        "ligand_b_smiles": "CCN",
                        "ddg_kcal_mol": -1.0,
                        "ddg_uncertainty": 0.2,
                        "n_repeats": 1,
                        "method": "openfe",
                        "per_repeat_ddg": {"repeat_1": -1.0},
                        "converged": True,
                    }
                }
            }
        ),
        encoding="utf-8",
    )
    log_path = tmp_path / "openfe-calls.jsonl"
    openfe = _write_executable(
        tmp_path / "openfe_fake.py",
        "\n".join(
            [
                "#!/usr/bin/env python3",
                "import json, os, pathlib, sys",
                "args = sys.argv[1:]",
                "log_path = pathlib.Path(os.environ['OPENFE_FAKE_LOG'])",
                "with log_path.open('a', encoding='utf-8') as handle:",
                "    handle.write(json.dumps(args) + '\\n')",
                "if args[0] == 'quickrun':",
                "    output = pathlib.Path(args[args.index('-o') + 1])",
                "    repeat_part = next(",
                "        part for part in output.parts if part.startswith('repeat-')",
                "    )",
                "    repeat_number = int(repeat_part.split('-')[1])",
                "    is_complex = 'complex' in pathlib.Path(args[1]).name",
                "    if is_complex:",
                "        estimate = -2.0 if repeat_number == 1 else -4.0",
                "    else:",
                "        estimate = -1.0",
                "    output.write_text(",
                "        json.dumps({",
                "            'estimate': {",
                "                'magnitude': estimate, 'unit': 'kilocalorie / mole'",
                "            },",
                "            'uncertainty': {",
                "                'magnitude': 0.0,",
                "                'unit': 'kilocalorie / mole',",
                "            },",
                "            'protocol_result': {'data': {'repeat-1': [{",
                "                'outputs': {",
                "                    'unit_estimate': {",
                "                        'magnitude': estimate,",
                "                        'unit': 'kilocalorie / mole',",
                "                    },",
                "                    'unit_estimate_error': {",
                "                        'magnitude': 0.1414213562373095,",
                "                        'unit': 'kilocalorie / mole',",
                "                    },",
                "                }",
                "            }]}},",
                "            'unit_results': {'unit-1': {}},",
                "        }),",
                "        encoding='utf-8',",
                "    )",
                "elif args[0] == 'gather':",
                "    output = pathlib.Path(args[args.index('-o') + 1])",
                "    repeat_part = next(",
                "        part for part in output.parts if part.startswith('repeat-')",
                "    )",
                "    repeat_number = int(repeat_part.split('-')[1])",
                "    ddg = -1.0 if repeat_number == 1 else -3.0",
                "    output.write_text(",
                "        'ligand_i\\tligand_j\\tDDG(i->j) (kcal/mol)\\tuncertainty (kcal/mol)\\n'",
                "        f'CCO\\tCCN\\t{ddg}\\t0.2\\n',",
                "        encoding='utf-8',",
                "    )",
                "else:",
                "    raise SystemExit(f'unexpected command: {args}')",
            ]
        )
        + "\n",
    )

    completed = _run_wrapper(
        "openfe_json_runner.py",
        {
            "project_id": "project-1",
            "request_id": "request-1",
            "batch_id": "batch-1",
            "protein_pdb_id": "7abc",
            "reference_ligand_smiles": "CCO",
            "test_ligand_smiles": ["CCN"],
            "method": "openfe",
            "n_repeats": 2,
        },
        {
            "OPENFE_CLI_PATH": str(openfe),
            "OPENFE_FAKE_LOG": str(log_path),
            "OPENFE_TRANSFORMATION_REGISTRY": str(transformation_registry),
            "OPENFE_RESULT_REGISTRY": str(result_registry),
            "OPENFE_WORK_DIR": str(tmp_path / "work"),
        },
    )

    payload = _json_stdout(completed)
    row = payload["results"][0]
    assert row["ddg_kcal_mol"] == pytest.approx(-2.0)
    assert row["ddg_uncertainty"] == pytest.approx(1.0099504938)
    assert row["n_repeats"] == 2
    assert row["per_repeat_ddg"] == {
        "repeat_1": pytest.approx(-1.0),
        "repeat_2": pytest.approx(-3.0),
    }
    calls = [json.loads(line) for line in log_path.read_text(encoding="utf-8").splitlines()]
    assert [call[0] for call in calls] == [
        "quickrun",
        "quickrun",
        "gather",
        "quickrun",
        "quickrun",
        "gather",
    ]


def test_openfe_json_runner_isolates_gather_by_requested_ligand_pair(
    tmp_path: Path,
) -> None:
    paths = {}
    for ligand in ("ccn", "ccc"):
        for phase in ("complex", "solvent"):
            path = tmp_path / f"edge-{ligand}-{phase}.json"
            _write_openfe_transformation(path)
            paths[(ligand, phase)] = path
    registry = tmp_path / "openfe-registry.json"
    registry.write_text(
        json.dumps(
            {
                "7abc": {
                    "CCO>>CCN": {
                        "complex": str(paths[("ccn", "complex")]),
                        "solvent": str(paths[("ccn", "solvent")]),
                        "ligand_a_name": "reference-ligand",
                        "ligand_b_name": "ligand-ccn",
                    },
                    "CCO>>CCC": {
                        "complex": str(paths[("ccc", "complex")]),
                        "solvent": str(paths[("ccc", "solvent")]),
                        "ligand_a_name": "reference-ligand",
                        "ligand_b_name": "ligand-ccc",
                    },
                }
            }
        ),
        encoding="utf-8",
    )
    openfe = _write_executable(
        tmp_path / "openfe_fake.py",
        "\n".join(
            [
                "#!/usr/bin/env python3",
                "import json, pathlib, sys",
                "args = sys.argv[1:]",
                "output = pathlib.Path(args[args.index('-o') + 1])",
                "if args[0] == 'quickrun':",
                "    group_number = int(",
                "        next(part for part in output.parts if part.startswith('group-'))",
                "        .split('-')[1]",
                "    )",
                "    is_complex = 'complex' in pathlib.Path(args[1]).name",
                "    if is_complex:",
                "        estimate = -2.0 if group_number == 0 else -4.0",
                "    else:",
                "        estimate = -1.0",
                "    output.write_text(json.dumps({",
                "        'estimate': {",
                "            'magnitude': estimate, 'unit': 'kilocalorie / mole'",
                "        },",
                "        'uncertainty': {",
                "            'magnitude': 0.0,",
                "            'unit': 'kilocalorie / mole',",
                "        },",
                "        'protocol_result': {'data': {'repeat-1': [{",
                "            'outputs': {",
                "                'unit_estimate': {",
                "                    'magnitude': estimate,",
                "                    'unit': 'kilocalorie / mole',",
                "                },",
                "                'unit_estimate_error': {",
                "                    'magnitude': 0.07071067811865475,",
                "                    'unit': 'kilocalorie / mole',",
                "                },",
                "            }",
                "        }]}},",
                "        'unit_results': {'unit-1': {}},",
                "    }), encoding='utf-8')",
                "elif args[0] == 'gather':",
                "    group_number = int(output.parent.name.split('-')[1])",
                "    ligand = 'ligand-ccn' if group_number == 0 else 'ligand-ccc'",
                "    ddg = -1.0 if group_number == 0 else -3.0",
                "    output.write_text(",
                "        'ligand_i\\tligand_j\\tDDG(i->j) (kcal/mol)\\tuncertainty (kcal/mol)\\n'",
                "        f'reference-ligand\\t{ligand}\\t{ddg}\\t0.1\\n',",
                "        encoding='utf-8',",
                "    )",
            ]
        )
        + "\n",
    )

    completed = _run_wrapper(
        "openfe_json_runner.py",
        {
            "project_id": "project-1",
            "request_id": "request-1",
            "batch_id": "batch-1",
            "protein_pdb_id": "7abc",
            "reference_ligand_smiles": "CCO",
            "test_ligand_smiles": ["CCN", "CCC"],
            "method": "openfe",
            "n_repeats": 1,
        },
        {
            "OPENFE_CLI_PATH": str(openfe),
            "OPENFE_TRANSFORMATION_REGISTRY": str(registry),
        },
    )

    payload = _json_stdout(completed)
    assert [row["ligand_b_smiles"] for row in payload["results"]] == ["CCN", "CCC"]
    assert [row["ddg_kcal_mol"] for row in payload["results"]] == pytest.approx(
        [-1.0, -3.0]
    )


def test_openfe_json_runner_rejects_replay_without_scientific_identity(
    tmp_path: Path,
) -> None:
    replay = tmp_path / "openfe-result.json"
    replay.write_text(
        json.dumps(
            {
                "results": [
                    {
                        "ligand_a_smiles": "CCO",
                        "ligand_b_smiles": "CCN",
                        "ddg_kcal_mol": -1.2,
                        "ddg_uncertainty": 0.3,
                        "n_repeats": 1,
                        "method": "openfe",
                        "per_repeat_ddg": {"repeat_1": -1.2},
                        "converged": True,
                    }
                ]
            }
        ),
        encoding="utf-8",
    )

    completed = _run_wrapper(
        "openfe_json_runner.py",
        {
            "project_id": "project-1",
            "request_id": "request-1",
            "batch_id": "batch-1",
            "protein_pdb_id": "7abc",
            "reference_ligand_smiles": "CCO",
            "test_ligand_smiles": ["CCN"],
            "method": "openfe",
            "n_repeats": 1,
        },
        {"OPENFE_RESULT_REPLAY_PATH": str(replay)},
    )

    assert completed.returncode == 2
    assert "missing scientific identity fields" in completed.stderr


def test_openfe_json_runner_rejects_mismatched_replay_scientific_identity(
    tmp_path: Path,
) -> None:
    replay = tmp_path / "openfe-result.json"
    replay.write_text(
        json.dumps(
            {
                "project_id": "project-1",
                "protein_pdb_id": "wrong-protein",
                "reference_ligand_smiles": "CCO",
                "test_ligand_smiles": ["CCN"],
                "method": "openfe",
                "n_repeats": 1,
                "results": [
                    {
                        "ligand_a_smiles": "CCO",
                        "ligand_b_smiles": "CCN",
                        "ddg_kcal_mol": -1.2,
                        "ddg_uncertainty": 0.3,
                        "n_repeats": 1,
                        "method": "openfe",
                        "per_repeat_ddg": {"repeat_1": -1.2},
                        "converged": True,
                    }
                ],
            }
        ),
        encoding="utf-8",
    )

    completed = _run_wrapper(
        "openfe_json_runner.py",
        {
            "project_id": "project-1",
            "request_id": "request-1",
            "batch_id": "batch-1",
            "protein_pdb_id": "7abc",
            "reference_ligand_smiles": "CCO",
            "test_ligand_smiles": ["CCN"],
            "method": "openfe",
            "n_repeats": 1,
        },
        {"OPENFE_RESULT_REPLAY_PATH": str(replay)},
    )

    assert completed.returncode == 2
    assert "protein_pdb_id" in completed.stderr


def test_fep_wrapper_propagates_openfe_data_error_exit_code(tmp_path: Path) -> None:
    openfe_runner = _write_executable(
        tmp_path / "openfe_fake.py",
        "\n".join(
            [
                "#!/usr/bin/env python3",
                "import sys",
                "print('invalid replay', file=sys.stderr)",
                "raise SystemExit(2)",
            ]
        )
        + "\n",
    )

    completed = _run_wrapper(
        "fep_oracle_wrapper.py",
        {
            "project_id": "project-1",
            "request_id": "request-1",
            "batch_id": "batch-1",
            "protein_pdb_id": "7abc",
            "reference_ligand_smiles": "CCO",
            "test_ligand_smiles": ["CCN"],
            "method": "openfe",
            "n_repeats": 1,
        },
        {"OPENFE_RUNNER_PATH": str(openfe_runner)},
    )

    assert completed.returncode == 2
    assert "invalid replay" in completed.stderr


def test_fep_wrapper_propagates_openfe_timeout_exit_code(tmp_path: Path) -> None:
    openfe_runner = _write_executable(
        tmp_path / "openfe_timeout.py",
        "#!/usr/bin/env python3\nraise SystemExit(124)\n",
    )

    completed = _run_wrapper(
        "fep_oracle_wrapper.py",
        {
            "project_id": "project-1",
            "request_id": "request-1",
            "batch_id": "batch-1",
            "protein_pdb_id": "7abc",
            "reference_ligand_smiles": "CCO",
            "test_ligand_smiles": ["CCN"],
            "method": "openfe",
            "n_repeats": 1,
        },
        {"OPENFE_RUNNER_PATH": str(openfe_runner)},
    )

    assert completed.returncode == 124
    assert "timed out" in completed.stderr


def test_fep_wrapper_classifies_response_identity_mismatch_as_data_error(
    tmp_path: Path,
) -> None:
    openfe_runner = _write_executable(
        tmp_path / "openfe_fake.py",
        "\n".join(
            [
                "#!/usr/bin/env python3",
                "import json, sys",
                "request = json.load(sys.stdin)",
                "request['request_id'] = 'wrong-request'",
                "request['results'] = [{",
                "    'ligand_a_smiles': 'CCO',",
                "    'ligand_b_smiles': 'CCN',",
                "    'ddg_kcal_mol': -1.0,",
                "    'ddg_uncertainty': 0.2,",
                "    'n_repeats': 1,",
                "    'method': 'openfe',",
                "    'per_repeat_ddg': {'repeat_1': -1.0},",
                "    'converged': True,",
                "}]",
                "print(json.dumps(request))",
            ]
        )
        + "\n",
    )

    completed = _run_wrapper(
        "fep_oracle_wrapper.py",
        {
            "project_id": "project-1",
            "request_id": "request-1",
            "batch_id": "batch-1",
            "protein_pdb_id": "7abc",
            "reference_ligand_smiles": "CCO",
            "test_ligand_smiles": ["CCN"],
            "method": "openfe",
            "n_repeats": 1,
        },
        {"OPENFE_RUNNER_PATH": str(openfe_runner)},
    )

    assert completed.returncode == 2
    assert "request_id" in completed.stderr


def test_openfe_json_runner_reports_missing_registry_pair(tmp_path: Path) -> None:
    registry = tmp_path / "openfe-registry.json"
    registry.write_text(json.dumps({"7abc": {}}), encoding="utf-8")

    completed = _run_wrapper(
        "openfe_json_runner.py",
        {
            "project_id": "project-1",
            "request_id": "request-1",
            "batch_id": "batch-1",
            "protein_pdb_id": "7abc",
            "reference_ligand_smiles": "CCO",
            "test_ligand_smiles": ["CCN"],
            "method": "openfe",
            "n_repeats": 1,
        },
        {"OPENFE_TRANSFORMATION_REGISTRY": str(registry)},
    )

    assert completed.returncode == 1
    assert "OpenFE transformation registry missing pair 7abc CCO>>CCN" in completed.stderr


def test_openfe_json_runner_requires_executable_transformation_input() -> None:
    completed = _run_wrapper(
        "openfe_json_runner.py",
        {
            "project_id": "project-1",
            "request_id": "request-1",
            "batch_id": "batch-1",
            "protein_pdb_id": "7abc",
            "reference_ligand_smiles": "CCO",
            "test_ligand_smiles": ["CCN"],
            "method": "openfe",
            "n_repeats": 1,
        },
        {},
    )

    assert completed.returncode == 1
    assert (
        "requires OPENFE_RESULT_REPLAY_PATH or openfe_transformation_json_paths"
        in completed.stderr
    )


def test_fep_wrapper_requires_explicit_openfe_runner_path() -> None:
    completed = _run_wrapper(
        "fep_oracle_wrapper.py",
        {
            "project_id": "project-1",
            "request_id": "request-1",
            "batch_id": "batch-1",
            "protein_pdb_id": "7abc",
            "reference_ligand_smiles": "CCO",
            "test_ligand_smiles": ["CCN"],
            "method": "openfe",
            "n_repeats": 1,
        },
        {"OPENFE_RUNNER_PATH": ""},
    )

    assert completed.returncode == 1
    assert "OPENFE_RUNNER_PATH is required" in completed.stderr


def test_openfe_json_runner_rejects_gather_without_uncertainty(
    tmp_path: Path,
) -> None:
    complex_transformation = tmp_path / "edge-complex.json"
    solvent_transformation = tmp_path / "edge-solvent.json"
    _write_openfe_transformation(complex_transformation)
    _write_openfe_transformation(solvent_transformation)
    registry = tmp_path / "openfe-registry.json"
    registry.write_text(
        json.dumps(
            {
                "7abc": {
                    "CCO>>CCN": {
                        "complex": str(complex_transformation),
                        "solvent": str(solvent_transformation),
                    }
                }
            }
        ),
        encoding="utf-8",
    )
    openfe = _write_executable(
        tmp_path / "openfe_fake.py",
        "\n".join(
            [
                "#!/usr/bin/env python3",
                "import json, pathlib, sys",
                "args = sys.argv[1:]",
                "output = pathlib.Path(args[args.index('-o') + 1])",
                "if args[0] == 'quickrun':",
                "    output.write_text(json.dumps({",
                "        'estimate': -1.0, 'uncertainty': 0.2, 'converged': True",
                "    }), encoding='utf-8')",
                "else:",
                "    output.write_text(",
                "        'ligand_i\\tligand_j\\tDDG(i->j) (kcal/mol)\\n'",
                "        'CCO\\tCCN\\t-1.0\\n',",
                "        encoding='utf-8',",
                "    )",
            ]
        )
        + "\n",
    )

    completed = _run_wrapper(
        "openfe_json_runner.py",
        {
            "project_id": "project-1",
            "request_id": "request-1",
            "batch_id": "batch-1",
            "protein_pdb_id": "7abc",
            "reference_ligand_smiles": "CCO",
            "test_ligand_smiles": ["CCN"],
            "method": "openfe",
            "n_repeats": 1,
        },
        {
            "OPENFE_CLI_PATH": str(openfe),
            "OPENFE_TRANSFORMATION_REGISTRY": str(registry),
        },
    )

    assert completed.returncode == 2
    assert "uncertainty" in completed.stderr


def test_openfe_json_runner_rejects_foreign_gather_pair_as_data_error(
    tmp_path: Path,
) -> None:
    complex_transformation = tmp_path / "edge-complex.json"
    solvent_transformation = tmp_path / "edge-solvent.json"
    _write_openfe_transformation(complex_transformation)
    _write_openfe_transformation(solvent_transformation)
    registry = tmp_path / "openfe-registry.json"
    registry.write_text(
        json.dumps(
            {
                "7abc": {
                    "CCO>>CCN": {
                        "complex": str(complex_transformation),
                        "solvent": str(solvent_transformation),
                    }
                }
            }
        ),
        encoding="utf-8",
    )
    openfe = _write_executable(
        tmp_path / "openfe_fake.py",
        "\n".join(
            [
                "#!/usr/bin/env python3",
                "import json, pathlib, sys",
                "args = sys.argv[1:]",
                "output = pathlib.Path(args[args.index('-o') + 1])",
                "if args[0] == 'quickrun':",
                "    output.write_text(json.dumps({",
                "        'estimate': -1.0, 'uncertainty': 0.2, 'converged': True",
                "    }), encoding='utf-8')",
                "else:",
                "    output.write_text(",
                "        'ligand_i\\tligand_j\\tDDG(i->j) (kcal/mol)'",
                "        '\\tuncertainty (kcal/mol)\\n'",
                "        'CCO\\tCCC\\t-1.0\\t0.2\\n',",
                "        encoding='utf-8',",
                "    )",
            ]
        )
        + "\n",
    )

    completed = _run_wrapper(
        "openfe_json_runner.py",
        {
            "project_id": "project-1",
            "request_id": "request-1",
            "batch_id": "batch-1",
            "protein_pdb_id": "7abc",
            "reference_ligand_smiles": "CCO",
            "test_ligand_smiles": ["CCN"],
            "method": "openfe",
            "n_repeats": 1,
        },
        {
            "OPENFE_CLI_PATH": str(openfe),
            "OPENFE_TRANSFORMATION_REGISTRY": str(registry),
        },
    )

    assert completed.returncode == 2
    assert "ligand identity" in completed.stderr


@pytest.mark.parametrize(
    "registry_format",
    ("missing_leg", "string", "one_item_list", "two_item_list"),
)
def test_openfe_json_runner_rejects_incomplete_rbfe_registry(
    tmp_path: Path,
    registry_format: str,
) -> None:
    transformation = tmp_path / "edge-complex.json"
    transformation.write_text("{}", encoding="utf-8")
    registry_value: object
    if registry_format == "missing_leg":
        registry_value = {"complex": str(transformation)}
    elif registry_format == "string":
        registry_value = str(transformation)
    elif registry_format == "one_item_list":
        registry_value = [str(transformation)]
    else:
        registry_value = [str(transformation), str(transformation)]
    registry = tmp_path / "openfe-registry.json"
    registry.write_text(
        json.dumps(
            {
                "7abc": {
                    "CCO>>CCN": registry_value,
                }
            }
        ),
        encoding="utf-8",
    )
    openfe = _write_executable(
        tmp_path / "openfe_fake.py",
        "#!/usr/bin/env python3\nraise SystemExit(0)\n",
    )

    completed = _run_wrapper(
        "openfe_json_runner.py",
        {
            "project_id": "project-1",
            "request_id": "request-1",
            "batch_id": "batch-1",
            "protein_pdb_id": "7abc",
            "reference_ligand_smiles": "CCO",
            "test_ligand_smiles": ["CCN"],
            "method": "openfe",
            "n_repeats": 1,
        },
        {
            "OPENFE_CLI_PATH": str(openfe),
            "OPENFE_TRANSFORMATION_REGISTRY": str(registry),
        },
    )

    assert completed.returncode == 2
    assert "complex and solvent" in completed.stderr


def test_openfe_json_runner_rejects_nested_protocol_repeats(
    tmp_path: Path,
) -> None:
    complex_transformation = _write_openfe_transformation(
        tmp_path / "edge-complex.json",
        protocol_repeats=3,
    )
    solvent_transformation = _write_openfe_transformation(
        tmp_path / "edge-solvent.json"
    )
    registry = tmp_path / "openfe-registry.json"
    registry.write_text(
        json.dumps(
            {
                "7abc": {
                    "CCO>>CCN": {
                        "complex": str(complex_transformation),
                        "solvent": str(solvent_transformation),
                    }
                }
            }
        ),
        encoding="utf-8",
    )
    marker = tmp_path / "openfe-invoked"
    openfe = _write_executable(
        tmp_path / "openfe_fake.py",
        "#!/usr/bin/env python3\n"
        "import os\n"
        "from pathlib import Path\n"
        "Path(os.environ['OPENFE_MARKER']).write_text('invoked')\n",
    )

    completed = _run_wrapper(
        "openfe_json_runner.py",
        {
            "project_id": "project-1",
            "request_id": "request-1",
            "batch_id": "batch-1",
            "protein_pdb_id": "7abc",
            "reference_ligand_smiles": "CCO",
            "test_ligand_smiles": ["CCN"],
            "method": "openfe",
            "n_repeats": 1,
        },
        {
            "OPENFE_CLI_PATH": str(openfe),
            "OPENFE_MARKER": str(marker),
            "OPENFE_TRANSFORMATION_REGISTRY": str(registry),
        },
    )

    assert completed.returncode == 2
    assert "protocol_repeats=1" in completed.stderr
    assert not marker.exists()


def test_admet_wrapper_calls_configured_http_service(tmp_path: Path) -> None:
    requests: list[dict] = []

    class Handler(BaseHTTPRequestHandler):
        def do_POST(self) -> None:
            length = int(self.headers["Content-Length"])
            payload = json.loads(self.rfile.read(length))
            requests.append({"path": self.path, "payload": payload})
            response = {
                "results": [
                    {
                        "smiles": "CCO",
                        "predictions": {"clearance": 1.5, "herg": 0.2},
                        "uncertainties": {"clearance": 0.1, "herg": 0.03},
                    }
                ]
            }
            body = json.dumps(response).encode("utf-8")
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def log_message(self, format: str, *args: object) -> None:
            return

    server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        completed = _run_wrapper(
            "admet_oracle_wrapper.py",
            {
                "smiles": ["CCO"],
                "properties": ["clearance", "herg"],
                "return_uncertainty": True,
            },
            {
                "ADMET_SERVICE_URL": f"http://127.0.0.1:{server.server_port}",
                "ADMET_BATCH_SIZE": "8",
            },
        )
    finally:
        server.shutdown()
        thread.join(timeout=5)

    payload = _json_stdout(completed)
    assert requests == [
        {
            "path": "/predict",
            "payload": {
                "smiles": ["CCO"],
                "endpoints": ["clearance", "herg"],
                "batch_size": 8,
                "return_uncertainty": True,
            },
        }
    ]
    assert payload["results"][0]["predictions"] == {"clearance": 1.5, "herg": 0.2}
    assert payload["results"][0]["uncertainties"] == {"clearance": 0.1, "herg": 0.03}


def test_openadmet_json_runner_maps_cli_csv_predictions(tmp_path: Path) -> None:
    model_dir = tmp_path / "openadmet-clearance"
    model_dir.mkdir()
    openadmet = _write_executable(
        tmp_path / "openadmet_fake.py",
        "\n".join(
            [
                "#!/usr/bin/env python3",
                "import csv, pathlib, sys",
                "args = sys.argv",
                "assert args[1] == 'predict'",
                "input_path = pathlib.Path(args[args.index('--input-path') + 1])",
                "model_dir = pathlib.Path(args[args.index('--model-dir') + 1])",
                "output_path = pathlib.Path(args[args.index('--output-csv') + 1])",
                "assert model_dir.is_dir()",
                "assert args[args.index('--input-col') + 1] == 'smiles'",
                "assert args[args.index('--accelerator') + 1] == 'gpu'",
                "with input_path.open(newline='', encoding='utf-8') as handle:",
                "    rows = list(csv.DictReader(handle))",
                "assert rows == [{'smiles': 'CCO'}]",
                "with output_path.open('w', newline='', encoding='utf-8') as handle:",
                "    writer = csv.DictWriter(handle, fieldnames=[",
                "        'smiles', 'OADMET_PRED_clearance', 'OADMET_STD_clearance'",
                "    ])",
                "    writer.writeheader()",
                "    writer.writerow({",
                "        'smiles': 'CCO',",
                "        'OADMET_PRED_clearance': '1.5',",
                "        'OADMET_STD_clearance': '0.1',",
                "    })",
            ]
        )
        + "\n",
    )

    completed = _run_wrapper(
        "openadmet_json_runner.py",
        {
            "smiles": ["CCO"],
            "properties": ["clearance"],
            "return_uncertainty": True,
        },
        {
            "OPENADMET_BINARY": str(openadmet),
            "OPENADMET_MODEL_DIR": str(model_dir),
            "OPENADMET_ACCELERATOR": "gpu",
            "OPENADMET_PROPERTY_COLUMNS": "clearance=OADMET_PRED_clearance",
            "OPENADMET_UNCERTAINTY_COLUMNS": "clearance=OADMET_STD_clearance",
        },
    )

    payload = _json_stdout(completed)
    assert payload["results"] == [
        {
            "smiles": "CCO",
            "predictions": {"clearance": 1.5},
            "uncertainties": {"clearance": 0.1},
        }
    ]


def test_openadmet_json_runner_requires_model_dir_for_property() -> None:
    completed = _run_wrapper(
        "openadmet_json_runner.py",
        {"smiles": ["CCO"], "properties": ["clearance"]},
        {},
    )

    assert completed.returncode == 1
    assert "requires OPENADMET_MODEL_DIR" in completed.stderr


VALIDATION_MARKER = "synthetic_pipeline_validation_only"


def test_retrosyn_agent_rejects_synthetic_validation_route() -> None:
    from mf_core.proto_gen.moleculeforge.v1.retrosyn import retrosyn_pb2
    from retrosyn_agent import agent as retrosyn_agent

    route = retrosyn_pb2.SyntheticRoute(
        route_id="validation-route",
        source_engine=VALIDATION_MARKER,
    )

    with pytest.raises(retrosyn_agent.RetrosynRouteValueError, match="synthetic validation"):
        retrosyn_agent._route_from_proto(route)


@pytest.mark.asyncio
@pytest.mark.parametrize("marker_field", ("source_engine", "validation_marker"))
async def test_retrosyn_agent_rejects_synthetic_validation_planner_result(
    marker_field: str,
) -> None:
    from retrosyn_agent import agent as retrosyn_agent

    class Planner:
        async def find_routes(self, _smiles: str, max_routes: int) -> list[dict]:
            return [
                {
                    "route_id": "validation-route",
                    marker_field: VALIDATION_MARKER,
                    "steps": [],
                }
            ][:max_routes]

    with pytest.raises(retrosyn_agent.RetrosynRouteValueError, match="synthetic validation"):
        await retrosyn_agent._find_routes_with_planner(Planner(), "CCO", 1)


@pytest.mark.parametrize(
    ("catalog_version", "source"),
    (
        (VALIDATION_MARKER, "production-catalog"),
        ("production-v1", VALIDATION_MARKER),
    ),
)
def test_supply_agent_rejects_synthetic_validation_catalog_record(
    catalog_version: str,
    source: str,
) -> None:
    from supply_agent import agent as supply_agent

    identity = {
        "request_id": "request-supply",
        "project_id": "project-validation",
        "candidate_id": "candidate-1",
        "candidate_index": 0,
        "canonical_smiles": "CCO",
    }
    record = {
        **identity,
        "smiles": "CCO",
        "available": True,
        "catalog_id": "validation-ethanol",
        "source": source,
        "source_timestamp": "2026-01-01T00:00:00Z",
        "evidence_id": "validation-evidence",
        "catalog_version": catalog_version,
        "catalog_checksum": f"sha256:{'0' * 64}",
    }

    with pytest.raises(RuntimeError, match="synthetic validation"):
        supply_agent._availability_record(
            "CCO",
            record,
            expected_identity=identity,
        )


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("catalog_version", "source"),
    (
        (VALIDATION_MARKER, "production-catalog"),
        ("production-v1", VALIDATION_MARKER),
    ),
)
async def test_supply_agent_rejects_synthetic_validation_catalog_in_legacy_request(
    catalog_version: str,
    source: str,
) -> None:
    from supply_agent import agent as supply_agent

    class CatalogClient:
        async def check_availability(self, smiles: str) -> dict:
            return {
                "smiles": smiles,
                "available": True,
                "catalog_id": "validation-ethanol",
                "source": source,
                "source_timestamp": "2026-01-01T00:00:00Z",
                "catalog_version": catalog_version,
            }

    agent = supply_agent.SupplyAgent(
        supply_client=CatalogClient(),
        crg_repository=None,
    )

    with pytest.raises(RuntimeError, match="synthetic validation"):
        await agent.process(
            {
                "smiles": "CCO",
                "building_blocks": ["CCO"],
            }
        )


def test_admet_validation_marker_takes_precedence_over_model_version() -> None:
    from admet_svc import main as admet

    model_version = admet._admet_model_version_for_smiles(
        [
            {
                "smiles": "CCO",
                "model_version": "production-model-v1",
                "validation_marker": VALIDATION_MARKER,
            }
        ],
        "CCO",
    )

    assert model_version == VALIDATION_MARKER


def test_boltz_validation_marker_takes_precedence_over_model_version() -> None:
    from boltz2_svc import main as boltz

    affinity = boltz._binding_affinity(
        {
            "protein_pdb_id": "6OIM",
            "ligand_smiles": "CCO",
            "delta_g_kcal_mol": -7.0,
            "uncertainty": 0.1,
            "ki_nm": 10.0,
            "ensemble_size": 1,
            "per_member_dg": [-7.0],
            "model_version": "production-model-v1",
            "validation_marker": VALIDATION_MARKER,
        }
    )

    assert affinity.model_version == VALIDATION_MARKER


def test_fep_validation_marker_takes_precedence_over_model_version() -> None:
    from fep_svc import main as fep

    result = fep._fep_result_from_record(
        {
            "ligand_a_smiles": "CO",
            "ligand_b_smiles": "CCO",
            "ddg_kcal_mol": -1.2,
            "ddg_uncertainty": 0.1,
            "n_repeats": 1,
            "method": "openfe",
            "per_repeat_ddg": {"repeat_1": -1.2},
            "converged": True,
            "model_version": "production-model-v1",
            "validation_marker": VALIDATION_MARKER,
        }
    )

    assert result.model_version == VALIDATION_MARKER


def _validation_runner_payloads() -> dict[str, dict]:
    return {
        "admet_svc": {
            "smiles": ["CCO", "CCN"],
            "properties": ["admet_score", "toxicity"],
            "return_uncertainty": True,
        },
        "boltz2_svc": {
            "protein_pdb_id": "6OIM",
            "ligand_smiles": ["CCO", "CCN"],
            "ensemble_size": 3,
        },
        "dock_svc": {
            "engine": "gnina",
            "smiles": "CCO",
            "protein_pdb": "file:///validation/receptor.pdb",
        },
        "fep_svc": {
            "project_id": "project-validation",
            "request_id": "request-validation",
            "batch_id": "batch-validation",
            "protein_pdb_id": "7ABC",
            "reference_ligand_smiles": "CO",
            "test_ligand_smiles": ["CCO", "CCN"],
            "method": "openfe",
            "n_repeats": 3,
        },
        "retrosyn_svc": {
            "smiles": "CCO",
            "max_routes": 1,
            "engine": "aizynth",
        },
    }


def _validation_runner_command(module_name: str) -> str:
    return f"{sys.executable} -m {module_name}.main --validation-runner"


def _run_validation_runner(
    module_name: str,
    payload: dict,
    *,
    allow: bool,
) -> subprocess.CompletedProcess[str]:
    env = os.environ.copy()
    if allow:
        env["MF_ALLOW_SYNTHETIC_VALIDATION"] = "true"
    else:
        env.pop("MF_ALLOW_SYNTHETIC_VALIDATION", None)
    return subprocess.run(  # noqa: S603
        [sys.executable, "-m", f"{module_name}.main", "--validation-runner"],
        input=json.dumps(payload, sort_keys=True),
        capture_output=True,
        check=False,
        cwd=ROOT,
        env=env,
        text=True,
        timeout=10,
    )


@pytest.mark.parametrize(
    ("module_name", "payload"),
    tuple(_validation_runner_payloads().items()),
)
def test_synthetic_validation_runners_require_explicit_gate(
    module_name: str,
    payload: dict,
) -> None:
    completed = _run_validation_runner(module_name, payload, allow=False)

    assert completed.returncode == 1
    assert completed.stdout == ""
    assert "MF_ALLOW_SYNTHETIC_VALIDATION=true" in completed.stderr


@pytest.mark.parametrize(
    ("module_name", "payload"),
    tuple(_validation_runner_payloads().items()),
)
def test_synthetic_validation_runners_reject_unknown_protocol_fields(
    module_name: str,
    payload: dict,
) -> None:
    invalid_payload = {**payload, "unexpected": True}

    completed = _run_validation_runner(module_name, invalid_payload, allow=True)

    assert completed.returncode == 2
    assert completed.stdout == ""
    assert "unexpected fields" in completed.stderr


@pytest.mark.parametrize(
    ("module_name", "payload"),
    tuple(_validation_runner_payloads().items()),
)
def test_synthetic_validation_runner_outputs_are_marked_and_deterministic(
    module_name: str,
    payload: dict,
) -> None:
    first = _run_validation_runner(module_name, payload, allow=True)
    second = _run_validation_runner(module_name, payload, allow=True)

    assert first.returncode == 0, first.stderr
    assert second.returncode == 0, second.stderr
    assert first.stdout == second.stdout
    response = json.loads(first.stdout)
    assert response["validation_marker"] == VALIDATION_MARKER
    result_fields = {
        "admet_svc": "results",
        "boltz2_svc": "affinities",
        "fep_svc": "results",
        "retrosyn_svc": "routes",
    }
    result_field = result_fields.get(module_name)
    if result_field is not None:
        assert all(
            item["validation_marker"] == VALIDATION_MARKER
            for item in response[result_field]
        )


@pytest.mark.asyncio
async def test_synthetic_validation_runners_cross_production_command_adapters(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from types import SimpleNamespace

    from admet_svc import main as admet
    from boltz2_svc import main as boltz
    from dock_svc import main as dock
    from fep_svc import main as fep
    from mf_core.proto_gen.moleculeforge.v1.oracle import fep_pb2
    from mf_core.proto_gen.moleculeforge.v1.retrosyn import retrosyn_pb2
    from mf_retrosyn._route_validation import partition_retrosyn_results
    from retrosyn_svc import main as retrosyn

    monkeypatch.setenv("MF_ALLOW_SYNTHETIC_VALIDATION", "true")

    admet_rows = admet.ADMETCommandRunner(_validation_runner_command("admet_svc")).predict(
        ["CCO", "CCN"],
        ["admet_score", "toxicity"],
        True,
    )
    assert [row["smiles"] for row in admet_rows] == ["CCO", "CCN"]
    assert all(set(row["predictions"]) == {"admet_score", "toxicity"} for row in admet_rows)
    assert all(set(row["uncertainties"]) == {"admet_score", "toxicity"} for row in admet_rows)
    assert all(row["validation_marker"] == VALIDATION_MARKER for row in admet_rows)

    boltz_rows = boltz.BoltzCommandRunner(
        _validation_runner_command("boltz2_svc")
    ).predict_affinity(
        "6OIM",
        ["CCO", "CCN"],
        3,
    )
    assert [row["ligand_smiles"] for row in boltz_rows] == ["CCO", "CCN"]
    assert all(row["protein_pdb_id"] == "6OIM" for row in boltz_rows)
    assert all(len(row["per_member_dg"]) == 3 for row in boltz_rows)
    assert all(
        row["delta_g_kcal_mol"] == pytest.approx(sum(row["per_member_dg"]) / 3)
        for row in boltz_rows
    )
    assert all(row["validation_marker"] == VALIDATION_MARKER for row in boltz_rows)

    monkeypatch.setenv("DOCK_ORACLE_COMMAND", _validation_runner_command("dock_svc"))
    dock_result = dock._run_dock_command(
        SimpleNamespace(
            smiles="CCO",
            protein_pdb="file:///validation/receptor.pdb",
        ),
        "gnina",
    )
    assert dock_result.smiles == "CCO"
    assert dock_result.receptor_uri == "file:///validation/receptor.pdb"
    assert dock_result.engine == "gnina"
    assert dock_result.validation_marker == VALIDATION_MARKER

    monkeypatch.setenv("FEP_ORACLE_COMMAND", _validation_runner_command("fep_svc"))
    fep_request = fep_pb2.FEPBatchRequest(
        project_id="project-validation",
        request_id="request-validation",
        batch_id="batch-validation",
        protein_pdb_id="7ABC",
        reference_ligand_smiles="CO",
        test_ligand_smiles=["CCO", "CCN"],
        method="openfe",
        n_repeats=3,
    )
    fep_response = await fep._run_fep_command_async(fep_request)
    assert fep_response.project_id == fep_request.project_id
    assert fep_response.request_id == fep_request.request_id
    assert fep_response.batch_id == fep_request.batch_id
    assert list(fep_response.test_ligand_smiles) == list(fep_request.test_ligand_smiles)
    assert [result.ligand_b_smiles for result in fep_response.results] == ["CCO", "CCN"]
    assert all(result.model_version == VALIDATION_MARKER for result in fep_response.results)
    assert all(
        set(result.per_repeat_ddg) == {"repeat_1", "repeat_2", "repeat_3"}
        for result in fep_response.results
    )
    raw_fep = _run_validation_runner(
        "fep_svc",
        _validation_runner_payloads()["fep_svc"],
        allow=True,
    )
    raw_fep_payload = json.loads(raw_fep.stdout)
    assert raw_fep_payload["validation_marker"] == VALIDATION_MARKER
    assert all(
        result["validation_marker"] == VALIDATION_MARKER
        for result in raw_fep_payload["results"]
    )

    retrosyn_routes = await retrosyn._run_planner_command(
        _validation_runner_command("retrosyn_svc"),
        smiles="CCO",
        max_routes=1,
        engine="aizynth",
    )
    executable, assessments = partition_retrosyn_results(
        retrosyn_routes,
        "synthetic validation planner",
    )
    assert assessments == []
    assert len(executable) == 1
    assert executable[0]["source_engine"] == VALIDATION_MARKER
    assert executable[0]["validation_marker"] == VALIDATION_MARKER

    monkeypatch.setenv(
        "RETROSYN_PLANNER_COMMAND",
        _validation_runner_command("retrosyn_svc"),
    )
    retrosyn_response = await retrosyn.RetrosynServicer().FindRoutes(
        retrosyn_pb2.RetrosynthesisRequest(
            project_id="project-validation",
            request_id="request-retrosyn",
            molecule_smiles="CCO",
            canonical_smiles="CCO",
            max_routes=1,
            engine="aizynth",
        ),
        None,
    )
    assert len(retrosyn_response.routes) == 1
    assert retrosyn_response.routes[0].source_engine == VALIDATION_MARKER
    assert retrosyn_response.routes[0].reaction_smiles == ["CO.C>>CCO"]
    assert retrosyn_response.routes[0].steps[0].yield_fraction == pytest.approx(0.75)


@pytest.mark.asyncio
async def test_multilevel_validation_consumer_rejects_synthetic_service_results(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    from admet_svc import main as admet
    from boltz2_svc import main as boltz
    from dock_svc import main as dock
    from fep_svc import main as fep
    from mf_core.proto_gen.moleculeforge.v1.oracle import oracle_pb2

    monkeypatch.setenv("MF_ALLOW_SYNTHETIC_VALIDATION", "true")
    candidates = ["CCO", "CCN"]

    admet_response = await admet.ADMETOracleServicer(
        service=admet.ADMETServicer(
            runner=admet.ADMETCommandRunner(
                _validation_runner_command("admet_svc")
            ),
        )
    ).Evaluate(
        oracle_pb2.OracleBatchRequest(
            project_id="project-validation",
            request_id="request-admet",
            molecule_smiles=candidates,
            requested_properties=["admet_score"],
            level=oracle_pb2.L1_ML_SURROGATE,
        ),
        None,
    )
    assert [item.molecule_smiles for item in admet_response.evaluations] == candidates
    assert all(not item.success for item in admet_response.evaluations)
    assert all(item.error_code == "SYNTHETIC_VALIDATION_ONLY" for item in admet_response.evaluations)
    assert all(item.model_version == VALIDATION_MARKER for item in admet_response.evaluations)

    boltz_response = await boltz.Boltz2OracleServicer(
        service=boltz.Boltz2Servicer(
            runner=boltz.BoltzCommandRunner(
                _validation_runner_command("boltz2_svc")
            ),
        )
    ).Evaluate(
        oracle_pb2.OracleBatchRequest(
            project_id="project-validation",
            request_id="request-boltz",
            molecule_smiles=candidates,
            requested_properties=["affinity", "ki_nm"],
            level=oracle_pb2.L1_ML_SURROGATE,
            protein_pdb_id="6OIM",
            oracle_parameters={"ensemble_size": "3"},
        ),
        None,
    )
    assert [item.molecule_smiles for item in boltz_response.evaluations] == candidates
    assert all(not item.success for item in boltz_response.evaluations)
    assert all(item.error_code == "SYNTHETIC_VALIDATION_ONLY" for item in boltz_response.evaluations)
    assert all(item.model_version == VALIDATION_MARKER for item in boltz_response.evaluations)

    receptor = tmp_path / "receptor.pdb"
    receptor.write_text(
        "ATOM      1  N   ALA A   1      11.104  13.207   2.345  1.00 20.00           N\n",
        encoding="utf-8",
    )
    monkeypatch.setenv("DOCK_ORACLE_COMMAND", _validation_runner_command("dock_svc"))
    dock_response = await dock.DockOracleServicer(service=dock.DockServicer()).Evaluate(
        oracle_pb2.OracleBatchRequest(
            project_id="project-validation",
            request_id="request-dock",
            molecule_smiles=candidates,
            requested_properties=["docking_score"],
            level=oracle_pb2.L2_DOCKING,
            receptor_uri=receptor.as_uri(),
            oracle_parameters={"engine": "gnina"},
        ),
        None,
    )
    assert [item.molecule_smiles for item in dock_response.evaluations] == candidates
    assert all(not item.success for item in dock_response.evaluations)
    assert all(item.error_code == "SYNTHETIC_VALIDATION_ONLY" for item in dock_response.evaluations)
    assert all(item.model_version == VALIDATION_MARKER for item in dock_response.evaluations)

    monkeypatch.setenv("FEP_ORACLE_COMMAND", _validation_runner_command("fep_svc"))
    fep_response = await fep.FEPOracleServicer(
        service=fep.FEPServicer(job_dir=tmp_path / "fep-jobs"),
    ).Evaluate(
        oracle_pb2.OracleBatchRequest(
            project_id="project-validation",
            request_id="request-fep",
            molecule_smiles=candidates,
            requested_properties=["rbfe"],
            level=oracle_pb2.L3_FEP,
            protein_pdb_id="7ABC",
            reference_ligand_smiles="CO",
            oracle_parameters={"method": "openfe", "n_repeats": "3"},
        ),
        None,
    )
    assert [item.molecule_smiles for item in fep_response.evaluations] == candidates
    assert all(not item.success for item in fep_response.evaluations)
    assert all(item.error_code == "SYNTHETIC_VALIDATION_ONLY" for item in fep_response.evaluations)
    assert all(item.model_version == VALIDATION_MARKER for item in fep_response.evaluations)


@pytest.mark.asyncio
async def test_supply_validation_catalog_bootstrap_is_atomic_and_consumable(
    tmp_path: Path,
) -> None:
    from types import SimpleNamespace

    from supply_oracle_svc import main as supply

    target = tmp_path / "supply-catalog.json"
    env = os.environ.copy()
    env.pop("MF_ALLOW_SYNTHETIC_VALIDATION", None)
    blocked = subprocess.run(  # noqa: S603
        [
            sys.executable,
            "-m",
            "supply_oracle_svc.main",
            "--bootstrap-validation-catalog",
            str(target),
        ],
        capture_output=True,
        check=False,
        cwd=ROOT,
        env=env,
        text=True,
        timeout=10,
    )
    assert blocked.returncode == 1
    assert "MF_ALLOW_SYNTHETIC_VALIDATION=true" in blocked.stderr
    assert not target.exists()

    env["MF_ALLOW_SYNTHETIC_VALIDATION"] = "true"
    completed = subprocess.run(  # noqa: S603
        [
            sys.executable,
            "-m",
            "supply_oracle_svc.main",
            "--bootstrap-validation-catalog",
            str(target),
        ],
        capture_output=True,
        check=False,
        cwd=ROOT,
        env=env,
        text=True,
        timeout=10,
    )
    assert completed.returncode == 0, completed.stderr
    bootstrap_result = json.loads(completed.stdout)
    assert bootstrap_result["validation_marker"] == VALIDATION_MARKER
    assert bootstrap_result["catalog_uri"] == target.resolve().as_uri()

    payload = json.loads(target.read_text(encoding="utf-8"))
    assert payload["validation_marker"] == VALIDATION_MARKER
    assert payload["catalog_version"] == VALIDATION_MARKER
    assert all(record["validation_marker"] == VALIDATION_MARKER for record in payload["records"])
    assert all(record["source"] == VALIDATION_MARKER for record in payload["records"])
    assert {path.name for path in tmp_path.iterdir()} == {target.name}
    supply._validate_catalog_schema(target.resolve().as_uri())

    service = supply.SupplyOracleServicer(
        catalog_client=supply.FileSupplyCatalog(target.resolve().as_uri())
    )
    response = await service.CheckAvailability(
        SimpleNamespace(
            request_id="request-supply",
            project_id="project-validation",
            candidate_id="candidate-1",
            candidate_index=0,
            canonical_smiles="CCO",
            smiles="CCO",
        ),
        None,
    )
    assert response.available is True
    assert response.catalog_version == VALIDATION_MARKER
    assert response.catalog_source == VALIDATION_MARKER

    original = target.read_bytes()
    duplicate = subprocess.run(  # noqa: S603
        [
            sys.executable,
            "-m",
            "supply_oracle_svc.main",
            "--bootstrap-validation-catalog",
            str(target),
        ],
        capture_output=True,
        check=False,
        cwd=ROOT,
        env=env,
        text=True,
        timeout=10,
    )
    assert duplicate.returncode == 0, duplicate.stderr
    assert json.loads(duplicate.stdout) == bootstrap_result
    assert target.read_bytes() == original

    production_target = tmp_path / "production-catalog.json"
    production_target.write_text(
        json.dumps(
            {
                "catalog_version": "production-v1",
                "records": [
                    {
                        "smiles": "CCO",
                        "available": True,
                        "catalog_id": "production-ethanol",
                        "source": "production",
                        "source_timestamp": "2026-01-01T00:00:00Z",
                    }
                ],
            }
        ),
        encoding="utf-8",
    )
    production_bytes = production_target.read_bytes()
    refused = subprocess.run(  # noqa: S603
        [
            sys.executable,
            "-m",
            "supply_oracle_svc.main",
            "--bootstrap-validation-catalog",
            str(production_target),
        ],
        capture_output=True,
        check=False,
        cwd=ROOT,
        env=env,
        text=True,
        timeout=10,
    )
    assert refused.returncode == 1
    assert "already exists" in refused.stderr
    assert production_target.read_bytes() == production_bytes
