from __future__ import annotations

import json
import os
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
            "protein_pdb_id": "7abc",
            "reference_ligand_smiles": "CCO",
            "test_ligand_smiles": ["CCN"],
            "method": "openfe",
            "n_repeats": 2,
        },
        {"OPENFE_RUNNER_PATH": str(openfe_runner)},
    )

    payload = _json_stdout(completed)
    assert payload["batch_id"] == "project-1"
    assert payload["total_elapsed_ms"] == 33
    assert payload["results"][0]["ddg_kcal_mol"] == pytest.approx(-1.2)
    assert payload["results"][0]["converged"] is True


def test_openfe_json_runner_replays_configured_result(tmp_path: Path) -> None:
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
            "protein_pdb_id": "7abc",
            "reference_ligand_smiles": "CCO",
            "test_ligand_smiles": ["CCN"],
            "method": "openfe",
            "n_repeats": 1,
        },
        {"OPENFE_RESULT_REPLAY_PATH": str(replay)},
    )

    payload = _json_stdout(completed)
    assert payload["batch_id"] == "project-1"
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
    complex_transformation.write_text("{}", encoding="utf-8")
    solvent_transformation.write_text("{}", encoding="utf-8")
    ddg_tsv = tmp_path / "ddg.tsv"
    ddg_tsv.write_text(
        "\n".join(
            [
                "ligand_i\tligand_j\tDDG(i->j) (kcal/mol)\tuncertainty (kcal/mol)",
                "lig_a\tlig_b\t1.25\t0.4",
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
        "complex": str(complex_transformation.resolve()),
        "solvent": str(solvent_transformation.resolve()),
    }
    assert result_payload["tyk2"]["CCO>>CCN"]["ddg_kcal_mol"] == pytest.approx(1.25)
    assert result_payload["tyk2"]["CCO>>CCN"]["ddg_uncertainty"] == pytest.approx(0.4)


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
                        "n_repeats": 3,
                        "method": "openfe",
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
            "protein_pdb_id": "tyk2",
            "reference_ligand_smiles": "OCC",
            "test_ligand_smiles": ["NCC"],
            "method": "openfe",
            "n_repeats": 1,
        },
        {"OPENFE_RESULT_REGISTRY": str(registry)},
    )

    payload = _json_stdout(completed)
    assert payload["batch_id"] == "project-1"
    assert payload["results"][0]["ligand_a_smiles"] == "CCO"
    assert payload["results"][0]["ligand_b_smiles"] == "CCN"
    assert payload["results"][0]["ddg_kcal_mol"] == pytest.approx(0.8)
    assert payload["results"][0]["ddg_uncertainty"] == pytest.approx(0.1)


def test_openfe_json_runner_uses_registry_and_gathered_ddg(tmp_path: Path) -> None:
    complex_transformation = tmp_path / "edge-ccn-complex.json"
    solvent_transformation = tmp_path / "edge-ccn-solvent.json"
    complex_transformation.write_text("{}", encoding="utf-8")
    solvent_transformation.write_text("{}", encoding="utf-8")
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
                "    work_dir = pathlib.Path(args[args.index('-d') + 1])",
                "    assert work_dir.is_dir()",
                "    output = pathlib.Path(args[args.index('-o') + 1])",
                "    output.parent.mkdir(parents=True, exist_ok=True)",
                "    output.write_text(json.dumps({'estimate': -1.4, 'uncertainty': 0.2}), encoding='utf-8')",
                "elif args[0] == 'gather':",
                "    assert args[args.index('--report') + 1] == 'ddg'",
                "    assert '--tsv' in args",
                "    output = pathlib.Path(args[args.index('-o') + 1])",
                "    output.write_text(",
                "        'ligand_i\\tligand_j\\tDDG(i->j) (kcal/mol)\\tuncertainty (kcal/mol)\\n'",
                "        'CCO\\tCCN\\t-1.4\\t0.2\\n',",
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
    assert payload["batch_id"] == "project-1"
    assert payload["results"][0]["ligand_a_smiles"] == "CCO"
    assert payload["results"][0]["ligand_b_smiles"] == "CCN"
    assert payload["results"][0]["ddg_kcal_mol"] == pytest.approx(-1.4)
    assert payload["results"][0]["ddg_uncertainty"] == pytest.approx(0.2)
    calls = [json.loads(line) for line in log_path.read_text(encoding="utf-8").splitlines()]
    assert calls[0][0] == "quickrun"
    assert calls[1][0] == "quickrun"
    assert calls[2][0] == "gather"


def test_openfe_json_runner_reports_missing_registry_pair(tmp_path: Path) -> None:
    registry = tmp_path / "openfe-registry.json"
    registry.write_text(json.dumps({"7abc": {}}), encoding="utf-8")

    completed = _run_wrapper(
        "openfe_json_runner.py",
        {
            "project_id": "project-1",
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
            "protein_pdb_id": "7abc",
            "reference_ligand_smiles": "CCO",
            "test_ligand_smiles": ["CCN"],
            "method": "openfe",
            "n_repeats": 1,
        },
        {},
    )

    assert completed.returncode == 1
    assert "requires OPENFE_RESULT_REPLAY_PATH or openfe_transformation_json_paths" in completed.stderr


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
