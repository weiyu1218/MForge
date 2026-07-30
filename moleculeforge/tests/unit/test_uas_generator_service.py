from __future__ import annotations

import asyncio
import hashlib
import importlib.util
import json
import math
import os
import shlex
import signal
import struct
import subprocess
import sys
import time
from pathlib import Path
from types import ModuleType

import grpc
import pytest
import torch
from mf_chem.molecule.parsing import canonicalize
from mf_core.proto_gen.moleculeforge.v1.core import audit_pb2, cig_pb2, humu_pb2
from mf_core.proto_gen.moleculeforge.v1.generator import generator_pb2, generator_pb2_grpc
from mf_generators.uas.autoencoder.molecule_ae import MoleculeAutoencoder

ROOT = Path(__file__).resolve().parents[2]
SERVICE_MAIN = ROOT / "services/uas-generator-svc/src/uas_generator_svc/main.py"
COMMAND_SCHEMA = "uas.command.v1"


def _load_service_module() -> ModuleType:
    spec = importlib.util.spec_from_file_location("uas_generator_service_test", SERVICE_MAIN)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def _write_runtime_files(tmp_path: Path) -> dict[str, str]:
    artifact_dir = tmp_path / "artifacts"
    artifact_dir.mkdir()
    checkpoint_path = artifact_dir / "autoencoder.pt"
    model = MoleculeAutoencoder(input_dim=129, latent_dim=2)
    for parameter in model.parameters():
        parameter.data.zero_()
    torch.save(model.state_dict(), checkpoint_path)
    checksum = hashlib.sha256(checkpoint_path.read_bytes()).hexdigest()
    manifest_path = artifact_dir / "training_manifest.json"
    manifest_path.write_text(
        json.dumps(
            {
                "schema_version": "uas_training.v1",
                "records": 2,
                "dim": 129,
                "latent_dim": 2,
                "autoencoder_path": "autoencoder.pt",
                "autoencoder_sha256": f"sha256:{checksum}",
            }
        ),
        encoding="utf-8",
    )

    candidate_script = tmp_path / "candidate.py"
    candidate_script.write_text(
        "import json,sys\n"
        "request=json.load(sys.stdin)\n"
        "count=request['n_samples']\n"
        "dim=request['embedding_dim']\n"
        "print(json.dumps({'schema_version':'uas.command.v1',"
        "'embeddings':[[0.0]*dim for _ in range(count)]}))\n",
        encoding="utf-8",
    )
    decoder_script = tmp_path / "decoder.py"
    decoder_script.write_text(
        "import json,sys\n"
        "request=json.load(sys.stdin)\n"
        "print(json.dumps({'schema_version':'uas.command.v1',"
        "'candidates':[{'smiles':'CCO','canonical_smiles':'CCO'}"
        " for _ in request['embeddings']]}))\n",
        encoding="utf-8",
    )
    return {
        "UAS_AUTOENCODER_PATH": str(checkpoint_path),
        "UAS_ARTIFACT_MANIFEST_PATH": str(manifest_path),
        "UAS_CANDIDATE_SOURCE_COMMAND": shlex.join([sys.executable, str(candidate_script)]),
        "UAS_DECODER_COMMAND": shlex.join([sys.executable, str(decoder_script)]),
    }


def _valid_generate_request() -> generator_pb2.GenerateRequest:
    coordinates = [1.0, *([0.0] * 128)]
    return generator_pb2.GenerateRequest(
        project_id="project-1",
        request_id="request-1",
        batch_size=2,
        total_molecules=2,
        context_schema_version="generator_context.v1",
        cig=cig_pb2.CIG(
            project_id="project-1",
            objectives=[
                cig_pb2.ObjectiveNode(
                    id="qed",
                    name="maximize QED",
                    type=cig_pb2.MAXIMIZE,
                    property="qed",
                    weight=1.0,
                    pareto_tier=1,
                )
            ],
            created_by="test",
        ),
        hciv=humu_pb2.HCIV(coordinates=coordinates, curvature=1.0),
        intent_cone=humu_pb2.IntentCone(
            axis=coordinates,
            half_angle=0.5,
            curvature=1.0,
            property_weights={"qed": 1.0},
        ).SerializeToString(),
        generator_params={"sampling_seed": "7"},
    )


def _valid_candidate_command_payload(n_samples: int = 4) -> dict[str, object]:
    coordinates = [1.0, *([0.0] * 128)]
    return {
        "schema_version": COMMAND_SCHEMA,
        "operation": "sample",
        "project_id": "project-1",
        "request_id": "request-1",
        "n_samples": n_samples,
        "embedding_dim": 129,
        "seed": 7,
        "attempt": 1,
        "hciv": {
            "coordinates": coordinates,
            "dim": 128,
            "curvature": 1.0,
            "manifold_type": "lorentz",
            "molecule_smiles": "",
            "parent_hciv_id": None,
        },
        "intent_cone": {
            "axis": coordinates,
            "half_angle": 0.5,
            "angle_radians": 0.5,
            "curvature": 1.0,
            "property_weights": {},
            "apex": None,
            "axis_direction": None,
            "length": 1.0,
        },
        "cig": {
            "project_id": "project-1",
            "objectives": [],
            "edges": [],
            "hyperedges": [],
            "constraints": {},
            "created_by": "test",
        },
        "generator_params": {},
    }


def _valid_decoder_command_payload(n_samples: int = 4) -> dict[str, object]:
    coordinates = [1.0, *([0.0] * 128)]
    return {
        "schema_version": COMMAND_SCHEMA,
        "operation": "decode",
        "project_id": "project-1",
        "request_id": "request-1",
        "embeddings": [coordinates for _ in range(n_samples)],
        "context": {
            "context_schema_version": "generator_context.v1",
            "hciv": {
                "coordinates": coordinates,
                "dim": 128,
                "curvature": 1.0,
                "manifold_type": "lorentz",
                "molecule_smiles": "",
                "parent_hciv_id": None,
            },
            "intent_cone": {
                "axis": coordinates,
                "half_angle": 0.5,
                "angle_radians": 0.5,
                "curvature": 1.0,
                "property_weights": {},
                "apex": None,
                "axis_direction": None,
                "length": 1.0,
            },
            "cig": {
                "project_id": "project-1",
                "objectives": [],
                "edges": [],
                "hyperedges": [],
                "constraints": {},
                "created_by": "test",
            },
            "generator_params": {},
        },
    }


def _run_uas_cli(command: str, payload: dict[str, object]) -> subprocess.CompletedProcess[str]:
    return subprocess.run(  # noqa: S603
        [sys.executable, "-m", "uas_generator_svc.main", command],
        cwd=ROOT,
        input=json.dumps(payload),
        check=False,
        capture_output=True,
        text=True,
        timeout=5,
    )


def _write_descendant_command(
    tmp_path: Path,
    *,
    name: str,
    output: str,
    exit_code: int = 0,
) -> tuple[Path, Path]:
    descendant_pid_path = tmp_path / f"{name}.pid"
    script_path = tmp_path / f"{name}.py"
    script_path.write_text(
        "import subprocess,sys\n"
        "child=subprocess.Popen("
        "[sys.executable,'-c','import time;time.sleep(30)'],"
        "stdin=subprocess.DEVNULL,stdout=subprocess.DEVNULL,"
        "stderr=subprocess.DEVNULL,close_fds=True)\n"
        f"open({str(descendant_pid_path)!r},'w').write(str(child.pid))\n"
        f"sys.stdout.write({output!r})\n"
        "sys.stdout.flush()\n"
        f"raise SystemExit({exit_code})\n",
        encoding="utf-8",
    )
    return script_path, descendant_pid_path


async def _assert_process_gone(pid: int) -> None:
    for _ in range(100):
        try:
            os.kill(pid, 0)
        except ProcessLookupError:
            return
        await asyncio.sleep(0.01)
    pytest.fail(f"process {pid} was not reaped")


@pytest.mark.asyncio
async def test_load_runtime_validates_artifact_and_round_trip_probe(tmp_path: Path) -> None:
    module = _load_service_module()
    env = _write_runtime_files(tmp_path)

    runtime = await module.load_runtime(env, command_timeout_seconds=1.0)

    assert runtime.dim == 129
    assert runtime.latent_dim == 2
    assert runtime.checksum.startswith("sha256:")
    assert runtime.validated is True


@pytest.mark.asyncio
async def test_bootstrap_validation_artifacts_writes_explicit_deterministic_manifest(
    tmp_path: Path,
) -> None:
    module = _load_service_module()
    first_paths = await module.bootstrap_validation_artifacts(tmp_path / "first")
    second_paths = await module.bootstrap_validation_artifacts(tmp_path / "second")

    first_manifest = json.loads(first_paths["manifest"].read_text(encoding="utf-8"))
    second_manifest = json.loads(second_paths["manifest"].read_text(encoding="utf-8"))
    assert first_manifest["validation_artifact"] == {
        "schema_version": "moleculeforge.validation_artifact.v1",
        "purpose": "synthetic_pipeline_validation_only",
        "seed": 7,
    }
    assert first_manifest["schema_version"] == "uas_training.v1"
    assert first_manifest["dim"] == 129
    assert first_manifest["autoencoder_path"] == "autoencoder.pt"
    assert first_manifest["autoencoder_sha256"] == (
        f"sha256:{hashlib.sha256(first_paths['checkpoint'].read_bytes()).hexdigest()}"
    )
    assert first_manifest == second_manifest
    first_state = torch.load(first_paths["checkpoint"], map_location="cpu", weights_only=True)
    second_state = torch.load(second_paths["checkpoint"], map_location="cpu", weights_only=True)
    assert first_state.keys() == second_state.keys()
    assert all(
        torch.equal(first_state[name], second_state[name])
        for name in first_state
    )


@pytest.mark.asyncio
async def test_bootstrap_validation_artifacts_is_idempotent_and_refuses_other_files(
    tmp_path: Path,
) -> None:
    module = _load_service_module()
    target = tmp_path / "validation"
    paths = await module.bootstrap_validation_artifacts(target)
    checkpoint_bytes = paths["checkpoint"].read_bytes()
    manifest_bytes = paths["manifest"].read_bytes()

    repeated = await module.bootstrap_validation_artifacts(target)

    assert repeated == paths
    assert paths["checkpoint"].read_bytes() == checkpoint_bytes
    assert paths["manifest"].read_bytes() == manifest_bytes

    production_target = tmp_path / "production"
    production_target.mkdir()
    production_checkpoint = production_target / "autoencoder.pt"
    production_checkpoint.write_bytes(b"production-checkpoint")
    production_manifest = production_target / "training_manifest.json"
    production_manifest.write_bytes(b'{"schema_version":"uas_training.v1"}')
    with pytest.raises(FileExistsError, match="refusing to overwrite"):
        await module.bootstrap_validation_artifacts(production_target)
    assert production_checkpoint.read_bytes() == b"production-checkpoint"
    assert production_manifest.read_bytes() == b'{"schema_version":"uas_training.v1"}'


@pytest.mark.asyncio
async def test_bootstrap_validation_artifacts_removes_partial_atomic_write(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    module = _load_service_module()
    target = tmp_path / "validation"

    def fail_after_partial_write(state: object, path: Path) -> None:
        del state
        path.write_bytes(b"partial")
        raise RuntimeError("injected checkpoint failure")

    monkeypatch.setattr(module.torch, "save", fail_after_partial_write)
    with pytest.raises(RuntimeError, match="injected checkpoint failure"):
        await module.bootstrap_validation_artifacts(target)

    assert not target.exists()
    assert list(tmp_path.glob(".validation.*")) == []


@pytest.mark.asyncio
async def test_bootstrap_validation_artifacts_does_not_publish_failed_runtime_probe(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    module = _load_service_module()
    target = tmp_path / "validation"

    async def fail_runtime_probe(directory: Path) -> None:
        del directory
        raise RuntimeError("injected runtime probe failure")

    monkeypatch.setattr(
        module,
        "_probe_validation_artifact_runtime",
        fail_runtime_probe,
        raising=False,
    )
    with pytest.raises(RuntimeError, match="injected runtime probe failure"):
        await module.bootstrap_validation_artifacts(target)

    assert not target.exists()
    assert list(tmp_path.glob(".validation.*")) == []


@pytest.mark.asyncio
async def test_load_runtime_requires_explicit_validation_artifact_opt_in(
    tmp_path: Path,
) -> None:
    module = _load_service_module()
    paths = await module.bootstrap_validation_artifacts(tmp_path / "validation")
    env = _write_runtime_files(tmp_path)
    env["UAS_AUTOENCODER_PATH"] = str(paths["checkpoint"])
    env["UAS_ARTIFACT_MANIFEST_PATH"] = str(paths["manifest"])

    with pytest.raises(RuntimeError, match="UAS_ALLOW_VALIDATION_ARTIFACT=true"):
        await module.load_runtime(env, command_timeout_seconds=1.0)

    env["UAS_ALLOW_VALIDATION_ARTIFACT"] = "true"
    runtime = await module.load_runtime(env, command_timeout_seconds=1.0)

    assert runtime.validated is True
    assert runtime.checkpoint_path == paths["checkpoint"]


@pytest.mark.asyncio
async def test_validation_artifacts_run_real_load_probe_and_exact_generate(
    tmp_path: Path,
) -> None:
    module = _load_service_module()
    paths = await module.bootstrap_validation_artifacts(tmp_path / "validation")
    command_prefix = [sys.executable, str(SERVICE_MAIN)]
    runtime = await module.load_runtime(
        {
            "UAS_AUTOENCODER_PATH": str(paths["checkpoint"]),
            "UAS_ARTIFACT_MANIFEST_PATH": str(paths["manifest"]),
            "UAS_CANDIDATE_SOURCE_COMMAND": shlex.join(
                [*command_prefix, "validation-candidate"]
            ),
            "UAS_DECODER_COMMAND": shlex.join([*command_prefix, "validation-decoder"]),
            "UAS_ALLOW_VALIDATION_ARTIFACT": "true",
        },
        command_timeout_seconds=5.0,
    )
    request = _valid_generate_request()
    request.batch_size = 8
    request.total_molecules = 8

    response = await module.UASGeneratorServicer(runtime).Generate(request, None)

    molecules = [json.loads(payload.decode("utf-8")) for payload in response.molecules]
    assert runtime.validated is True
    assert len(molecules) == 8
    assert len(response.humu_embeddings) == 8
    assert len({molecule["canonical_smiles"] for molecule in molecules}) == 8
    assert all(
        canonicalize(molecule["canonical_smiles"]) == molecule["canonical_smiles"]
        for molecule in molecules
    )


def test_validation_candidate_cli_returns_exact_deterministic_lorentz_batch() -> None:
    payload = _valid_candidate_command_payload(n_samples=4)

    first = _run_uas_cli("validation-candidate", payload)
    second = _run_uas_cli("validation-candidate", payload)

    assert first.returncode == 0, first.stderr
    assert second.returncode == 0, second.stderr
    first_response = json.loads(first.stdout)
    second_response = json.loads(second.stdout)
    assert first_response == second_response
    assert set(first_response) == {"schema_version", "embeddings"}
    assert first_response["schema_version"] == COMMAND_SCHEMA
    embeddings = first_response["embeddings"]
    assert len(embeddings) == 4
    assert all(len(row) == 129 for row in embeddings)
    assert all(all(math.isfinite(float(value)) for value in row) for row in embeddings)
    assert all(
        row[0] > 0.0
        and math.isclose(
            row[0] ** 2 - sum(value**2 for value in row[1:]),
            1.0,
            rel_tol=0.0,
            abs_tol=1e-12,
        )
        for row in embeddings
    )


def test_validation_candidate_cli_rejects_noncanonical_nested_protocol() -> None:
    payload = _valid_candidate_command_payload()
    hciv = payload["hciv"]
    assert isinstance(hciv, dict)
    hciv["unexpected"] = True

    result = _run_uas_cli("validation-candidate", payload)

    assert result.returncode != 0
    assert "validation candidate hciv has invalid fields" in result.stderr


def test_validation_decoder_cli_returns_exact_unique_canonical_smiles_batch() -> None:
    payload = _valid_decoder_command_payload(n_samples=8)

    first = _run_uas_cli("validation-decoder", payload)
    second = _run_uas_cli("validation-decoder", payload)

    assert first.returncode == 0, first.stderr
    assert second.returncode == 0, second.stderr
    first_response = json.loads(first.stdout)
    assert first_response == json.loads(second.stdout)
    assert set(first_response) == {"schema_version", "candidates"}
    assert first_response["schema_version"] == COMMAND_SCHEMA
    candidates = first_response["candidates"]
    assert len(candidates) == 8
    canonical_smiles = [candidate["canonical_smiles"] for candidate in candidates]
    assert len(set(canonical_smiles)) == 8
    assert all(
        candidate["smiles"] == canonical_smiles[index]
        for index, candidate in enumerate(candidates)
    )
    assert all(canonicalize(smiles) == smiles for smiles in canonical_smiles)


def test_validation_decoder_cli_rejects_wrong_context_schema() -> None:
    payload = _valid_decoder_command_payload()
    context = payload["context"]
    assert isinstance(context, dict)
    context["context_schema_version"] = "wrong"

    result = _run_uas_cli("validation-decoder", payload)

    assert result.returncode != 0
    assert "context_schema_version must be generator_context.v1" in result.stderr


def test_uas_service_cli_routes_validation_artifact_bootstrap(tmp_path: Path) -> None:
    target = tmp_path / "validation"

    result = subprocess.run(  # noqa: S603
        [
            sys.executable,
            "-m",
            "uas_generator_svc.main",
            "bootstrap-validation-artifacts",
            str(target),
        ],
        cwd=ROOT,
        check=False,
        capture_output=True,
        text=True,
        timeout=15,
    )

    assert result.returncode == 0, result.stderr
    assert json.loads(result.stdout) == {
        "checkpoint": str(target / "autoencoder.pt"),
        "manifest": str(target / "training_manifest.json"),
    }


@pytest.mark.asyncio
async def test_training_artifact_loads_in_runtime(tmp_path: Path) -> None:
    module = _load_service_module()
    env = _write_runtime_files(tmp_path)
    data_path = tmp_path / "embeddings.jsonl"
    data_path.write_text(
        "\n".join(
            [
                json.dumps({"id": "mol-1", "embedding": [0.0] * 129}),
                json.dumps({"id": "mol-2", "embedding": [0.1] * 129}),
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    result = subprocess.run(  # noqa: S603
        [
            sys.executable,
            str(ROOT / "models/mf-generators/uas/train.py"),
            "--data",
            str(data_path),
            "--output-dir",
            str(Path(env["UAS_ARTIFACT_MANIFEST_PATH"]).parent),
            "--epochs",
            "1",
            "--batch-size",
            "2",
            "--device",
            "cpu",
        ],
        cwd=ROOT,
        check=False,
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, result.stderr

    runtime = await module.load_runtime(env, command_timeout_seconds=1.0)

    assert runtime.validated is True
    assert runtime.dim == 129
    assert runtime.latent_dim == 64


@pytest.mark.parametrize(
    "missing_env",
    [
        "UAS_AUTOENCODER_PATH",
        "UAS_ARTIFACT_MANIFEST_PATH",
        "UAS_CANDIDATE_SOURCE_COMMAND",
        "UAS_DECODER_COMMAND",
    ],
)
@pytest.mark.asyncio
async def test_load_runtime_requires_each_production_env(
    tmp_path: Path,
    missing_env: str,
) -> None:
    module = _load_service_module()
    env = _write_runtime_files(tmp_path)
    env.pop(missing_env)

    with pytest.raises(RuntimeError, match=missing_env):
        await module.load_runtime(env, command_timeout_seconds=1.0)


@pytest.mark.asyncio
async def test_load_runtime_rejects_checkpoint_checksum_mismatch(tmp_path: Path) -> None:
    module = _load_service_module()
    env = _write_runtime_files(tmp_path)
    manifest_path = Path(env["UAS_ARTIFACT_MANIFEST_PATH"])
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["autoencoder_sha256"] = f"sha256:{'0' * 64}"
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")

    with pytest.raises(RuntimeError, match="checksum does not match"):
        await module.load_runtime(env, command_timeout_seconds=1.0)


@pytest.mark.asyncio
async def test_load_runtime_rejects_manifest_state_dimension_mismatch(
    tmp_path: Path,
) -> None:
    module = _load_service_module()
    env = _write_runtime_files(tmp_path)
    manifest_path = Path(env["UAS_ARTIFACT_MANIFEST_PATH"])
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["latent_dim"] = 3
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")

    with pytest.raises(RuntimeError, match="does not match manifest dimensions"):
        await module.load_runtime(env, command_timeout_seconds=1.0)


@pytest.mark.asyncio
async def test_load_runtime_rejects_nonfinite_checkpoint_tensor(tmp_path: Path) -> None:
    module = _load_service_module()
    env = _write_runtime_files(tmp_path)
    checkpoint_path = Path(env["UAS_AUTOENCODER_PATH"])
    state = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
    state["encoder.0.weight"][0, 0] = torch.nan
    torch.save(state, checkpoint_path)
    manifest_path = Path(env["UAS_ARTIFACT_MANIFEST_PATH"])
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["autoencoder_sha256"] = (
        f"sha256:{hashlib.sha256(checkpoint_path.read_bytes()).hexdigest()}"
    )
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")

    with pytest.raises(RuntimeError, match="tensor is not finite"):
        await module.load_runtime(env, command_timeout_seconds=1.0)


@pytest.mark.parametrize(
    ("field", "value", "message"),
    [
        ("schema_version", "uas_training.v0", "manifest schema"),
        ("dim", 128, "manifest dim must be 129"),
        ("latent_dim", 0, "manifest latent_dim must be a positive integer"),
        (
            "autoencoder_path",
            "/var/lib/moleculeforge/autoencoder.pt",
            "relative basename",
        ),
        ("validation_artifact", None, "validation artifact marker is invalid"),
    ],
)
@pytest.mark.asyncio
async def test_load_runtime_rejects_invalid_manifest_contract(
    tmp_path: Path,
    field: str,
    value: object,
    message: str,
) -> None:
    module = _load_service_module()
    env = _write_runtime_files(tmp_path)
    manifest_path = Path(env["UAS_ARTIFACT_MANIFEST_PATH"])
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest[field] = value
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")

    with pytest.raises(RuntimeError, match=message):
        await module.load_runtime(env, command_timeout_seconds=1.0)


@pytest.mark.parametrize(
    ("env_name", "value", "message"),
    [
        ("UAS_AUTOENCODER_PATH", "missing.pt", "existing file"),
        ("UAS_ARTIFACT_MANIFEST_PATH", "missing.json", "existing file"),
        (
            "UAS_CANDIDATE_SOURCE_COMMAND",
            "missing-uas-candidate",
            "executable not found",
        ),
        ("UAS_DECODER_COMMAND", "missing-uas-decoder", "executable not found"),
    ],
)
@pytest.mark.asyncio
async def test_load_runtime_rejects_unavailable_paths_and_commands(
    tmp_path: Path,
    env_name: str,
    value: str,
    message: str,
) -> None:
    module = _load_service_module()
    env = _write_runtime_files(tmp_path)
    env[env_name] = value

    with pytest.raises(RuntimeError, match=message):
        await module.load_runtime(env, command_timeout_seconds=1.0)


@pytest.mark.parametrize(
    ("command_env", "script_body", "message"),
    [
        (
            "UAS_CANDIDATE_SOURCE_COMMAND",
            "import json\nprint(json.dumps({'schema_version':'wrong','embeddings':[]}))\n",
            "candidate response schema",
        ),
        (
            "UAS_CANDIDATE_SOURCE_COMMAND",
            "import json\nprint(json.dumps({'schema_version':'uas.command.v1',"
            "'embeddings':[[0.0]*128]}))\n",
            "exactly 129",
        ),
        (
            "UAS_CANDIDATE_SOURCE_COMMAND",
            "import json\nprint(json.dumps({'schema_version':'uas.command.v1',"
            "'embeddings':[[1e39]*129]}))\n",
            "finite float32",
        ),
        (
            "UAS_DECODER_COMMAND",
            "import json\nprint(json.dumps({'schema_version':'uas.command.v1','candidates':[]}))\n",
            "count must match",
        ),
        (
            "UAS_DECODER_COMMAND",
            "import json\nprint(json.dumps({'schema_version':'uas.command.v1',"
            "'candidates':[{'smiles':'not a smiles'}]}))\n",
            "invalid SMILES",
        ),
        (
            "UAS_DECODER_COMMAND",
            "import json\nprint(json.dumps({'schema_version':'uas.command.v1',"
            "'candidates':[{'smiles':'CCO','unexpected':1}]}))\n",
            "candidate has invalid fields",
        ),
        (
            "UAS_DECODER_COMMAND",
            'print(\'{"schema_version":"uas.command.v1",'
            '"candidates":[{"smiles":"CCO","properties":{"score":NaN}}]}\')\n',
            "invalid JSON",
        ),
        (
            "UAS_DECODER_COMMAND",
            'print(\'{"schema_version":"uas.command.v1",'
            '"candidates":[{"smiles":"CCO","properties":{"score":1e400}}]}\')\n',
            "non-finite",
        ),
    ],
)
@pytest.mark.asyncio
async def test_load_runtime_rejects_invalid_probe_protocol(
    tmp_path: Path,
    command_env: str,
    script_body: str,
    message: str,
) -> None:
    module = _load_service_module()
    env = _write_runtime_files(tmp_path)
    script_path = Path(shlex.split(env[command_env])[-1])
    script_path.write_text(script_body, encoding="utf-8")

    with pytest.raises(RuntimeError, match=message):
        await module.load_runtime(env, command_timeout_seconds=1.0)


@pytest.mark.asyncio
async def test_decoder_canonicalizes_valid_smiles(tmp_path: Path) -> None:
    module = _load_service_module()
    env = _write_runtime_files(tmp_path)
    runtime = await module.load_runtime(env, command_timeout_seconds=1.0)
    decoder_path = Path(shlex.split(env["UAS_DECODER_COMMAND"])[-1])
    decoder_path.write_text(
        "import json,sys\n"
        "request=json.load(sys.stdin)\n"
        "print(json.dumps({'schema_version':'uas.command.v1',"
        "'candidates':[{'smiles':'OCC','canonical_smiles':'not a smiles'}"
        " for _ in request['embeddings']]}))\n",
        encoding="utf-8",
    )

    candidates = await module._decode_embeddings(
        runtime,
        operation="decode",
        embeddings=torch.zeros((1, 129), dtype=torch.float32),
        context=module._probe_context(),
    )

    assert candidates == [{"smiles": "OCC", "canonical_smiles": "CCO"}]


@pytest.mark.asyncio
async def test_candidate_timeout_terminates_and_reaps_process(tmp_path: Path) -> None:
    module = _load_service_module()
    env = _write_runtime_files(tmp_path)
    pid_path = tmp_path / "candidate.pid"
    candidate_path = Path(shlex.split(env["UAS_CANDIDATE_SOURCE_COMMAND"])[-1])
    candidate_path.write_text(
        f"import os,time\nopen({str(pid_path)!r},'w').write(str(os.getpid()))\ntime.sleep(10)\n",
        encoding="utf-8",
    )

    started = time.perf_counter()
    with pytest.raises(RuntimeError, match="timed out"):
        await module.load_runtime(env, command_timeout_seconds=0.05)

    assert time.perf_counter() - started < 2.0
    pid = int(pid_path.read_text(encoding="utf-8"))
    with pytest.raises(ProcessLookupError):
        os.kill(pid, 0)


@pytest.mark.asyncio
async def test_command_timeout_terminates_spawned_process_group(tmp_path: Path) -> None:
    module = _load_service_module()
    child_pid_path = tmp_path / "child.pid"
    script_path = tmp_path / "process_tree.py"
    script_path.write_text(
        "import subprocess,sys,time\n"
        "child=subprocess.Popen([sys.executable,'-c','import time;time.sleep(30)'])\n"
        f"open({str(child_pid_path)!r},'w').write(str(child.pid))\n"
        "time.sleep(30)\n",
        encoding="utf-8",
    )
    child_pid = None
    try:
        with pytest.raises(RuntimeError, match="timed out"):
            await module._run_json_command(
                (sys.executable, str(script_path)),
                {"schema_version": COMMAND_SCHEMA},
                timeout_seconds=0.2,
                name="test command",
            )
        child_pid = int(child_pid_path.read_text(encoding="utf-8"))
        await _assert_process_gone(child_pid)
    finally:
        if child_pid is not None:
            try:
                os.kill(child_pid, signal.SIGKILL)
            except ProcessLookupError:
                pass


@pytest.mark.parametrize(
    ("name", "output", "exit_code", "message"),
    [
        (
            "successful",
            '{"schema_version":"uas.command.v1"}',
            0,
            None,
        ),
        (
            "nonzero",
            "",
            7,
            "failed with exit code 7",
        ),
        (
            "invalid_json",
            "not-json",
            0,
            "returned invalid JSON",
        ),
    ],
)
@pytest.mark.asyncio
async def test_command_completion_reaps_spawned_descendants(
    tmp_path: Path,
    name: str,
    output: str,
    exit_code: int,
    message: str | None,
) -> None:
    module = _load_service_module()
    script_path, descendant_pid_path = _write_descendant_command(
        tmp_path,
        name=name,
        output=output,
        exit_code=exit_code,
    )
    descendant_pid = None
    try:
        command = module._run_json_command(
            (sys.executable, str(script_path)),
            {"schema_version": COMMAND_SCHEMA},
            timeout_seconds=1.0,
            name="test command",
        )
        if message is None:
            assert await command == {"schema_version": COMMAND_SCHEMA}
        else:
            with pytest.raises(RuntimeError, match=message):
                await command

        descendant_pid = int(descendant_pid_path.read_text(encoding="utf-8"))
        await _assert_process_gone(descendant_pid)
    finally:
        if descendant_pid is not None:
            try:
                os.kill(descendant_pid, signal.SIGKILL)
            except ProcessLookupError:
                pass


@pytest.mark.asyncio
async def test_invalid_candidate_protocol_reaps_spawned_descendants(
    tmp_path: Path,
) -> None:
    module = _load_service_module()
    script_path, descendant_pid_path = _write_descendant_command(
        tmp_path,
        name="invalid_protocol",
        output='{"schema_version":"wrong","embeddings":[]}',
    )
    runtime = module.UASRuntime(
        checkpoint_path=tmp_path / "autoencoder.pt",
        manifest_path=tmp_path / "training_manifest.json",
        checksum=f"sha256:{'0' * 64}",
        dim=129,
        latent_dim=2,
        model=MoleculeAutoencoder(input_dim=129, latent_dim=2),
        candidate_command=(sys.executable, str(script_path)),
        decoder_command=(sys.executable,),
        command_timeout_seconds=1.0,
        validated=True,
    )
    descendant_pid = None
    try:
        with pytest.raises(RuntimeError, match="candidate response schema"):
            await module._candidate_embeddings(
                runtime,
                operation="sample",
                n_samples=1,
                seed=7,
                attempt=1,
                context=module._probe_context(),
            )

        descendant_pid = int(descendant_pid_path.read_text(encoding="utf-8"))
        await _assert_process_gone(descendant_pid)
    finally:
        if descendant_pid is not None:
            try:
                os.kill(descendant_pid, signal.SIGKILL)
            except ProcessLookupError:
                pass


@pytest.mark.asyncio
async def test_command_cancellation_terminates_and_reaps_process(tmp_path: Path) -> None:
    module = _load_service_module()
    pid_path = tmp_path / "command.pid"
    child_pid_path = tmp_path / "command-child.pid"
    script_path = tmp_path / "blocking.py"
    script_path.write_text(
        "import os,subprocess,sys,time\n"
        "child=subprocess.Popen("
        "[sys.executable,'-c','import time;time.sleep(30)'],"
        "stdin=subprocess.DEVNULL,stdout=subprocess.DEVNULL,"
        "stderr=subprocess.DEVNULL,close_fds=True)\n"
        f"open({str(pid_path)!r},'w').write(str(os.getpid()))\n"
        f"open({str(child_pid_path)!r},'w').write(str(child.pid))\n"
        "time.sleep(10)\n",
        encoding="utf-8",
    )
    task = asyncio.create_task(
        module._run_json_command(
            (sys.executable, str(script_path)),
            {"schema_version": COMMAND_SCHEMA},
            timeout_seconds=10.0,
            name="test command",
        )
    )
    for _ in range(100):
        if pid_path.is_file() and child_pid_path.is_file():
            break
        await asyncio.sleep(0.01)
    assert pid_path.is_file()
    assert child_pid_path.is_file()

    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task

    pid = int(pid_path.read_text(encoding="utf-8"))
    with pytest.raises(ProcessLookupError):
        os.kill(pid, 0)
    await _assert_process_gone(int(child_pid_path.read_text(encoding="utf-8")))


@pytest.mark.asyncio
async def test_repeated_cancellation_waits_for_process_group_cleanup(
    tmp_path: Path,
) -> None:
    module = _load_service_module()
    parent_pid_path = tmp_path / "parent.pid"
    child_pid_path = tmp_path / "child.pid"
    child_ready_path = tmp_path / "child.ready"
    parent_term_path = tmp_path / "parent.term"
    child_term_path = tmp_path / "child.term"
    child_script = (
        "import pathlib,signal,time\n"
        f"term=pathlib.Path({str(child_term_path)!r})\n"
        "signal.signal(signal.SIGTERM,lambda *_:term.write_text('term'))\n"
        f"pathlib.Path({str(child_ready_path)!r}).write_text('ready')\n"
        "time.sleep(30)\n"
    )
    script_path = tmp_path / "ignore_term.py"
    script_path.write_text(
        "import os,pathlib,signal,subprocess,sys,time\n"
        f"term=pathlib.Path({str(parent_term_path)!r})\n"
        "signal.signal(signal.SIGTERM,lambda *_:term.write_text('term'))\n"
        f"child=subprocess.Popen([sys.executable,'-c',{child_script!r}],"
        "stdin=subprocess.DEVNULL,stdout=subprocess.DEVNULL,"
        "stderr=subprocess.DEVNULL,close_fds=True)\n"
        f"pathlib.Path({str(child_pid_path)!r}).write_text(str(child.pid))\n"
        f"ready=pathlib.Path({str(child_ready_path)!r})\n"
        "while not ready.is_file():time.sleep(0.005)\n"
        f"pathlib.Path({str(parent_pid_path)!r}).write_text(str(os.getpid()))\n"
        "time.sleep(30)\n",
        encoding="utf-8",
    )
    task = asyncio.create_task(
        module._run_json_command(
            (sys.executable, str(script_path)),
            {"schema_version": COMMAND_SCHEMA},
            timeout_seconds=10.0,
            name="test command",
        )
    )
    parent_pid = None
    child_pid = None
    try:
        for _ in range(100):
            if parent_pid_path.is_file() and child_pid_path.is_file():
                break
            await asyncio.sleep(0.01)
        assert parent_pid_path.is_file()
        assert child_pid_path.is_file()
        parent_pid = int(parent_pid_path.read_text(encoding="utf-8"))
        child_pid = int(child_pid_path.read_text(encoding="utf-8"))

        task.cancel()
        for _ in range(100):
            if parent_term_path.is_file() and child_term_path.is_file():
                break
            await asyncio.sleep(0.01)
        assert parent_term_path.is_file()
        assert child_term_path.is_file()

        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await asyncio.wait_for(task, timeout=3.0)

        await _assert_process_gone(parent_pid)
        await _assert_process_gone(child_pid)
    finally:
        if parent_pid is not None:
            try:
                os.killpg(parent_pid, signal.SIGKILL)
            except ProcessLookupError:
                pass
        if child_pid is not None:
            try:
                os.kill(child_pid, signal.SIGKILL)
            except ProcessLookupError:
                pass


@pytest.mark.asyncio
async def test_command_rejects_nonfinite_request_before_starting_process(
    tmp_path: Path,
) -> None:
    module = _load_service_module()
    marker = tmp_path / "started"
    script_path = tmp_path / "marker.py"
    script_path.write_text(
        f"from pathlib import Path\nPath({str(marker)!r}).write_text('started')\n",
        encoding="utf-8",
    )

    with pytest.raises(RuntimeError, match="request is not valid JSON"):
        await module._run_json_command(
            (sys.executable, str(script_path)),
            {"value": float("nan")},
            timeout_seconds=1.0,
            name="test command",
        )

    assert not marker.exists()


@pytest.mark.parametrize(
    ("invalid_field", "message"),
    [
        ("context_schema", "context_schema_version"),
        ("batch_size", "generator limit 256"),
        ("cig", "CIG has no objectives"),
        ("hciv", "hciv.coordinates"),
        ("intent_cone", "intent_cone.axis"),
    ],
)
def test_request_validation_rejects_invalid_typed_context(
    invalid_field: str,
    message: str,
) -> None:
    module = _load_service_module()
    request = _valid_generate_request()
    if invalid_field == "context_schema":
        request.context_schema_version = "wrong"
    elif invalid_field == "batch_size":
        request.batch_size = 257
    elif invalid_field == "cig":
        request.cig.ClearField("objectives")
    elif invalid_field == "hciv":
        del request.hciv.coordinates[:]
        request.hciv.coordinates.extend([1.0, *([0.0] * 127)])
    else:
        request.intent_cone = humu_pb2.IntentCone(
            axis=[1.0, *([0.0] * 127)],
            half_angle=0.5,
            curvature=1.0,
        ).SerializeToString()

    with pytest.raises(ValueError, match=message):
        module._validate_generate_request(request)


@pytest.mark.asyncio
async def test_grpc_service_returns_typed_uas_response(tmp_path: Path) -> None:
    module = _load_service_module()
    runtime = await module.load_runtime(
        _write_runtime_files(tmp_path),
        command_timeout_seconds=1.0,
    )
    server = grpc.aio.server()
    generator_pb2_grpc.add_GeneratorServiceServicer_to_server(
        module.UASGeneratorServicer(runtime),
        server,
    )
    port = server.add_insecure_port("127.0.0.1:0")
    await server.start()
    channel = grpc.aio.insecure_channel(f"127.0.0.1:{port}")
    stub = generator_pb2_grpc.GeneratorServiceStub(channel)
    try:
        info = await stub.Info(generator_pb2.GeneratorInfo(), timeout=2)
        response = await stub.Generate(
            _valid_generate_request(),
            timeout=2,
        )
    finally:
        await channel.close()
        await server.stop(None)

    assert info.name == "uas"
    assert info.runtime_status == audit_pb2.GENERATOR_RUNTIME_STATUS_READY
    assert info.status_message == "ready"
    assert info.artifacts[0].checksum == runtime.checksum
    assert response.generator_name == "uas"
    assert response.generation_id == "project-1"
    assert response.request_id == "request-1"
    assert response.molecule_payload_schema == "molecule.v1"
    assert response.embedding_payload_schema == "humu.float32.v1"
    assert len(response.molecules) == 2
    assert len(response.humu_embeddings) == 2
    assert all(len(embedding) == 516 for embedding in response.humu_embeddings)
    assert all(
        struct.unpack("<129f", embedding) == tuple([0.0] * 129)
        for embedding in response.humu_embeddings
    )
    assert [json.loads(molecule.decode("utf-8"))["smiles"] for molecule in response.molecules] == [
        "CCO",
        "CCO",
    ]
    assert response.artifacts[0].checksum == runtime.checksum


@pytest.mark.asyncio
async def test_start_server_binds_registered_generator_service(tmp_path: Path) -> None:
    module = _load_service_module()
    runtime = await module.load_runtime(
        _write_runtime_files(tmp_path),
        command_timeout_seconds=1.0,
    )

    server, port = await module.start_server(runtime, "127.0.0.1:0")
    channel = grpc.aio.insecure_channel(f"127.0.0.1:{port}")
    try:
        response = await generator_pb2_grpc.GeneratorServiceStub(channel).Info(
            generator_pb2.GeneratorInfo(),
            timeout=2,
        )
    finally:
        await channel.close()
        await server.stop(None)

    assert response.name == "uas"
    assert response.runtime_status == audit_pb2.GENERATOR_RUNTIME_STATUS_READY
