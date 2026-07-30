"""HUMU Encoder Service - gRPC server for molecular/route/pocket encoding."""

import argparse
import asyncio
import hashlib
import hmac
import json
import math
import os
import re
import struct
import sys
import tempfile
import time
from collections.abc import Mapping
from concurrent import futures
from pathlib import Path

import grpc
from mf_core.artifacts import (
    ArtifactRequirement,
    RequirementStatus,
    check_artifact,
    require_available,
)
from mf_core.proto_gen.moleculeforge.v1.humu import encoder_pb2, encoder_pb2_grpc
from mf_humu.encoders import (
    HUMU_CHECKPOINT_SCHEMA,
    HUMU_VALIDATION_ARTIFACT_PURPOSE,
    HUMU_VALIDATION_ARTIFACT_SCHEMA,
    HUMU_VALIDATION_ARTIFACT_SEED,
    HUMUEncoderWrapper,
    build_humu_model_config,
    validate_humu_model_config,
)

_REQUIREMENTS = (ArtifactRequirement("humu_checkpoint", "HUMU_CHECKPOINT_PATH"),)
_ALLOW_VALIDATION_ARTIFACT_ENV = "HUMU_ALLOW_VALIDATION_ARTIFACT"
_ESM2_CHECKPOINT_PATH_ENV = "HUMU_ESM2_CHECKPOINT_PATH"
_ESM2_CHECKPOINT_SHA256_ENV = "HUMU_ESM2_CHECKPOINT_SHA256"
_LEGACY_MODEL_CONFIG_PATH_ENV = "HUMU_LEGACY_MODEL_CONFIG_PATH"
_SHA256_PATTERN = re.compile(r"^[0-9a-f]{64}$")
_VALIDATION_ARTIFACT = {
    "schema_version": HUMU_VALIDATION_ARTIFACT_SCHEMA,
    "purpose": HUMU_VALIDATION_ARTIFACT_PURPOSE,
    "seed": HUMU_VALIDATION_ARTIFACT_SEED,
}
_ENCODER_STATE_KEYS = {
    "molecule": "encoder_mol",
    "pocket": "encoder_pocket",
    "route": "encoder_route",
}


def _require_runtime() -> list[RequirementStatus]:
    statuses = [check_artifact(requirement) for requirement in _REQUIREMENTS]
    require_available(statuses)
    return statuses


def runtime_status() -> list[dict]:
    return [check_artifact(requirement).to_dict() for requirement in _REQUIREMENTS]


async def _abort_unavailable(context):
    statuses = [check_artifact(requirement) for requirement in _REQUIREMENTS]
    try:
        require_available(statuses)
    except RuntimeError as exc:
        message = str(exc)
    else:
        message = "HUMU checkpoint loading and input routing are not configured"
    if context is not None and hasattr(context, "abort"):
        await context.abort(grpc.StatusCode.FAILED_PRECONDITION, message)
    raise RuntimeError(message)


class HUMUEncoderRouter:
    def __init__(
        self,
        checkpoint_path: str,
        device: str = "cpu",
        *,
        esm2_checkpoint_path: str | None = None,
        esm2_checkpoint_sha256: str | None = None,
        legacy_model_config_path: str | None = None,
        _allow_validation_artifact: bool | None = None,
    ):
        import torch

        self.checkpoint_path = checkpoint_path
        self.device = torch.device(device)
        checkpoint = torch.load(
            checkpoint_path,
            map_location=self.device,
            weights_only=True,
        )
        self.encoders = _encoders_from_checkpoint(
            checkpoint,
            self.device,
            allow_validation_artifact=(
                (
                    os.environ.get(
                        _ALLOW_VALIDATION_ARTIFACT_ENV,
                        "",
                    ).strip()
                    == "true"
                )
                if _allow_validation_artifact is None
                else _allow_validation_artifact
            ),
            esm2_checkpoint_path=esm2_checkpoint_path,
            esm2_checkpoint_sha256=esm2_checkpoint_sha256,
            legacy_model_config_path=legacy_model_config_path,
        )

    def encode(self, input_type: str, payload) -> list[float]:
        import torch

        encoder = self.encoders[input_type]
        with torch.no_grad():
            embedding = encoder(payload)
        return [float(value) for value in embedding.detach().cpu().reshape(-1).tolist()]

    def curvature(self, input_type: str) -> float:
        curvature = getattr(self.encoders[input_type].manifold, "k", 1.0)
        if hasattr(curvature, "detach"):
            return float(curvature.detach().cpu().reshape(-1)[0])
        return float(curvature)


def _build_raw_encoders(
    model_config: Mapping,
    device,
) -> dict[str, object]:
    from mf_encoders.humu_mol.encoder import HUMUMoleculeEncoder
    from mf_encoders.humu_pocket.encoder import HUMUPocketEncoder
    from mf_encoders.humu_route.encoder import HUMURouteEncoder

    config = validate_humu_model_config(model_config)
    dim = config["embedding_dim"]
    curvature = config["curvature"]
    learnable_curvature = config["learnable_curvature"]
    mol = config["encoders"]["mol"]
    pocket = config["encoders"]["pocket"]
    route = config["encoders"]["route"]
    return {
        "molecule": HUMUMoleculeEncoder(
            dim=dim,
            curvature=curvature,
            learnable_curvature=learnable_curvature,
            hidden_dim=mol["hidden_dim"],
            n_layers=mol["n_layers"],
            n_heads=mol["n_heads"],
            dropout=mol["dropout"],
            use_3d_geometry=mol["use_3d_geometry"],
        ).to(device),
        "pocket": HUMUPocketEncoder(
            dim=dim,
            curvature=curvature,
            learnable_curvature=learnable_curvature,
            hidden_dim=pocket["hidden_dim"],
            n_layers=pocket["n_layers"],
            n_heads=pocket["n_heads"],
            dropout=pocket["dropout"],
            radius_angstrom=pocket["radius_angstrom"],
            max_neighbors=pocket["max_neighbors"],
            use_3d_geometry=pocket["use_3d_geometry"],
            use_esm2=pocket["use_esm2"],
            esm2_checkpoint=pocket["esm2_checkpoint"],
            esm2_layer=pocket["esm2_layer"],
            esm2_dim=pocket["esm2_dim"],
            esm2_batch_tokens=pocket["esm2_batch_tokens"],
            esm2_max_sequence_length=pocket["esm2_max_sequence_length"],
            esm2_required_sources=pocket["esm2_required_sources"],
        ).to(device),
        "route": HUMURouteEncoder(
            dim=dim,
            curvature=curvature,
            learnable_curvature=learnable_curvature,
            hidden_dim=route["hidden_dim"],
            n_layers=route["n_layers"],
            n_heads=route["n_heads"],
            dropout=route["dropout"],
            use_tree_pooling=route["use_tree_pooling"],
        ).to(device),
    }


def _encoders_from_checkpoint(
    checkpoint,
    device,
    *,
    allow_validation_artifact: bool,
    esm2_checkpoint_path: str | None = None,
    esm2_checkpoint_sha256: str | None = None,
    legacy_model_config_path: str | None = None,
) -> dict[str, object]:
    if not isinstance(checkpoint, Mapping):
        raise ValueError("HUMU checkpoint must contain a mapping")
    artifact = checkpoint.get("artifact")
    if artifact is not None:
        if artifact != _VALIDATION_ARTIFACT:
            raise ValueError("HUMU validation artifact marker is invalid")
        if not allow_validation_artifact:
            raise RuntimeError(
                f"{_ALLOW_VALIDATION_ARTIFACT_ENV}=true is required "
                "for synthetic HUMU validation artifacts"
            )

    encoder_states = {}
    for name, state_key in _ENCODER_STATE_KEYS.items():
        state_dict = checkpoint.get(state_key)
        if not isinstance(state_dict, Mapping):
            raise ValueError(f"{state_key} must contain a state dictionary")
        if not state_dict:
            raise ValueError(f"{state_key} state dictionary is empty")
        encoder_states[name] = state_dict

    model_config = checkpoint.get("model_config")
    checkpoint_schema = checkpoint.get("checkpoint_schema_version")
    uses_implicit_legacy_default = False
    if model_config is None:
        if checkpoint_schema is not None or artifact is not None:
            raise ValueError("versioned HUMU checkpoint requires model_config")
        if legacy_model_config_path:
            config = _load_legacy_model_config(legacy_model_config_path)
        else:
            config = build_humu_model_config({})
            uses_implicit_legacy_default = True
        config = _resolve_esm2_checkpoint(
            config,
            checkpoint_path=esm2_checkpoint_path,
            expected_sha256=esm2_checkpoint_sha256,
        )
        raw_encoders = _build_raw_encoders(config, device)
        encoders = {
            name: (
                HUMUEncoderWrapper(
                    encoder,
                    config["embedding_dim"],
                    device,
                    config["curvature"],
                    model_config=config,
                )
                if any(
                    isinstance(key, str)
                    and key.startswith(("inner.", "proj."))
                    for key in encoder_states[name]
                )
                else encoder
            )
            for name, encoder in raw_encoders.items()
        }
    else:
        if checkpoint_schema != HUMU_CHECKPOINT_SCHEMA:
            raise ValueError("HUMU checkpoint_schema_version is unsupported")
        config = validate_humu_model_config(model_config)
        config = _resolve_esm2_checkpoint(
            config,
            checkpoint_path=esm2_checkpoint_path,
            expected_sha256=esm2_checkpoint_sha256,
        )
        raw_encoders = _build_raw_encoders(config, device)
        encoders = {
            name: HUMUEncoderWrapper(
                encoder,
                config["embedding_dim"],
                device,
                config["curvature"],
                model_config=config,
            )
            for name, encoder in raw_encoders.items()
        }

    for name, state_key in _ENCODER_STATE_KEYS.items():
        try:
            encoders[name].load_state_dict(encoder_states[name], strict=True)
        except (RuntimeError, TypeError) as exc:
            if uses_implicit_legacy_default:
                raise ValueError(
                    f"{state_key} weights are incompatible with the default legacy "
                    f"architecture; {_LEGACY_MODEL_CONFIG_PATH_ENV} is required "
                    f"for non-default legacy checkpoints: {exc}"
                ) from exc
            raise ValueError(
                f"{state_key} weights are incompatible with model_config: {exc}"
            ) from exc
        encoders[name].eval()
    return encoders


def _load_legacy_model_config(path: str) -> dict:
    config_path = Path(path).expanduser()
    if not config_path.is_file():
        raise FileNotFoundError(f"legacy HUMU model config not found: {config_path}")
    try:
        with config_path.open(encoding="utf-8") as config_file:
            model_config = json.load(config_file)
    except json.JSONDecodeError as exc:
        raise ValueError(f"legacy HUMU model config is invalid JSON: {config_path}") from exc
    return validate_humu_model_config(model_config)


def _resolve_esm2_checkpoint(
    model_config: Mapping,
    *,
    checkpoint_path: str | None,
    expected_sha256: str | None,
) -> dict:
    config = validate_humu_model_config(model_config)
    pocket_config = config["encoders"]["pocket"]
    if not pocket_config["use_esm2"]:
        return config

    configured_path = (checkpoint_path or "").strip()
    if not configured_path:
        raise RuntimeError(
            f"{_ESM2_CHECKPOINT_PATH_ENV} is required when HUMU pocket ESM-2 is enabled"
        )
    resolved_path = Path(configured_path).expanduser()
    if not resolved_path.is_file():
        raise FileNotFoundError(f"ESM-2 checkpoint not found: {resolved_path}")

    checksum = (expected_sha256 or "").strip()
    if checksum:
        if not _SHA256_PATTERN.fullmatch(checksum):
            raise ValueError(
                f"{_ESM2_CHECKPOINT_SHA256_ENV} must contain 64 lowercase "
                "hexadecimal characters"
            )
        digest = hashlib.sha256()
        with resolved_path.open("rb") as checkpoint_file:
            for block in iter(lambda: checkpoint_file.read(1024 * 1024), b""):
                digest.update(block)
        if not hmac.compare_digest(digest.hexdigest(), checksum):
            raise RuntimeError("ESM-2 checkpoint SHA-256 does not match configured value")

    pocket_config["esm2_checkpoint"] = str(resolved_path)
    return config


def bootstrap_validation_checkpoint(
    checkpoint_path: str | Path,
) -> Path:
    """Create a deterministic synthetic checkpoint without replacing any file."""
    import torch

    target = Path(checkpoint_path)
    if target.exists() or target.is_symlink():
        _validate_existing_validation_checkpoint(target)
        return target
    target.parent.mkdir(parents=True, exist_ok=True)

    model_config = build_humu_model_config({})
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(HUMU_VALIDATION_ARTIFACT_SEED)
        raw_encoders = _build_raw_encoders(model_config, torch.device("cpu"))
        encoders = {
            name: HUMUEncoderWrapper(
                encoder,
                model_config["embedding_dim"],
                torch.device("cpu"),
                model_config["curvature"],
                model_config=model_config,
            )
            for name, encoder in raw_encoders.items()
        }
        checkpoint = {
            "checkpoint_schema_version": HUMU_CHECKPOINT_SCHEMA,
            "model_config": model_config,
            "artifact": dict(_VALIDATION_ARTIFACT),
            **{
                _ENCODER_STATE_KEYS[name]: encoder.state_dict()
                for name, encoder in encoders.items()
            },
        }

    file_descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{target.name}.",
        suffix=".tmp",
        dir=target.parent,
    )
    os.close(file_descriptor)
    temporary_path = Path(temporary_name)
    try:
        torch.save(checkpoint, temporary_path)
        with temporary_path.open("rb") as checkpoint_file:
            os.fsync(checkpoint_file.fileno())
        _probe_validation_checkpoint(temporary_path)
        try:
            os.link(temporary_path, target)
        except FileExistsError:
            _validate_existing_validation_checkpoint(target)
        else:
            _fsync_directory(target.parent)
    finally:
        temporary_path.unlink(missing_ok=True)
    return target


def _probe_validation_checkpoint(path: Path) -> None:
    router = HUMUEncoderRouter(
        str(path),
        device="cpu",
        _allow_validation_artifact=True,
    )
    payloads = {
        "molecule": "CCO",
        "pocket": {
            "coords": [[0.0, 0.0, 0.0]],
            "elements": ["C"],
            "residue_types": ["ALA"],
            "protein_sequence": "A",
        },
        "route": {
            "reactions": ["CCO>>CC=O"],
            "steps": 1,
            "intermediates": [],
            "score": 0.0,
        },
    }
    for input_type, payload in payloads.items():
        embedding = router.encode(input_type, payload)
        if len(embedding) != 129:
            raise RuntimeError(f"{input_type} validation embedding must contain 129 values")
        if not all(math.isfinite(value) for value in embedding):
            raise RuntimeError(f"{input_type} validation embedding must contain finite values")


def _fsync_directory(path: Path) -> None:
    flags = os.O_RDONLY | getattr(os, "O_DIRECTORY", 0)
    directory_descriptor = os.open(path, flags)
    try:
        os.fsync(directory_descriptor)
    finally:
        os.close(directory_descriptor)


def _validate_existing_validation_checkpoint(path: Path) -> None:
    import torch

    try:
        checkpoint = torch.load(
            path,
            map_location="cpu",
            weights_only=True,
        )
        if (
            not isinstance(checkpoint, Mapping)
            or checkpoint.get("artifact") != _VALIDATION_ARTIFACT
        ):
            raise ValueError("existing checkpoint is not a validation artifact")
        _encoders_from_checkpoint(
            checkpoint,
            torch.device("cpu"),
            allow_validation_artifact=True,
        )
        _probe_validation_checkpoint(path)
    except Exception as exc:
        raise FileExistsError(f"refusing to overwrite existing HUMU checkpoint: {path}") from exc


def _build_router() -> HUMUEncoderRouter:
    checkpoint_path = os.environ.get("HUMU_CHECKPOINT_PATH")
    if not checkpoint_path:
        raise RuntimeError("HUMU_CHECKPOINT_PATH is required")
    return HUMUEncoderRouter(
        checkpoint_path=checkpoint_path,
        device=os.environ.get("HUMU_DEVICE", "cpu"),
        esm2_checkpoint_path=os.environ.get(_ESM2_CHECKPOINT_PATH_ENV),
        esm2_checkpoint_sha256=os.environ.get(_ESM2_CHECKPOINT_SHA256_ENV),
        legacy_model_config_path=os.environ.get(_LEGACY_MODEL_CONFIG_PATH_ENV),
    )


def _normalise_input_type(value: str | None) -> str:
    key = (value or "").strip().lower()
    aliases = {
        "mol": "molecule",
        "molecule": "molecule",
        "smiles": "molecule",
        "pocket": "pocket",
        "route": "route",
    }
    if key not in aliases:
        raise ValueError("input_type must be one of molecule, pocket, route")
    return aliases[key]


def _payload_from_request(request, input_type: str):
    input_data_payload = _input_data_payload(request)
    if input_type == "molecule":
        smiles = getattr(request, "smiles", None)
        if not smiles:
            payload = getattr(request, "payload", None)
            smiles = payload.get("smiles") if isinstance(payload, dict) else None
        if not smiles and isinstance(input_data_payload, dict):
            smiles = input_data_payload.get("smiles")
        if not smiles and isinstance(input_data_payload, str):
            smiles = input_data_payload
        if not smiles:
            raise ValueError("molecule encoding requires request.smiles")
        if isinstance(input_data_payload, dict):
            return {**input_data_payload, "smiles": str(smiles)}
        return str(smiles)
    if input_type == "pocket":
        payload = getattr(request, "pocket_data", None) or getattr(request, "payload", None)
        if payload is None:
            payload = input_data_payload
        if not isinstance(payload, dict):
            raise ValueError("pocket encoding requires request.pocket_data")
        return payload
    payload = getattr(request, "route_data", None) or getattr(request, "payload", None)
    if payload is None:
        payload = input_data_payload
    if not isinstance(payload, dict):
        reactions = getattr(request, "reactions", None)
        payload = {"reactions": list(reactions)} if reactions is not None else None
    if not isinstance(payload, dict):
        raise ValueError("route encoding requires request.route_data")
    return payload


def _input_data_payload(request):
    input_data = getattr(request, "input_data", b"")
    if not input_data:
        return None
    if isinstance(input_data, str):
        text = input_data
    else:
        text = bytes(input_data).decode("utf-8")
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        return text


def _is_proto_encode_request(request) -> bool:
    return isinstance(request, encoder_pb2.EncodeRequest)


def _encode_response(
    input_type: str,
    checkpoint_path: str,
    embedding: list[float],
    curvature: float,
    proto_response: bool,
):
    if proto_response:
        return encoder_pb2.EncodeResponse(
            humu_embedding=struct.pack(f"<{len(embedding)}f", *embedding),
            curvature=curvature,
            elapsed_ms=0,
        )
    return type(
        "EncodeResponse",
        (),
        {
            "input_type": input_type,
            "checkpoint_path": checkpoint_path,
            "embedding": embedding,
            "embedding_dim": len(embedding),
        },
    )()


class HUMUEncoderServicer:
    def __init__(self, router: HUMUEncoderRouter | None = None):
        self.router = router

    def _router(self) -> HUMUEncoderRouter:
        if self.router is None:
            self.router = _build_router()
        return self.router

    async def Encode(self, request, context):
        try:
            _require_runtime()
        except RuntimeError:
            return await _abort_unavailable(context)
        router = self._router()
        input_type = _normalise_input_type(
            getattr(request, "input_type", None) or getattr(request, "entity_type", None)
        )
        payload = _payload_from_request(request, input_type)
        embedding = router.encode(input_type, payload)
        return _encode_response(
            input_type,
            router.checkpoint_path,
            embedding,
            router.curvature(input_type),
            proto_response=_is_proto_encode_request(request),
        )

    async def BatchEncode(self, request, context):
        start_time = time.perf_counter()
        responses = []
        for req in request.requests:
            responses.append(await self.Encode(req, context))
        elapsed_ms = int((time.perf_counter() - start_time) * 1000)
        if isinstance(request, encoder_pb2.BatchEncodeRequest):
            return encoder_pb2.BatchEncodeResponse(
                responses=responses,
                batch_id=request.batch_id,
                total_elapsed_ms=elapsed_ms,
            )
        return type(
            "Response",
            (),
            {
                "responses": responses,
                "batch_id": request.batch_id,
                "total_elapsed_ms": elapsed_ms,
            },
        )()


async def serve():
    _require_runtime()
    router = _build_router()
    server = grpc.aio.server(futures.ThreadPoolExecutor(max_workers=10))
    encoder_pb2_grpc.add_HUMUEncoderServiceServicer_to_server(
        HUMUEncoderServicer(router),
        server,
    )
    server.add_insecure_port("[::]:50051")
    await server.start()
    print("HUMU Encoder Service running on :50051")
    await server.wait_for_termination()


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(prog="humu_encoder_svc.main")
    parser.add_argument(
        "--bootstrap-validation-checkpoint",
        action="store_true",
    )
    args = parser.parse_args(argv)
    if args.bootstrap_validation_checkpoint:
        checkpoint_path = os.environ.get("HUMU_CHECKPOINT_PATH", "").strip()
        if not checkpoint_path:
            raise RuntimeError(
                "HUMU_CHECKPOINT_PATH is required to bootstrap a validation checkpoint"
            )
        path = bootstrap_validation_checkpoint(checkpoint_path)
        sys.stdout.write(f"HUMU validation checkpoint is available at {path}\n")
        return
    asyncio.run(serve())


if __name__ == "__main__":
    main()
