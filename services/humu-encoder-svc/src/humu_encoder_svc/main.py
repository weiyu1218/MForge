"""HUMU Encoder Service - gRPC server for molecular/route/pocket encoding."""
import asyncio
import os
import time
from concurrent import futures

import grpc
from mf_core.artifacts import (
    ArtifactRequirement,
    RequirementStatus,
    check_artifact,
    require_available,
)
from mf_core.proto_gen.moleculeforge.v1.humu import encoder_pb2_grpc

_REQUIREMENTS = (ArtifactRequirement("humu_checkpoint", "HUMU_CHECKPOINT_PATH"),)


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
    def __init__(self, checkpoint_path: str, device: str = "cpu"):
        import torch
        from mf_encoders.humu_mol.encoder import HUMUMoleculeEncoder
        from mf_encoders.humu_pocket.encoder import HUMUPocketEncoder
        from mf_encoders.humu_route.encoder import HUMURouteEncoder

        self.checkpoint_path = checkpoint_path
        self.device = torch.device(device)
        self.encoders = {
            "molecule": HUMUMoleculeEncoder(dim=128).to(self.device),
            "pocket": HUMUPocketEncoder(dim=128).to(self.device),
            "route": HUMURouteEncoder(dim=128).to(self.device),
        }
        checkpoint = torch.load(
            checkpoint_path,
            map_location=self.device,
            weights_only=True,
        )
        self.encoders["molecule"].load_state_dict(checkpoint["encoder_mol"], strict=False)
        self.encoders["pocket"].load_state_dict(checkpoint["encoder_pocket"], strict=False)
        self.encoders["route"].load_state_dict(checkpoint["encoder_route"], strict=False)
        for encoder in self.encoders.values():
            encoder.eval()

    def encode(self, input_type: str, payload) -> list[float]:
        import torch

        encoder = self.encoders[input_type]
        with torch.no_grad():
            embedding = encoder(payload)
        return [float(value) for value in embedding.detach().cpu().reshape(-1).tolist()]


def _build_router() -> HUMUEncoderRouter:
    checkpoint_path = os.environ.get("HUMU_CHECKPOINT_PATH")
    if not checkpoint_path:
        raise RuntimeError("HUMU_CHECKPOINT_PATH is required")
    return HUMUEncoderRouter(
        checkpoint_path=checkpoint_path,
        device=os.environ.get("HUMU_DEVICE", "cpu"),
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
    if input_type == "molecule":
        smiles = getattr(request, "smiles", None)
        if not smiles:
            payload = getattr(request, "payload", None)
            smiles = payload.get("smiles") if isinstance(payload, dict) else None
        if not smiles:
            raise ValueError("molecule encoding requires request.smiles")
        return str(smiles)
    if input_type == "pocket":
        payload = (
            getattr(request, "pocket_data", None)
            or getattr(request, "payload", None)
        )
        if not isinstance(payload, dict):
            raise ValueError("pocket encoding requires request.pocket_data")
        return payload
    payload = getattr(request, "route_data", None) or getattr(request, "payload", None)
    if not isinstance(payload, dict):
        reactions = getattr(request, "reactions", None)
        payload = {"reactions": list(reactions)} if reactions is not None else None
    if not isinstance(payload, dict):
        raise ValueError("route encoding requires request.route_data")
    return payload


def _encode_response(input_type: str, checkpoint_path: str, embedding: list[float]):
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
        input_type = _normalise_input_type(getattr(request, "input_type", None))
        payload = _payload_from_request(request, input_type)
        embedding = router.encode(input_type, payload)
        return _encode_response(input_type, router.checkpoint_path, embedding)

    async def BatchEncode(self, request, context):
        start_time = time.perf_counter()
        responses = []
        for req in request.requests:
            responses.append(await self.Encode(req, context))
        elapsed_ms = int((time.perf_counter() - start_time) * 1000)
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
    server = grpc.aio.server(futures.ThreadPoolExecutor(max_workers=10))
    encoder_pb2_grpc.add_HUMUEncoderServiceServicer_to_server(HUMUEncoderServicer(), server)
    server.add_insecure_port("[::]:50051")
    await server.start()
    print("HUMU Encoder Service running on :50051")
    await server.wait_for_termination()


if __name__ == "__main__":
    asyncio.run(serve())
