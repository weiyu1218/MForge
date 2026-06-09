#!/usr/bin/env python3
from __future__ import annotations

import contextlib
import json
import os
import re
import sys
import time
from pathlib import Path


def main() -> int:
    try:
        response = _run(_read_request())
    except Exception as exc:
        print(str(exc), file=sys.stderr)
        return 1
    json.dump(response, sys.stdout, sort_keys=True)
    sys.stdout.write("\n")
    return 0


def _read_request() -> dict[str, object]:
    try:
        payload = json.load(sys.stdin)
    except json.JSONDecodeError as exc:
        raise RuntimeError("RSGPT planner wrapper requires JSON stdin") from exc
    if not isinstance(payload, dict):
        raise RuntimeError("RSGPT planner wrapper request must be a JSON object")
    return payload


def _run(payload: dict[str, object]) -> dict[str, object]:
    smiles = str(payload.get("smiles") or "")
    if not smiles:
        raise RuntimeError("RSGPT planner wrapper requires smiles")
    max_routes = int(payload.get("max_routes") or 10)
    if max_routes <= 0:
        raise RuntimeError("RSGPT planner wrapper requires max_routes > 0")
    engine = str(payload.get("engine") or "rsgpt").strip().lower()
    if engine != "rsgpt":
        raise RuntimeError(f"Unsupported RSGPT planner engine: {engine}")

    start = time.perf_counter()
    routes = _find_routes(smiles, max_routes=max_routes)
    elapsed_ms = int((time.perf_counter() - start) * 1000)
    normalized = _validated_routes(routes)
    return {
        "routes": normalized,
        "total_routes_found": len(normalized),
        "elapsed_ms": elapsed_ms,
    }


def _find_routes(smiles: str, max_routes: int) -> list[dict[str, object]]:
    paths = _rsgpt_paths_from_env()
    beam_size = _positive_int(
        os.environ.get("RSGPT_BEAM_SIZE", "").strip() or str(max_routes),
        "RSGPT_BEAM_SIZE",
    )
    max_length = _positive_int(
        os.environ.get("RSGPT_MAX_LENGTH", "").strip() or "100",
        "RSGPT_MAX_LENGTH",
    )
    device = _rsgpt_device()
    use_half = _bool_env(
        "RSGPT_HALF_PRECISION",
        default=str(device).startswith("cuda"),
    )
    if _config_path_is_llama_json(paths.config_path):
        return _find_routes_with_llama(
            paths,
            smiles=smiles,
            max_routes=max_routes,
            beam_size=beam_size,
            max_length=max_length,
            device=device,
            use_half=use_half,
        )
    old_cwd = Path.cwd()
    sys.path.insert(0, str(paths.source_dir))
    try:
        os.chdir(paths.source_dir)
        with contextlib.redirect_stdout(sys.stderr):
            from infer import beam_search_gpt, jiexi
            from models.rxngpt import RxnGPT
            from rdkit import Chem
            from tokenizer.tokenization import SMILESBPETokenizer
            from transformers.modeling_utils import no_init_weights

            mol = Chem.MolFromSmiles(smiles)
            if mol is None:
                raise RuntimeError(f"RSGPT planner received invalid SMILES: {smiles}")
            std_smiles = Chem.MolToSmiles(mol)
            tokenizer = SMILESBPETokenizer.get_hf_tokenizer(
                str(paths.tokenizer_path),
                model_max_length=max_length,
            )
            cfg = _rsgpt_config(paths.config_path, max_length=max_length)
            with no_init_weights():
                model = RxnGPT(cfg, Tokenizer=1)
            model.load_state_dict(_load_state_dict(paths.model_path, device))
            if use_half:
                model.half()
            model.to(device)
            model.eval()
            decoded = beam_search_gpt(
                model,
                tokenizer,
                std_smiles,
                beam_size=beam_size,
                max_length=max_length,
                device=device,
            )
            reaction_smiles = jiexi(decoded)
    finally:
        os.chdir(old_cwd)
        try:
            sys.path.remove(str(paths.source_dir))
        except ValueError:
            pass
    return _routes_from_reactions(
        reaction_smiles,
        smiles=smiles,
        max_routes=max_routes,
    )


def _find_routes_with_llama(
    paths: _RSGPTPaths,
    *,
    smiles: str,
    max_routes: int,
    beam_size: int,
    max_length: int,
    device: str,
    use_half: bool,
) -> list[dict[str, object]]:
    from rdkit import Chem

    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        raise RuntimeError(f"RSGPT planner received invalid SMILES: {smiles}")
    std_smiles = Chem.MolToSmiles(mol)
    tokenizer = _load_hf_tokenizer(paths.tokenizer_path, max_length=max_length)
    model = _load_llama_model(
        paths.config_path,
        paths.model_path,
        device=device,
        use_half=use_half,
    )
    _validate_tokenizer_ids(tokenizer, std_smiles, model.config.vocab_size)
    decoded = _beam_search_gpt(
        model,
        tokenizer,
        std_smiles,
        beam_size=beam_size,
        max_length=max_length,
        device=device,
    )
    return _routes_from_reactions(
        _parse_decoded_reactions(decoded),
        smiles=smiles,
        max_routes=max_routes,
    )


def _load_hf_tokenizer(tokenizer_path: Path, *, max_length: int) -> object:
    from transformers import PreTrainedTokenizerFast

    tokenizer = PreTrainedTokenizerFast(tokenizer_file=str(tokenizer_path))
    tokenizer.add_special_tokens(
        {
            "pad_token": "<pad>",
            "bos_token": "<s>",
            "eos_token": "</s>",
            "unk_token": "<unk>",
        }
    )
    tokenizer.model_max_length = max_length
    return tokenizer


def _load_llama_model(
    config_path: Path,
    model_path: Path,
    *,
    device: str,
    use_half: bool,
) -> object:
    from transformers import LlamaConfig, LlamaForCausalLM
    from transformers.modeling_utils import no_init_weights

    config = LlamaConfig.from_pretrained(str(config_path))
    with no_init_weights():
        model = LlamaForCausalLM(config)
    model.load_state_dict(_load_llama_state_dict(model_path))
    if use_half:
        model.half()
    model.to(device)
    model.eval()
    return model


def _load_llama_state_dict(model_path: Path) -> dict[str, object]:
    loaded = _torch_load(model_path, map_location="cpu")
    if isinstance(loaded, dict):
        if isinstance(loaded.get("state_dict"), dict):
            loaded = loaded["state_dict"]
        elif isinstance(loaded.get("model"), dict):
            loaded = loaded["model"]
    if not isinstance(loaded, dict):
        raise RuntimeError("RSGPT model checkpoint must contain a state dict")
    direct_state = {}
    for key, value in loaded.items():
        key_str = str(key)
        if key_str.startswith("module.model."):
            direct_state[key_str.removeprefix("module.model.")] = value
        elif key_str.startswith("model."):
            direct_state[key_str.removeprefix("model.")] = value
    if direct_state:
        return direct_state
    return _strip_module_prefix(loaded)


def _torch_load(model_path: Path, *, map_location: str) -> object:
    import torch

    try:
        return torch.load(
            str(model_path),
            map_location=map_location,
            weights_only=True,
            mmap=True,
        )
    except TypeError:
        return torch.load(str(model_path), map_location=map_location)


def _validate_tokenizer_ids(tokenizer: object, std_smiles: str, vocab_size: int) -> None:
    prefix_ids = tokenizer.encode(
        f"<s><Isyn><O>{std_smiles}<F1>",
        add_special_tokens=False,
    )
    eos_ids = tokenizer.encode("</s>", add_special_tokens=False)
    invalid_ids = [token_id for token_id in [*prefix_ids, *eos_ids] if token_id >= vocab_size]
    if invalid_ids:
        raise RuntimeError(
            "RSGPT tokenizer ids exceed model vocab size "
            f"{vocab_size}: {invalid_ids}"
        )


def _beam_search_gpt(
    model: object,
    tokenizer: object,
    std_smiles: str,
    *,
    beam_size: int,
    max_length: int,
    device: str,
) -> list[str]:
    import torch
    import torch.nn.functional as torch_functional

    prefix = f"<s><Isyn><O>{std_smiles}<F1>"
    input_ids = tokenizer.encode(prefix, add_special_tokens=False)
    input_ids = torch.tensor(input_ids).unsqueeze(0).to(device)
    sequences = [(input_ids, 0.0)]
    end_token_id = tokenizer.encode("</s>", add_special_tokens=False)[0]
    completed_sequences = []
    for _ in range(max_length):
        all_candidates = []
        active_sequences = []
        active_scores = []
        for seq, score in sequences:
            if seq[0, -1].item() == end_token_id:
                completed_sequences.append((seq, score))
            else:
                active_sequences.append(seq)
                active_scores.append(score)
        if not active_sequences:
            break
        batch_input = torch.cat(active_sequences, dim=0)
        logits = model(input_ids=batch_input).logits[:, -1, :]
        logits = torch_functional.log_softmax(logits, dim=-1)
        for index, seq in enumerate(active_sequences):
            seq_logits = logits[index]
            topk_probs, topk_indices = torch.topk(seq_logits, beam_size + 10, dim=-1)
            for candidate_index in range(beam_size + 10):
                candidate_seq = torch.cat(
                    [seq, topk_indices[candidate_index].unsqueeze(0).unsqueeze(0)],
                    dim=1,
                )
                candidate_score = (
                    active_scores[index] - topk_probs[candidate_index].item()
                )
                all_candidates.append((candidate_seq, candidate_score))
        sequences = sorted(all_candidates, key=lambda item: item[1])[: beam_size + 10]
    completed_sequences.extend(sequences)
    completed_sequences = sorted(completed_sequences, key=lambda item: item[1])
    decoded_sequences = [
        tokenizer.decode(seq[0].squeeze().tolist()) for seq, _score in completed_sequences
    ]
    return _deduplicate(decoded_sequences)[:beam_size]


def _parse_decoded_reactions(input_texts: list[str]) -> list[str]:
    results = []
    for text in input_texts:
        cleaned = f"<F1>{text.split('<F1>')[-1]}"[:-4]
        product = f"<F1>{text.split('<F1>')[0]}".split("<O>")[-1]
        reactants = re.findall(r"<F\d+>(.*?)(?=<F|$)", cleaned)
        if reactants:
            results.append(f"{'.'.join(reactants)}>>{product}")
    return results


def _deduplicate(items: list[str]) -> list[str]:
    seen = set()
    result = []
    for item in items:
        if item in seen:
            continue
        seen.add(item)
        result.append(item)
    return result


def _rsgpt_config(config_path: Path, *, max_length: int) -> object:
    from omegaconf import OmegaConf

    loaded_config = OmegaConf.load(str(config_path))
    if OmegaConf.select(loaded_config, "MODEL.GPT_MODEL") is not None:
        from utils.utils import args_parse

        return args_parse(str(config_path))
    if _is_llama_config(loaded_config):
        return OmegaConf.create(
            {
                "MODEL": {
                    "GPT_MODEL": {
                        "config_path": str(config_path),
                    }
                },
                "DATA": {
                    "MAX_ATOM_NUM": max_length,
                },
            }
        )
    from utils.utils import args_parse

    return args_parse(str(config_path))


def _config_path_is_llama_json(config_path: Path) -> bool:
    from omegaconf import OmegaConf

    return _is_llama_config(OmegaConf.load(str(config_path)))


def _is_llama_config(config: object) -> bool:
    from omegaconf import OmegaConf

    return (
        OmegaConf.select(config, "model_type") == "llama"
        and OmegaConf.select(config, "hidden_size") is not None
        and OmegaConf.select(config, "num_hidden_layers") is not None
        and OmegaConf.select(config, "num_attention_heads") is not None
    )


class _RSGPTPaths:
    def __init__(
        self,
        *,
        source_dir: Path,
        model_path: Path,
        config_path: Path,
        tokenizer_path: Path,
    ) -> None:
        self.source_dir = source_dir
        self.model_path = model_path
        self.config_path = config_path
        self.tokenizer_path = tokenizer_path


def _rsgpt_paths_from_env() -> _RSGPTPaths:
    source_dir = _required_dir("RSGPT_SOURCE_DIR")
    return _RSGPTPaths(
        source_dir=source_dir,
        model_path=_required_file("RSGPT_MODEL_PATH"),
        config_path=_required_file("RSGPT_CONFIG_PATH"),
        tokenizer_path=_required_file("RSGPT_TOKENIZER_PATH"),
    )


def _required_dir(env_name: str) -> Path:
    raw = os.environ.get(env_name, "").strip()
    if not raw:
        raise RuntimeError(f"{env_name} is required for RSGPT planner wrapper")
    path = Path(raw)
    if not path.is_dir():
        raise RuntimeError(f"{env_name} directory not found: {path}")
    return path


def _required_file(env_name: str) -> Path:
    raw = os.environ.get(env_name, "").strip()
    if not raw:
        raise RuntimeError(f"{env_name} is required for RSGPT planner wrapper")
    path = Path(raw)
    if not path.is_file():
        raise RuntimeError(f"{env_name} file not found: {path}")
    return path


def _rsgpt_device() -> str:
    configured = os.environ.get("RSGPT_DEVICE", "").strip()
    if configured:
        return configured
    import torch

    return "cuda:0" if torch.cuda.is_available() else "cpu"


def _load_state_dict(model_path: Path, device: str) -> dict[str, object]:
    loaded = _torch_load(model_path, map_location=device)
    if isinstance(loaded, dict):
        if isinstance(loaded.get("state_dict"), dict):
            loaded = loaded["state_dict"]
        elif isinstance(loaded.get("model"), dict):
            loaded = loaded["model"]
    if not isinstance(loaded, dict):
        raise RuntimeError("RSGPT model checkpoint must contain a state dict")
    return _strip_module_prefix(loaded)


def _strip_module_prefix(loaded: dict[object, object]) -> dict[str, object]:
    has_module_prefix = any(str(key).startswith("module.") for key in loaded)
    if has_module_prefix:
        return {
            str(key)[7:] if str(key).startswith("module.") else str(key): value
            for key, value in loaded.items()
        }
    return {str(key): value for key, value in loaded.items()}


def _routes_from_reactions(
    reaction_smiles: object,
    *,
    smiles: str,
    max_routes: int,
) -> list[dict[str, object]]:
    if not isinstance(reaction_smiles, list):
        raise RuntimeError("RSGPT inference must return a list of reaction SMILES")
    routes = []
    seen: set[str] = set()
    for reaction in reaction_smiles:
        if not isinstance(reaction, str):
            continue
        normalized_reaction = reaction.strip()
        if not normalized_reaction or normalized_reaction in seen:
            continue
        seen.add(normalized_reaction)
        reactants = _reactants_from_reaction(normalized_reaction)
        route_index = len(routes) + 1
        routes.append(
            {
                "route_id": f"rsgpt-{route_index}",
                "smiles": smiles,
                "source_engine": "rsgpt",
                "reaction_smiles": [normalized_reaction],
                "steps": [
                    {
                        "step_id": "rsgpt-1",
                        "reaction": normalized_reaction,
                        "reactants": [{"smiles": item} for item in reactants],
                        "building_blocks": [{"smiles": item} for item in reactants],
                        "conditions": {"source": "rsgpt"},
                    }
                ],
                "building_blocks": [{"smiles": item} for item in reactants],
                "n_steps": 1,
            }
        )
        if len(routes) >= max_routes:
            break
    return routes


def _reactants_from_reaction(reaction: str) -> list[str]:
    if ">>" not in reaction:
        raise RuntimeError(f"RSGPT reaction must contain '>>': {reaction}")
    reactants, _product = reaction.split(">>", 1)
    blocks = [item.strip() for item in reactants.split(".") if item.strip()]
    if not blocks:
        raise RuntimeError(f"RSGPT reaction must contain reactants: {reaction}")
    return blocks


def _validated_routes(routes: object) -> list[dict[str, object]]:
    if not isinstance(routes, list):
        raise RuntimeError("RSGPT planner must return a list of route dictionaries")
    if not routes:
        raise RuntimeError("RSGPT planner returned no routes")
    normalized = []
    for index, route in enumerate(routes):
        if not isinstance(route, dict):
            raise RuntimeError("RSGPT planner routes must be JSON objects")
        route_id = str(route.get("route_id") or f"rsgpt-{index + 1}")
        steps = route.get("steps")
        reaction_smiles = route.get("reaction_smiles")
        if not isinstance(steps, list) and not isinstance(reaction_smiles, list):
            raise RuntimeError("RSGPT planner route requires steps or reaction_smiles")
        normalized_route = dict(route)
        normalized_route["route_id"] = route_id
        normalized.append(normalized_route)
    return normalized


def _positive_int(value: str, name: str) -> int:
    parsed = int(value)
    if parsed <= 0:
        raise RuntimeError(f"{name} must be positive")
    return parsed


def _bool_env(name: str, *, default: bool) -> bool:
    raw = os.environ.get(name, "").strip().lower()
    if not raw:
        return default
    if raw in {"1", "true", "yes", "on"}:
        return True
    if raw in {"0", "false", "no", "off"}:
        return False
    raise RuntimeError(f"{name} must be a boolean")


if __name__ == "__main__":
    raise SystemExit(main())
