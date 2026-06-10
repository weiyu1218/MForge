from __future__ import annotations

import importlib.util
import json
import os
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
AIZYNTH_WRAPPER = ROOT / "tools" / "retrosyn" / "aizynth_planner_wrapper.py"
RSGPT_WRAPPER = ROOT / "tools" / "retrosyn" / "rsgpt_planner_wrapper.py"
UALIGN_WRAPPER = ROOT / "tools" / "retrosyn" / "ualign_planner_wrapper.py"
RASCORE_WRAPPER = ROOT / "tools" / "retrosyn" / "rascore_planner_wrapper.py"


def _write_fake_no_init_weights(tmp_path: Path) -> None:
    transformers_dir = tmp_path / "transformers"
    transformers_dir.mkdir()
    (transformers_dir / "__init__.py").write_text("", encoding="utf-8")
    (transformers_dir / "modeling_utils.py").write_text(
        "\n".join(
            [
                "from contextlib import contextmanager",
                "",
                "@contextmanager",
                "def no_init_weights(_enable=True):",
                "    yield",
            ]
        ),
        encoding="utf-8",
    )


def test_aizynth_planner_wrapper_adapts_retrosyn_command_contract(tmp_path: Path) -> None:
    fake_src = tmp_path / "fake_src"
    package_dir = fake_src / "mf_retrosyn" / "aizynth"
    package_dir.mkdir(parents=True)
    (fake_src / "mf_retrosyn" / "__init__.py").write_text("", encoding="utf-8")
    (package_dir / "__init__.py").write_text("", encoding="utf-8")
    (package_dir / "retrosyn.py").write_text(
        "\n".join(
            [
                "class AiZynthRetrosyn:",
                "    @classmethod",
                "    def from_env(cls):",
                "        return cls()",
                "",
                "    async def find_routes(self, smiles, max_routes=10):",
                "        assert smiles == 'CCO'",
                "        assert max_routes == 1",
                "        return [",
                "            {",
                "                'route_id': 'aizynth-route-1',",
                "                'score': 0.84,",
                "                'predicted_yield': 0.61,",
                "                'steps': [",
                "                    {",
                "                        'reaction': 'CO.C>>CCO',",
                "                        'building_blocks': [{'smiles': 'CO'}, {'smiles': 'C'}],",
                "                    }",
                "                ],",
                "            }",
                "        ]",
            ]
        ),
        encoding="utf-8",
    )
    env = os.environ.copy()
    env["PYTHONPATH"] = f"{fake_src}{os.pathsep}{env.get('PYTHONPATH', '')}"

    completed = subprocess.run(  # noqa: S603
        [sys.executable, str(AIZYNTH_WRAPPER)],
        input=json.dumps({"smiles": "CCO", "max_routes": 1, "engine": "aizynth"}),
        capture_output=True,
        check=False,
        cwd=ROOT,
        env=env,
        text=True,
    )

    assert completed.returncode == 0, completed.stderr
    payload = json.loads(completed.stdout)
    assert payload["total_routes_found"] == 1
    assert payload["routes"][0]["route_id"] == "aizynth-route-1"
    assert payload["routes"][0]["score"] == 0.84
    assert payload["routes"][0]["steps"][0]["reaction"] == "CO.C>>CCO"
    assert payload["routes"][0]["steps"][0]["step_id"] == "aizynth-route-1-step-1"
    assert payload["routes"][0]["steps"][0]["operation"] == "add"
    assert payload["routes"][0]["steps"][0]["reaction_type"] == "generic"
    assert payload["routes"][0]["steps"][0]["reactants"] == [
        {"smiles": "CO"},
        {"smiles": "C"},
    ]
    assert payload["elapsed_ms"] >= 0


def test_aizynth_planner_wrapper_uses_inline_stock_without_loading_config_stock(
    tmp_path: Path,
) -> None:
    fake_src = tmp_path / "fake_src"
    aizynth_dir = fake_src / "aizynthfinder"
    stock_dir = aizynth_dir / "context" / "stock"
    chem_dir = aizynth_dir / "chem"
    retrosyn_dir = fake_src / "mf_retrosyn" / "aizynth"
    stock_dir.mkdir(parents=True)
    chem_dir.mkdir(parents=True)
    retrosyn_dir.mkdir(parents=True)
    (aizynth_dir / "__init__.py").write_text("", encoding="utf-8")
    (aizynth_dir / "context" / "__init__.py").write_text("", encoding="utf-8")
    (stock_dir / "__init__.py").write_text("", encoding="utf-8")
    (chem_dir / "__init__.py").write_text(
        "\n".join(
            [
                "class Molecule:",
                "    def __init__(self, smiles):",
                "        self.smiles = smiles",
            ]
        ),
        encoding="utf-8",
    )
    (stock_dir / "queries.py").write_text(
        "\n".join(
            [
                "class StockQueryMixin:",
                "    def __contains__(self, mol):",
                "        return False",
                "    def availability_string(self, mol):",
                "        return 'Not in stock'",
            ]
        ),
        encoding="utf-8",
    )
    (aizynth_dir / "aizynthfinder.py").write_text(
        "\n".join(
            [
                "class Collection:",
                "    def __init__(self):",
                "        self.selected = None",
                "        self.loaded = []",
                "    def select(self, name):",
                "        self.selected = name",
                "    def load(self, query, name):",
                "        self.loaded.append((query, name))",
                "",
                "class Routes:",
                "    dicts = [{'type': 'mol', 'smiles': 'CCO', 'children': [",
                "        {'type': 'reaction', 'smiles': 'CCO>>CO.C', 'children': []}",
                "    ]}]",
                "",
                "class AiZynthFinder:",
                "    def __init__(self, configdict=None, configfile=None):",
                "        assert configfile is None",
                "        assert 'stock' not in configdict",
                "        assert configdict['search']['iteration_limit'] == 7",
                "        assert configdict['search']['time_limit'] == 11",
                "        assert configdict['search']['max_transforms'] == 2",
                "        assert configdict['search']['return_first'] is True",
                "        assert configdict['post_processing']['min_routes'] == 1",
                "        assert configdict['post_processing']['max_routes'] == 1",
                "        assert configdict['expansion']['uspto']['cutoff_number'] == 3",
                "        self.expansion_policy = Collection()",
                "        self.filter_policy = Collection()",
                "        self.stock = Collection()",
                "        self.routes = Routes()",
                "    def tree_search(self):",
                "        assert self.target_smiles == 'CCO'",
                "    def build_routes(self):",
                "        assert self.expansion_policy.selected == 'uspto'",
                "        assert self.filter_policy.selected == 'uspto'",
                "        assert self.stock.selected == 'inline'",
                "        mol = type('M', (), {'smiles': 'CO'})()",
                "        assert self.stock.loaded[0][0].availability_string(mol) == 'inline'",
            ]
        ),
        encoding="utf-8",
    )
    (fake_src / "mf_retrosyn" / "__init__.py").write_text("", encoding="utf-8")
    (retrosyn_dir / "__init__.py").write_text("", encoding="utf-8")
    (retrosyn_dir / "retrosyn.py").write_text(
        "\n".join(
            [
                "def _route_dicts_from_collection(routes):",
                "    return list(routes.dicts)",
                "",
                "def _normalise_aizynth_route(route, smiles, index):",
                "    assert route['children'][0]['reaction'] == 'CCO>>CO.C'",
                "    route = dict(route)",
                "    route.setdefault('route_id', f'aizynth-{index + 1}')",
                "    route.setdefault('smiles', smiles)",
                "    route.setdefault('steps', [{'reaction': route['children'][0]['reaction']}])",
                "    return route",
                "",
                "def _complete_aizynth_route(route):",
                "    return route",
            ]
        ),
        encoding="utf-8",
    )
    config_path = tmp_path / "config.yml"
    config_path.write_text(
        "\n".join(
            [
                "expansion:",
                "  uspto:",
                "    - /model.onnx",
                "    - /templates.csv.gz",
                "filter:",
                "  uspto: /filter.onnx",
                "stock:",
                "  zinc: /must-not-load.hdf5",
            ]
        ),
        encoding="utf-8",
    )
    env = os.environ.copy()
    env["PYTHONPATH"] = f"{fake_src}{os.pathsep}{env.get('PYTHONPATH', '')}"
    env["AIZYNTH_CONFIG_PATH"] = str(config_path)
    env["AIZYNTH_STOCK"] = "inline"
    env["AIZYNTH_STOCK_SMILES"] = "CO,CC(=O)OC(C)=O"
    env["AIZYNTH_EXPANSION_POLICY"] = "uspto"
    env["AIZYNTH_FILTER_POLICY"] = "uspto"
    env["AIZYNTH_SEARCH_ITERATION_LIMIT"] = "7"
    env["AIZYNTH_SEARCH_TIME_LIMIT"] = "11"
    env["AIZYNTH_SEARCH_MAX_TRANSFORMS"] = "2"
    env["AIZYNTH_SEARCH_RETURN_FIRST"] = "true"
    env["AIZYNTH_EXPANSION_CUTOFF_NUMBER"] = "3"

    completed = subprocess.run(  # noqa: S603
        [sys.executable, str(AIZYNTH_WRAPPER)],
        input=json.dumps({"smiles": "CCO", "max_routes": 1, "engine": "aizynth"}),
        capture_output=True,
        check=False,
        cwd=ROOT,
        env=env,
        text=True,
    )

    assert completed.returncode == 0, completed.stderr
    payload = json.loads(completed.stdout)
    assert payload["total_routes_found"] == 1
    assert payload["routes"][0]["route_id"] == "aizynth-1"
    assert payload["routes"][0]["steps"][0]["reaction"] == "CCO>>CO.C"


def test_rsgpt_planner_wrapper_adapts_official_inference_contract(tmp_path: Path) -> None:
    fake_src = tmp_path / "RSGPT"
    models_dir = fake_src / "models"
    tokenizer_dir = fake_src / "tokenizer"
    utils_dir = fake_src / "utils"
    models_dir.mkdir(parents=True)
    tokenizer_dir.mkdir(parents=True)
    utils_dir.mkdir(parents=True)
    (models_dir / "__init__.py").write_text("", encoding="utf-8")
    (tokenizer_dir / "__init__.py").write_text("", encoding="utf-8")
    (utils_dir / "__init__.py").write_text("", encoding="utf-8")
    (tmp_path / "torch.py").write_text(
        "\n".join(
            [
                "class _Cuda:",
                "    @staticmethod",
                "    def is_available():",
                "        return False",
                "",
                "cuda = _Cuda()",
                "",
                "def load(path, map_location=None):",
                "    return {'module.weight': 'value'}",
            ]
        ),
        encoding="utf-8",
    )
    _write_fake_no_init_weights(tmp_path)
    rdkit_dir = tmp_path / "rdkit"
    rdkit_dir.mkdir()
    (rdkit_dir / "__init__.py").write_text("", encoding="utf-8")
    (rdkit_dir / "Chem.py").write_text(
        "\n".join(
            [
                "def MolFromSmiles(smiles):",
                "    return {'smiles': smiles} if smiles and smiles != 'O=' else None",
                "",
                "def MolToSmiles(mol):",
                "    return mol['smiles']",
            ]
        ),
        encoding="utf-8",
    )
    (models_dir / "rxngpt.py").write_text(
        "\n".join(
            [
                "class RxnGPT:",
                "    def __init__(self, cfg, Tokenizer=None):",
                "        self.cfg = cfg",
                "    def load_state_dict(self, state):",
                "        self.state = state",
                "    def half(self):",
                "        self.precision = 'half'",
                "        return self",
                "    def to(self, device):",
                "        self.device = device",
                "        return self",
                "    def eval(self):",
                "        self.evaluated = True",
                "        return self",
            ]
        ),
        encoding="utf-8",
    )
    (tokenizer_dir / "tokenization.py").write_text(
        "\n".join(
            [
                "class SMILESBPETokenizer:",
                "    @classmethod",
                "    def get_hf_tokenizer(cls, path, model_max_length=100):",
                "        return cls()",
            ]
        ),
        encoding="utf-8",
    )
    (utils_dir / "utils.py").write_text(
        "\n".join(
            [
                "class Cfg:",
                "    pass",
                "",
                "def args_parse(config_file=''):",
                "    cfg = Cfg()",
                "    cfg.config_file = config_file",
                "    return cfg",
            ]
        ),
        encoding="utf-8",
    )
    (fake_src / "infer.py").write_text(
        "\n".join(
            [
                "def beam_search_gpt(",
                "    model, tokenizer, std_smiles, beam_size=10, max_length=50, device='cpu'",
                "):",
                "    assert std_smiles == 'CCO'",
                "    assert beam_size == 2",
                "    assert max_length == 12",
                "    assert device == 'cpu'",
                "    return ['decoded-a', 'decoded-b']",
                "",
                "def jiexi(input_texts):",
                "    assert input_texts == ['decoded-a', 'decoded-b']",
                "    return ['O=.C>>CCO', 'CO.C>>CCO']",
            ]
        ),
        encoding="utf-8",
    )
    model_path = tmp_path / "model.pth"
    config_path = tmp_path / "base.yml"
    tokenizer_path = tmp_path / "tokenizer.json"
    for path in (model_path, config_path, tokenizer_path):
        path.write_text("artifact", encoding="utf-8")

    env = os.environ.copy()
    env["PYTHONPATH"] = f"{tmp_path}{os.pathsep}{env.get('PYTHONPATH', '')}"
    env["RSGPT_SOURCE_DIR"] = str(fake_src)
    env["RSGPT_MODEL_PATH"] = str(model_path)
    env["RSGPT_CONFIG_PATH"] = str(config_path)
    env["RSGPT_TOKENIZER_PATH"] = str(tokenizer_path)
    env["RSGPT_DEVICE"] = "cpu"
    env["RSGPT_BEAM_SIZE"] = "2"
    env["RSGPT_MAX_LENGTH"] = "12"

    completed = subprocess.run(  # noqa: S603
        [sys.executable, str(RSGPT_WRAPPER)],
        input=json.dumps({"smiles": "CCO", "max_routes": 1, "engine": "rsgpt"}),
        capture_output=True,
        check=False,
        cwd=ROOT,
        env=env,
        text=True,
    )

    assert completed.returncode == 0, completed.stderr
    payload = json.loads(completed.stdout)
    assert payload["total_routes_found"] == 1
    assert payload["routes"][0]["route_id"] == "rsgpt-1"
    assert payload["routes"][0]["source_engine"] == "rsgpt"
    assert payload["routes"][0]["steps"][0]["reaction"] == "CO.C>>CCO"
    assert payload["routes"][0]["steps"][0]["operation"] == "add"
    assert payload["routes"][0]["steps"][0]["reaction_type"] == "generic"
    assert payload["routes"][0]["building_blocks"] == [{"smiles": "CO"}, {"smiles": "C"}]


def test_rsgpt_planner_wrapper_accepts_official_llama_config_json(tmp_path: Path) -> None:
    fake_src = tmp_path / "RSGPT"
    models_dir = fake_src / "models"
    tokenizer_dir = fake_src / "tokenizer"
    utils_dir = fake_src / "utils"
    models_dir.mkdir(parents=True)
    tokenizer_dir.mkdir(parents=True)
    utils_dir.mkdir(parents=True)
    (models_dir / "__init__.py").write_text("", encoding="utf-8")
    (tokenizer_dir / "__init__.py").write_text("", encoding="utf-8")
    (utils_dir / "__init__.py").write_text("", encoding="utf-8")
    (tmp_path / "torch.py").write_text(
        "\n".join(
            [
                "class _Cuda:",
                "    @staticmethod",
                "    def is_available():",
                "        return False",
                "",
                "cuda = _Cuda()",
                "",
                "def load(path, map_location=None):",
                "    return {'module.weight': 'value'}",
            ]
        ),
        encoding="utf-8",
    )
    _write_fake_no_init_weights(tmp_path)
    rdkit_dir = tmp_path / "rdkit"
    rdkit_dir.mkdir()
    (rdkit_dir / "__init__.py").write_text("", encoding="utf-8")
    (rdkit_dir / "Chem.py").write_text(
        "\n".join(
            [
                "def MolFromSmiles(smiles):",
                "    return {'smiles': smiles} if smiles else None",
                "",
                "def MolToSmiles(mol):",
                "    return mol['smiles']",
            ]
        ),
        encoding="utf-8",
    )
    (models_dir / "rxngpt.py").write_text(
        "\n".join(
            [
                "class RxnGPT:",
                "    def __init__(self, cfg, Tokenizer=None):",
                "        assert cfg.MODEL.GPT_MODEL.config_path.endswith('rxngpt_llama1B.json')",
                "        assert cfg.DATA.MAX_ATOM_NUM == 12",
                "    def load_state_dict(self, state):",
                "        self.state = state",
                "    def half(self):",
                "        self.precision = 'half'",
                "        return self",
                "    def to(self, device):",
                "        self.device = device",
                "        return self",
                "    def eval(self):",
                "        self.evaluated = True",
                "        return self",
            ]
        ),
        encoding="utf-8",
    )
    (tokenizer_dir / "tokenization.py").write_text(
        "\n".join(
            [
                "class SMILESBPETokenizer:",
                "    @classmethod",
                "    def get_hf_tokenizer(cls, path, model_max_length=100):",
                "        return cls()",
            ]
        ),
        encoding="utf-8",
    )
    (utils_dir / "utils.py").write_text(
        "\n".join(
            [
                "def args_parse(config_file=''):",
                "    raise AssertionError('Llama config JSON should not call args_parse')",
            ]
        ),
        encoding="utf-8",
    )
    (fake_src / "infer.py").write_text(
        "\n".join(
            [
                "def beam_search_gpt(",
                "    model, tokenizer, std_smiles, beam_size=10, max_length=50, device='cpu'",
                "):",
                "    assert beam_size == 1",
                "    assert max_length == 12",
                "    return ['decoded']",
                "",
                "def jiexi(input_texts):",
                "    return ['CO.C>>CCO']",
            ]
        ),
        encoding="utf-8",
    )
    model_path = tmp_path / "model.pth"
    config_path = tmp_path / "rxngpt_llama1B.json"
    tokenizer_path = tmp_path / "tokenizer.json"
    model_path.write_text("artifact", encoding="utf-8")
    tokenizer_path.write_text("artifact", encoding="utf-8")
    config_path.write_text(
        json.dumps(
            {
                "model_type": "llama",
                "hidden_size": 2048,
                "num_hidden_layers": 24,
                "num_attention_heads": 32,
            }
        ),
        encoding="utf-8",
    )

    spec = importlib.util.spec_from_file_location(
        "rsgpt_planner_wrapper",
        RSGPT_WRAPPER,
    )
    assert spec is not None
    assert spec.loader is not None
    rsgpt_planner_wrapper = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(rsgpt_planner_wrapper)

    cfg = rsgpt_planner_wrapper._rsgpt_config(config_path, max_length=12)

    assert rsgpt_planner_wrapper._config_path_is_llama_json(config_path)
    assert cfg.MODEL.GPT_MODEL.config_path == str(config_path)
    assert cfg.DATA.MAX_ATOM_NUM == 12


def test_rsgpt_planner_wrapper_requires_artifact_configuration() -> None:
    env = os.environ.copy()
    for name in (
        "RSGPT_SOURCE_DIR",
        "RSGPT_MODEL_PATH",
        "RSGPT_CONFIG_PATH",
        "RSGPT_TOKENIZER_PATH",
    ):
        env.pop(name, None)

    completed = subprocess.run(  # noqa: S603
        [sys.executable, str(RSGPT_WRAPPER)],
        input=json.dumps({"smiles": "CCO", "max_routes": 1, "engine": "rsgpt"}),
        capture_output=True,
        check=False,
        cwd=ROOT,
        env=env,
        text=True,
    )

    assert completed.returncode == 1
    assert "RSGPT_SOURCE_DIR is required" in completed.stderr


def test_rsgpt_planner_wrapper_returns_empty_routes_when_predictions_are_invalid() -> None:
    spec = importlib.util.spec_from_file_location(
        "rsgpt_planner_wrapper_empty_routes_test",
        RSGPT_WRAPPER,
    )
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

    class Chem:
        @staticmethod
        def MolFromSmiles(smiles: str):
            return {"smiles": smiles} if smiles != "O=" else None

    routes = module._routes_from_reactions(
        ["O=.C>>CCO"],
        smiles="CCO",
        max_routes=1,
        chem_module=Chem,
    )
    normalized = module._validated_routes(routes)

    assert normalized == []


def test_ualign_planner_wrapper_adapts_official_result_contract(tmp_path: Path) -> None:
    fake_src = tmp_path / "UAlign"
    fake_src.mkdir()
    script = fake_src / "inference_one.py"
    script.write_text(
        "\n".join(
            [
                "import argparse",
                "import json",
                "import pathlib",
                "import sys",
                "",
                "root = pathlib.Path(__file__).resolve().parents[1]",
                "sys.path.insert(0, str(root))",
                "from rdkit import Chem",
                "assert Chem.MolFromSmiles('O=') is None",
                "assert Chem.MolFromSmiles('CO') is not None",
                "",
                "parser = argparse.ArgumentParser()",
                "parser.add_argument('--model_arch_path')",
                "parser.add_argument('--checkpoint')",
                "parser.add_argument('--token_ckpt')",
                "parser.add_argument('--device')",
                "parser.add_argument('--beams')",
                "parser.add_argument('--max_len')",
                "parser.add_argument('--product_smiles')",
                "args = parser.parse_args()",
                "assert args.product_smiles == 'CCO'",
                "assert args.beams == '2'",
                "assert args.max_len == '17'",
                "print('[INFO] fake UAlign')",
                "print('[RESULT]')",
                "print(json.dumps({'answers': ['O=.C', 'CO.C'], 'probs': [-0.5, -1.2]}))",
            ]
        ),
        encoding="utf-8",
    )
    rdkit_dir = tmp_path / "rdkit"
    rdkit_dir.mkdir()
    (rdkit_dir / "__init__.py").write_text("", encoding="utf-8")
    (rdkit_dir / "Chem.py").write_text(
        "\n".join(
            [
                "def MolFromSmiles(smiles):",
                "    return {'smiles': smiles} if smiles and smiles != 'O=' else None",
            ]
        ),
        encoding="utf-8",
    )
    model_arch_path = tmp_path / "arch.json"
    checkpoint_path = tmp_path / "model.pth"
    token_path = tmp_path / "token.pkl"
    for path in (model_arch_path, checkpoint_path, token_path):
        path.write_text("artifact", encoding="utf-8")

    env = os.environ.copy()
    env["UALIGN_SOURCE_DIR"] = str(fake_src)
    env["UALIGN_MODEL_ARCH_PATH"] = str(model_arch_path)
    env["UALIGN_CHECKPOINT_PATH"] = str(checkpoint_path)
    env["UALIGN_TOKEN_CKPT_PATH"] = str(token_path)
    env["UALIGN_DEVICE"] = "0"
    env["UALIGN_BEAMS"] = "2"
    env["UALIGN_MAX_LEN"] = "17"

    completed = subprocess.run(  # noqa: S603
        [sys.executable, str(UALIGN_WRAPPER)],
        input=json.dumps({"smiles": "CCO", "max_routes": 1, "engine": "ualign"}),
        capture_output=True,
        check=False,
        cwd=ROOT,
        env=env,
        text=True,
    )

    assert completed.returncode == 0, completed.stderr
    payload = json.loads(completed.stdout)
    assert payload["total_routes_found"] == 1
    assert payload["routes"][0]["route_id"] == "ualign-1"
    assert payload["routes"][0]["source_engine"] == "ualign"
    assert payload["routes"][0]["steps"][0]["reaction"] == "CO.C>>CCO"
    assert payload["routes"][0]["steps"][0]["operation"] == "add"
    assert payload["routes"][0]["steps"][0]["reaction_type"] == "generic"
    assert payload["routes"][0]["score"] == -1.2
    assert payload["routes"][0]["building_blocks"] == [{"smiles": "CO"}, {"smiles": "C"}]


def test_ualign_planner_wrapper_requires_existing_artifacts(tmp_path: Path) -> None:
    env = os.environ.copy()
    env["UALIGN_SOURCE_DIR"] = str(tmp_path)
    env["UALIGN_MODEL_ARCH_PATH"] = str(tmp_path / "missing-arch.json")
    env["UALIGN_CHECKPOINT_PATH"] = str(tmp_path / "missing-model.pth")
    env["UALIGN_TOKEN_CKPT_PATH"] = str(tmp_path / "missing-token.pkl")

    completed = subprocess.run(  # noqa: S603
        [sys.executable, str(UALIGN_WRAPPER)],
        input=json.dumps({"smiles": "CCO", "max_routes": 1, "engine": "ualign"}),
        capture_output=True,
        check=False,
        cwd=ROOT,
        env=env,
        text=True,
    )

    assert completed.returncode == 1
    assert "UALIGN_MODEL_ARCH_PATH file not found" in completed.stderr


def test_ualign_planner_wrapper_returns_empty_routes_when_answers_are_invalid() -> None:
    spec = importlib.util.spec_from_file_location(
        "ualign_planner_wrapper_empty_routes_test",
        UALIGN_WRAPPER,
    )
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

    class Chem:
        @staticmethod
        def MolFromSmiles(smiles: str):
            return {"smiles": smiles} if smiles != "O=" else None

    routes = module._routes_from_result(
        {"answers": ["O=.C"], "probs": [-0.5]},
        smiles="CCO",
        max_routes=1,
        chem_module=Chem,
    )
    normalized = module._validated_routes(routes)

    assert normalized == []


def test_rascore_planner_wrapper_adapts_xgb_accessibility_score(tmp_path: Path) -> None:
    rdkit_dir = tmp_path / "rdkit"
    chem_dir = rdkit_dir / "Chem"
    chem_dir.mkdir(parents=True)
    (rdkit_dir / "__init__.py").write_text("", encoding="utf-8")
    (chem_dir / "__init__.py").write_text(
        "\n".join(
            [
                "def MolFromSmiles(smiles):",
                "    return {'smiles': smiles} if smiles else None",
            ]
        ),
        encoding="utf-8",
    )
    (chem_dir / "AllChem.py").write_text(
        "\n".join(
            [
                "class Fingerprint:",
                "    def GetNonzeroElements(self):",
                "        return {1: 2, 2049: 3}",
                "",
                "def GetMorganFingerprint(mol, radius, useCounts=True, useFeatures=False):",
                "    assert mol['smiles'] == 'CCO'",
                "    assert radius == 3",
                "    assert useCounts is True",
                "    assert useFeatures is False",
                "    return Fingerprint()",
            ]
        ),
        encoding="utf-8",
    )
    (tmp_path / "xgboost.py").write_text(
        "\n".join(
            [
                "class Booster:",
                "    def load_model(self, path):",
                "        assert path.endswith('model.json')",
                "    def inplace_predict(self, values):",
                "        assert values.shape == (1, 2048)",
                "        assert float(values[0][1]) == 5.0",
                "        return [0.73]",
            ]
        ),
        encoding="utf-8",
    )
    model_path = tmp_path / "model.json"
    model_path.write_text("{}", encoding="utf-8")

    env = os.environ.copy()
    env["PYTHONPATH"] = f"{tmp_path}{os.pathsep}{env.get('PYTHONPATH', '')}"
    env["RASCORE_MODEL_PATH"] = str(model_path)

    completed = subprocess.run(  # noqa: S603
        [sys.executable, str(RASCORE_WRAPPER)],
        input=json.dumps({"smiles": "CCO", "max_routes": 1, "engine": "rascore"}),
        capture_output=True,
        check=False,
        cwd=ROOT,
        env=env,
        text=True,
    )

    assert completed.returncode == 0, completed.stderr
    payload = json.loads(completed.stdout)
    assert payload["total_routes_found"] == 1
    route = payload["routes"][0]
    assert route["route_id"] == "rascore-1"
    assert route["source_engine"] == "rascore"
    assert route["route_type"] == "retrosynthetic_accessibility_score"
    assert route["score"] == 0.73
    assert route["predicted_score"] == 0.73
    assert route["accessibility_score"] == 0.73
    assert route["n_steps"] == 0
    assert route["steps"] == []


def test_rascore_planner_wrapper_requires_existing_model(tmp_path: Path) -> None:
    env = os.environ.copy()
    env["RASCORE_MODEL_PATH"] = str(tmp_path / "missing.json")

    completed = subprocess.run(  # noqa: S603
        [sys.executable, str(RASCORE_WRAPPER)],
        input=json.dumps({"smiles": "CCO", "max_routes": 1, "engine": "rascore"}),
        capture_output=True,
        check=False,
        cwd=ROOT,
        env=env,
        text=True,
    )

    assert completed.returncode == 1
    assert "RASCORE_MODEL_PATH file not found" in completed.stderr
