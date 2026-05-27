# HUMU Pretrain Data Processing

本文档描述 HUMU 预训练数据集的当前处理结果、目录契约和训练读取约束。训练入口为 `pipelines/humu_pretrain/train.py`，数据加载实现为 `pipelines/humu_pretrain/src/humu_pretrain/data_loader.py`。

## 数据来源

原始数据根目录为 `/workspace/MForge/zzzzz`。

| 组件 | 原始来源 | 用途 |
| --- | --- | --- |
| molecule | `/workspace/MForge/zzzzz/Chembl/chembl_36/chembl_36_sqlite/chembl_36.db` | 分子 SMILES 预训练样本 |
| pocket | `/workspace/MForge/zzzzz/CrossDocked2020/m/CrossDocked2020_v1.3.tar` + `/workspace/MForge/zzzzz/CrossDocked2020/CrossDocked2020_v1.3_types.tgz` | 蛋白口袋和配体位姿样本 |
| route | `/workspace/MForge/zzzzz/USPTO-MIT/USPTO-MIT.zip` | 反应路线训练样本 |

## 处理后目录

正式训练数据目录为：

```text
/workspace/MForge/moleculeforge/data/processing/humu_pretrain
```

目录结构：

```text
humu_pretrain/
  manifest.json
  mol/
    manifest.json
    shard_0000.jsonl ... shard_0057.jsonl
    rejects.jsonl
  pocket/
    manifest.json
    index.jsonl
    pocket_000000.json ... pocket_024532.json
    rejects.ndjson
  route/
    manifest.json
    routes.jsonl
  route_eval/
    manifest.json
    routes_valid.jsonl
    routes_test.jsonl
```

## 数据规模

| 组件 | 记录数 | 拒绝/无效 | 说明 |
| --- | ---: | ---: | --- |
| mol | 2,854,636 | invalid SMILES 15, duplicate SMILES 164 | ChEMBL SMILES 经 RDKit canonicalize 后分片写入 58 个 shard |
| pocket | 24,242 | 291 | CrossDocked official types 选择 24,533 个 receptor 条目，成功生成 24,242 个 pocket sidecar |
| route | 409,035 | 0 | USPTO-MIT train split |
| route_eval | 70,000 | 0 | USPTO-MIT valid 30,000 + test 40,000，不作为训练输入 |

根 manifest 当前内容以 `/workspace/MForge/moleculeforge/data/processing/humu_pretrain/manifest.json` 为准。

## 处理规则

### molecule

ChEMBL molecule 数据从 SQLite 表读取，使用 RDKit 解析并 canonicalize SMILES。无效 SMILES 写入 `mol/rejects.jsonl`，重复 canonical SMILES 不进入训练 shard。

训练样本字段由 `MoleculeDataset` 消费：

```text
smiles
inchikey
mw
logp
```

### pocket

CrossDocked pocket 数据使用 official types 成员：

```text
types/it2_tt_v1.3_completeset_test0.types
```

同一 `receptor_gnina` 的候选选择规则为：

```text
prefer label=1, then lower rmsd, then lower types_score
```

每条成功样本在 `pocket/index.jsonl` 中保留索引记录，并在同目录下写入 `pocket_XXXXXX.json` sidecar。sidecar 内包含 10A cutoff 内的 receptor pocket atoms、ligand canonical SMILES 和来源路径。

`PocketDataset` 会读取 `pocket/` 下所有 `*.jsonl` 文件，因此训练样本文件只能保留 `index.jsonl`。reject 明细必须使用 `rejects.ndjson`，不能命名为 `*.jsonl`。

### route

USPTO-MIT 的 train split 写入 `route/routes.jsonl`，valid/test split 写入 `route_eval/`。训练配置只应把 `data.route_source` 指向 `route/`，避免把评估 split 混入训练。

训练样本字段由 `RouteDataset` 消费：

```text
root_smiles
n_steps
steps
tree_depth
reaction_types
reactions
intermediates
score
```

## 训练配置契约

HUMU 训练配置中的数据路径必须指向处理后子目录：

```yaml
data:
  mol_source: "/workspace/MForge/moleculeforge/data/processing/humu_pretrain/mol"
  pocket_source: "/workspace/MForge/moleculeforge/data/processing/humu_pretrain/pocket"
  route_source: "/workspace/MForge/moleculeforge/data/processing/humu_pretrain/route"
```

默认 `configs/models/humu_pretrain.yaml` 中的原始 `zzzzz` 路径不能直接作为正式训练输入。

## 设备使用

数据处理阶段主要是 SQLite 读取、RDKit 解析、PDB/SDF/gzip/tar/zip 解析和 JSONL 写出，属于 CPU/I/O 任务。GPU 不参与数据处理。4 张 H200 应用于后续 `torchrun` 分布式训练。

## 验证状态

已用 HUMU data loader 对处理后目录做小样本读取验证：

```text
dataset_lengths {'mol': 64, 'pocket': 64, 'route': 64}
mol ['inchikey', 'logp', 'mw', 'smiles']
pocket ['coords', 'elements', 'ligand_smiles', 'pdb_id', 'residue_types']
route ['intermediates', 'n_steps', 'reaction_types', 'reactions', 'root_smiles', 'score', 'steps', 'tree_depth']
```

该验证只确认数据目录结构和 loader 字段契约可用，不代表完整训练已完成。
