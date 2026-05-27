# HUMU D3 Data Preparation TODO

更新时间：2026-05-20

本文只记录 HUMU D3 严格联合训练前的数据补齐任务，不记录训练成功结论。

## 当前判断

当前 CrossDocked 没有全量解压，可能导致 pocket 侧 ligand 覆盖不足，也会影响后续从 receptor PDB 提取 protein sequence 和预计算 ESM-2 embedding。

但这不能作为直接生成 `joint_source` 的依据。当前已处理数据中：

```text
pocket_total=24242
pocket_unique_raw=5471
route_total=409035
route_unique_raw=398768
raw_match_rows=0
```

因此当前 `data/processing/humu_pretrain/pocket` 和 `data/processing/humu_pretrain/route` 不能直接拼成 `mol-pocket-route` 三元正样本。全量解压后必须重新处理 pocket，并重新验证 CrossDocked ligand 与 route product 的交集；只有能证明同一个 ligand 同时绑定 pocket 和拥有真实 route，才能写入 `joint_source`。

## TODO 1. 继续解压全部 CrossDocked 数据集

目标：补齐 `/workspace/MForge/zzzzz/CrossDocked2020/m/CrossDocked2020_v1.3.tar` 中尚未解压的 CrossDocked 原始文件，使后续 pocket 重处理基于完整原始数据。

已确认输入：

```text
/workspace/MForge/zzzzz/CrossDocked2020/m/CrossDocked2020_v1.3.tar
/workspace/MForge/zzzzz/CrossDocked2020/CrossDocked2020_v1.3_types.tgz
```

执行要求：

1. 解压前检查剩余磁盘空间，避免解压中断造成新的半成品目录。
2. 完成 `CrossDocked2020_v1.3.tar` 全量解压到 `/workspace/MForge/zzzzz/CrossDocked2020/`。
3. 确认 `CrossDocked2020_v1.3_types.tgz` 中的 `types/it2_tt_v1.3_completeset_test0.types` 可用。
4. 使用当前 HUMU pocket 数据契约重建 `data/processing/humu_pretrain/pocket`，输出必须保留：
   - `index.jsonl`
   - `manifest.json`
   - `pocket_*.json`
   - `rejects.ndjson`
5. 重建后重新统计：
   - pocket 总记录数
   - unique `ligand_smiles`
   - 与 `route/routes.jsonl` 中 `root_smiles` 的 raw exact match
   - RDKit canonical SMILES match
6. 只有存在真实 ligand identity 交集时，才进入 `joint_source` 生成；如果交集仍为 0，则不能从 CrossDocked + USPTO-MIT 直接生成正式 joint 数据。

验收条件：

```text
data/processing/humu_pretrain/pocket/index.jsonl exists
data/processing/humu_pretrain/pocket/manifest.json exists
data/processing/humu_pretrain/pocket/pocket_*.json count > current count
joint candidate generation report contains raw_match_rows and canonical_match_rows
```

## TODO 2. 补齐 pocket 的 ESM-2 输入

目标：让 `encoders.pocket.use_esm2: true` 时，训练 batch 中每条 pocket 都包含真实 `protein_sequence`、`sequence` 或 `esm2_embedding`，避免首个 batch 失败。

当前阻塞：

```text
ValueError: ESM-2 input requires protein_sequence, sequence, or esm2_embedding
```

当前可用资源：

```text
models/esm2/esm2_t33_650M_UR50D.pt
models/esm2/esm2_alphabet.pkl
esm Python package import ok
```

执行要求：

1. 从每条 pocket 的 `source_receptor_pdb` 定位全量 CrossDocked receptor PDB。
2. 从 receptor PDB 的 ATOM 记录提取真实氨基酸序列，保留 chain 信息。
3. 将序列写入 pocket sidecar 或 joint record，字段使用当前 encoder 支持的 `protein_sequence` 或 `sequence`。
4. 优先离线预计算 ESM-2 embedding：
   - 使用 `esm2_t33_650M_UR50D.pt`
   - 取 `esm2_layer=33`
   - 生成 1280 维 mean pooled embedding
   - 写入字段 `esm2_embedding`
5. 更新 HUMU data loader，使 `PocketDataset.__getitem__()` 和 joint pocket payload 不丢弃 `protein_sequence`、`sequence`、`esm2_embedding`。
6. 增加验证：启用 `use_esm2: true` 时，至少一条真实 pocket 经过 `HUMUPocketEncoder.encode()` 不再因缺 ESM-2 输入失败。

验收条件：

```text
each training pocket has protein_sequence or esm2_embedding
esm2_embedding length is 1280 when present
HUMUPocketEncoder(use_esm2=True) accepts a processed pocket record
pipelines/humu_pretrain/train.py --preflight-only passes ESM-2 input contract check after loader validation is extended
```

在 `joint_source` 和 pocket ESM-2 输入都真实补齐前，不能声明 D3 严格联合训练已可运行。
