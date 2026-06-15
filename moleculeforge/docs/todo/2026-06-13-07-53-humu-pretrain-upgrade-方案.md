# HUMU 预训练升级方案

## 目标

在已完成新数据接入验证的基础上，就地升级 HUMU 预训练能力，使现有 `mol`、`pocket`、`route` 三塔从“可训练的双曲对齐 baseline”推进到更接近 CoreArchitecture v2 设想的共享双曲表示底座。升级必须保持现有 checkpoint key、HUMU encoder service 外部 129 维输出契约和三塔入口不破坏；不恢复 `intent_encoder`，不创建并行 `v2` 训练入口。

本方案只设计 HUMU 预训练升级，不实施代码修改，不更新 README。

## 当前证据

### 代码状态

- 当前分支：`feature/humu-pretrain-optimization`。
- 最近已提交基线：`9a4a555 chore: 同步 humu 预训练数据进度`。
- 当前工作区存在上一阶段数据接入改动，尚未提交：
  - `configs/models/humu_pretrain.yaml`
  - `pipelines/humu_pretrain/src/humu_pretrain/data_loader.py`
  - `pipelines/humu_pretrain/src/humu_pretrain/pipeline.py`
  - `tests/unit/test_humu_training.py`
  - `docs/todo/2026-06-13-07-11-humu-pretrain-data-integration-方案.md`

### 已验证的数据接入基线

上一阶段真实验证结果：

- `pytest tests/unit/test_humu_training.py tests/unit/test_learnable_curvature.py -q`：84 项通过。
- 当前配置 preflight 返回：
  - `activity_sources.records=3203943`
  - `bindingdb_activity.records=3145942`
  - `protac_sources.records=3581`
  - `pocket_source.esm2_records=43421`
  - `joint_source.records=4353`
  - `route_source.records=409035`
- 数据 smoke 显示 paired batch 已包含 `protac_component`，且存在 `component_smiles`。

### 架构文档与附件目标

- `docs/architecture/current-implementation-vs-corearchitecture-v2.md` 明确记录：HUMU 已有 Lorentz 基础、mol/pocket/route 预训练 pipeline，但仍不是 SE(3) message passing、E(3)-GNN 或图 Transformer + 双曲 TreeLSTM。
- 附件 `6-11.docx` 的目标是把小分子、大分子、PROTAC、靶点口袋、E3 连接酶、linker 构象、三元复合体几何关系和合成路径统一到同一双曲空间。该内容作为目标描述使用，不作为当前代码已达成能力。
- `docs/adr/0004-humu-dim-128.md` 已采纳 HUMU 空间维度为 128，即 Lorentz 坐标 129 维。升级不能改变外部向量维度。

### 处理后数据资产

`data/processing/humu_pretrain/manifest.json` 显示：

- `mol`: 2854636 条小分子记录。
- `pocket`: 43421 条 pocket 记录，来自 CrossDocked 和 PDBBind，pocket 记录已有坐标、残基类型、蛋白序列。
- `route`: 409035 条路线记录。
- `route_eval`: 70000 条路线评估记录。
- `joint`: 4353 条 mol-pocket-route 记录，344 个 unique canonical ligands。
- `bindingdb_activity`: 3145942 条活性记录，字段包含 `ligand_smiles`、`target_id`、`activity_value`。
- `protacpedia`: 1203 条 PROTAC 记录，PROTAC、E3 binder、ligand、linker SMILES 可用，其中 linker 有 1175 条有效。
- `activity`: 58001 条历史 activity 记录，已作为 activity validation 和 activity source 的一部分使用。
- `protacdb`: 46115 条 SDF 记录，组件表包含 e3_ligand、linker、warhead、mg、xtac 等，但当前主表与组件表没有稳定的一条记录内 anchor/component 训练配对合同。
- `protac8k`: archive index 覆盖 8007 个文件，目录包含 `target_pocket`、`ligase_pocket`、`target_ligand`、`ligase_ligand`、`protac` 和 `features`。
- `rcsb_mmcif`: 47454 条 mmCIF 路径索引，来源包含 BioLiP、Propedia、SKEMPI、PROTACpedia、PROTAC-DB。
- `interface_skempi2`: 7085 条突变/亲和力表记录，样本字段是 PDB complex、mutation、affinity，没有可直接进入当前 pocket encoder 的结构 payload。
- `pdcdb`: 2047 条 PDC 记录和 180 条 linker 记录，样本不含可编码 SMILES 或序列字段。
- `retropath_templates`: 1243711 条反应模板记录，可作为 route/template 结构监督信号。

## 当前调用链路分析

```text
pipelines/humu_pretrain/train.py
  -> yaml.safe_load(config)
  -> humu_pretrain.pipeline.run(config)
     -> _validate_config(config)
     -> _build_encoders(config, device)
        -> HUMUMoleculeEncoder
        -> HUMUPocketEncoder
        -> HUMURouteEncoder
        -> _wrap_as_module(...)
     -> humu_pretrain.data_loader.create_dataloaders(config)
        -> PairedHUMUDataset
           -> PocketDataset
           -> RouteDataset
           -> joint JSONL records
           -> PROTAC component records
        -> _record_collate(...)
     -> epoch loop
        -> _forward_paired_batch(encoders, paired_batch, cfg)
           -> mol encoder encodes ligand_smiles
           -> pocket encoder encodes pocket payload for mol_pocket / mol_pocket_route
           -> route encoder encodes route payload for mol_route / mol_pocket_route
           -> mol encoder encodes component_smiles for protac_component
        -> _compute_losses(...)
           -> l_mol_pocket
           -> l_mol_route
           -> l_pocket_route
           -> l_protac_component
           -> l_curvature_reg
        -> backward / optimizer / scheduler
        -> _validate_epoch(...)
        -> _save_checkpoint(...)
```

### 数据流转

- `PairedHUMUDataset.samples` 当前按 `pair_type` 区分训练样本：
  - `mol_pocket`
  - `mol_route`
  - `mol_pocket_route`
  - `protac_component`
- `_record_collate` 把不同 `pair_type` 的可选 payload 统一成 batch 字段。
- `_forward_paired_batch` 通过索引筛选每个 tower 需要的样本，缺失 tower 用 DDP dummy 保证分布式参数同步。
- `_compute_losses` 使用 Lorentz geodesic distance 构造 in-batch contrastive loss。
- `_validate_epoch` 复用 forward/loss，并可计算 route tree distortion 和 activity cliff AUROC。

### 现有实现偏差

- `configs/models/humu_pretrain.yaml` 中 `hidden_dim`、`n_layers`、`dropout` 等 encoder 配置大多没有被当前 encoder 消费。
- `HUMUMoleculeEncoder` 只使用 16 维 RDKit atom feature、2 轮邻接传播和 `LorentzAttention`；有坐标时只加入 E(3)-invariant 距离统计，不是 SE(3)-equivariant message passing。
- `HUMUPocketEncoder` 使用 pocket 点云距离统计、残基类别和可选 ESM2 embedding；不是 E(3)-GNN 或结构等变层。
- `HUMURouteEncoder` 使用 18 维人工 route feature 和 MLP；不是 reaction graph Transformer 或双曲 TreeLSTM。
- `PROTAC` 当前只有 PROTAC molecule 与组件 SMILES 的同塔对齐；还没有 POI pocket、E3 pocket、linker、ternary complex 的联合几何目标。
- `RCSB mmCIF` 当前只是文件路径索引；缺少可直接训练的链、残基、界面、配体/肽段、坐标 JSON contract。
- `PDCdb` 和 `SKEMPI2` 当前处理结果还不能直接进入 HUMU 三塔训练。
- `contrastive.negative_sampling` 配置写着支持 `hard_negative`，但 `_compute_losses` 当前只允许 `in_batch`。

### 方案 B：推荐方案，就地升级三塔 encoder 与训练目标

内容：

- 不创建新训练入口，不改外部 129 维输出。
- 让现有三塔真正消费配置中的 `hidden_dim`、`n_layers`、`n_heads`、`dropout`。
- Molecule tower 升级为 graph message passing + 可选 3D geometric branch，再投影到 Lorentz。
- Pocket tower 升级为 radius graph geometry encoder + ESM2 fusion，再投影到 Lorentz。
- Route tower 升级为 reaction step graph encoder + route tree pooling，再投影到 Lorentz。
- 在 pipeline 中加入 activity supervised objective、hard negatives、目标配额数据调度器、PROTAC component retrieval metrics。
- 对 RCSB/SKEMPI/PDC/PROTAC-8K 先补数据 contract；所有已处理数据集必须在训练或验证目标中有明确用途，不能只停留在 preflight 报告。

优点：

- 直接解决当前 HUMU 预训练与 CoreArchitecture v2 的核心结构偏差。
- 保持服务和 checkpoint 外部兼容。
- 可以分阶段合入，每阶段都有独立测试和 preflight。

缺点：

- 修改面覆盖三个 encoder、dataloader、loss、config 和测试。
- 训练成本会上升，需要通过 batch size、采样、缓存和 max sequence length 控制。

推荐选择方案 B。

## 详细设计

### 1. 配置与 encoder 构建

目标：

- 保留 `configs/models/humu_pretrain.yaml`，不新增 `humu_pretrain_v2.yaml`。
- `_build_encoders` 将 `encoders.mol`、`encoders.pocket`、`encoders.route` 的 `hidden_dim`、`n_layers`、`n_heads`、`dropout` 显式传入对应 encoder。
- encoder 内部完成最终 Lorentz projection，逐步减少 `_wrap_as_module` 的额外线性层职责；第一阶段可保留 wrapper 以兼容旧 checkpoint，后续只在确认 checkpoint 加载测试通过后改为 identity wrapper。

配置新增项只允许表达真实使用的行为：

```yaml
encoders:
  mol:
    hidden_dim: 256
    n_layers: 6
    n_heads: 8
    dropout: 0.1
    use_3d_geometry: true
    geometry_source: "rdkit_conformer"
  pocket:
    hidden_dim: 256
    n_layers: 4
    n_heads: 8
    dropout: 0.1
    radius_angstrom: 6.0
    max_neighbors: 32
    use_esm2: true
  route:
    hidden_dim: 256
    n_layers: 4
    n_heads: 8
    dropout: 0.1
    use_tree_pooling: true
```

如果某个配置项尚未在代码中真实使用，不能写入默认配置。

### 2. 最优数据分配方法

当前 `joint_oversample_factor=40` 是静态放大策略，只能缓解 `joint` 样本少的问题，不能处理多数据集、多目标之间的梯度占比。升级后废弃“单一倍数放大”作为主策略，改为 `TargetRatioMultiSourceBatchSampler`。

#### 核心原则

- 以训练目标为单位分配 batch，而不是以原始数据量为单位自然采样。
- 每个启用的数据集必须映射到一个或多个 objective。
- 每个 objective 在 batch 内先按有效样本求均值，再乘 `loss_weights`，防止大数据源天然主导总损失。
- epoch 使用固定 `steps_per_epoch`，不把 300 万级数据源完整扫完定义为一个 epoch。
- 小数据源允许 replacement，但必须记录 `source_repeat_rate` 和 `unique_source_coverage`。
- 采样比例先用目标配额启动，再根据 validation 指标做分阶段调整；不得再用固定 40 倍作为“最优”假设。

#### 初始目标配额

第一轮默认 batch 目标配额如下，后续只能基于验证指标调整：

| Objective | 初始配额 | 数据来源 | 作用 |
|---|---:|---|---|
| `mol_self` | 10% | `mol` | 小分子图/几何表示稳定性，防止只学习 paired 子集 |
| `mol_pocket` | 15% | `pocket` | 小分子-口袋对齐 |
| `mol_route` | 15% | `route` | 小分子-合成路线对齐 |
| `mol_pocket_route` | 20% | `joint` | 三塔联合对齐，替代原 `joint * 40` 主策略 |
| `activity_pair` | 15% | `activity`、`bindingdb_activity` | 同 target 活性排序、activity cliff 分离 |
| `protac_component` | 10% | `protacpedia`、`protacdb` | PROTAC、E3 ligand、target ligand、linker/component 对齐 |
| `protac_ternary` | 5% | `protac8k`、`protacpedia`、`rcsb_mmcif` | PROTAC target pocket、E3 pocket、linker/PROTAC 三元关系 |
| `protein_interface` | 5% | `rcsb_mmcif`、`interface_skempi2` | 大分子界面、突变亲和力、protein-protein 层级关系 |
| `route_template` | 3% | `retropath_templates`、`route_eval` | 反应模板和 route 泛化 |
| `pdc_component` | 2% | `pdcdb` | peptide-drug conjugate 的 peptide/linker/payload 组合关系 |

如果某个 objective 的数据合同尚未通过 preflight，该 objective 不能静默置空；训练启动必须 fail-fast，并说明缺失的是哪个数据合同。开发 smoke 可显式传 `enabled_objectives` 使用子集，但默认正式配置必须覆盖全部数据集。

#### 采样器行为

`TargetRatioMultiSourceBatchSampler` 每个 batch 按配额抽取 objective，再由 objective 选择对应 dataset shard。数据源内部使用温度采样：

```text
sample_probability(source_i) ∝ min(n_i, cap_i) ^ alpha
```

- `alpha=0` 表示各来源完全均衡。
- `alpha=1` 表示按 capped 数据量采样。
- 默认 `alpha=0.5`，兼顾覆盖率与数据规模。
- `cap_i` 防止 BindingDB、mol 这类百万级数据源完全支配 batch。

每个 epoch 输出以下统计：

- `objective_counts`
- `source_counts`
- `unique_source_coverage`
- `source_repeat_rate`
- `loss_by_objective`
- `retrieval_by_objective`

调配规则：

- 若某 objective 的 retrieval 或 margin 长期低于其他 objective，下一阶段最多提高 5 个百分点。
- 若小数据源 `source_repeat_rate` 过高且 validation 没有提升，降低该 objective 配额或增加增强方式，不能继续重复采样。
- 若 `collapse_ratio` 升高，优先降低单一大源配额并增加 hard negatives，而不是提高 batch size 掩盖问题。

### 3. 全数据集使用矩阵

| 数据集 | 第一轮用途 | 需要补齐的合同 | 对应 6.11 目标 |
|---|---|---|---|
| `mol` | `mol_self`，molecule graph/geometry 表示学习 | 已有 SMILES；可选 RDKit conformer | 小分子共享表示 |
| `pocket` | `mol_pocket`，pocket geometry + ESM2 fusion | 已有 pocket sidecar、coords、sequence | 靶点口袋表示 |
| `route` | `mol_route`，reaction step graph | 已有 reaction/root_smiles；多步 route 后增强 tree pooling | 合成路径表示 |
| `route_eval` | `route_template` 验证与 route 泛化评估 | 已有 valid/test route JSONL | 合成路径泛化 |
| `joint` | `mol_pocket_route` 三塔联合对齐 | 已有 ligand/pocket/route payload | 结构-口袋-路线联合空间 |
| `activity` | `activity_pair` 与 activity cliff validation | 已有 activity JSONL | 药效目标、activity cliff |
| `bindingdb_activity` | `activity_pair` 主活性监督 | 已有大规模 activity JSONL | 药效目标、活性排序 |
| `protacpedia` | `protac_component`，后续 `protac_ternary` | 已有 PROTAC/E3/ligand/linker SMILES；80 个 ligand PDB id 可连 RCSB | PROTAC 模块关系 |
| `protacdb` | `protac_component` 组件库、component type 对齐 | 需要构造稳定 anchor-component 或 component-only contract | E3 ligand、linker、warhead、分子胶/XTAC 组件 |
| `protac8k` | `protac_ternary` | 需要从 archive index 展开 target pocket、ligase pocket、target ligand、ligase ligand、PROTAC/features JSONL | PROTAC 三元复合体几何关系 |
| `rcsb_mmcif` | `protein_interface`、`protac_ternary` 结构来源 | 需要解析 chain、residue/atom coords、interface/pocket sidecar | 大分子、POI/E3、三元结构 |
| `interface_skempi2` | `protein_interface` mutation ranking | 需要把 PDB complex + mutation 映射到 WT/mutant interface payload 和 affinity label | 大分子界面与突变效应 |
| `pdcdb` | `pdc_component` | 需要 peptide sequence、linker SMILES、payload SMILES 或明确不可编码记录 | 大分子/肽药偶联物 |
| `retropath_templates` | `route_template` | 已有 template JSONL/TSV/CSV；需要接到 route encoder template objective | 合成模板与路线层级 |

该矩阵是完成定义的一部分。默认正式训练配置不能遗漏任何已处理数据集；若数据合同缺失，必须先实现合同并通过 preflight，再进入训练。

### 4. Molecule tower 升级

修改目标文件：

- `models/mf-encoders/humu_mol_encoder/src/mf_encoders/humu_mol/encoder.py`
- `models/mf-encoders/humu_mol_encoder/pyproject.toml`
- `tests/unit/test_humu_training.py`

设计：

- 保留 `HUMUMoleculeEncoder` 类名和 `encode/encode_batch/forward` 接口。
- 将当前 `_propagate` 替换为可配置的 graph message passing block：
  - atom feature projection
  - bond feature projection
  - message aggregation
  - residual + layer norm + dropout
  - LorentzAttention 或 tangent-space attention pooling
- 3D 分支只在有坐标或允许 RDKit conformer 生成时启用：
  - 输入 dict 有 `coords` 时直接使用。
  - 只有配置 `use_3d_geometry=true` 时才从 SMILES 生成 conformer。
  - conformer 生成失败必须返回明确错误或按配置跳过 batch，不能静默伪造坐标。
- 若引入 `e3nn`，必须在 `humu_mol_encoder/pyproject.toml` 中声明依赖，并运行 `uv lock`。如果不引入 `e3nn`，则只能声明为 E(3)-invariant geometry branch，不得声称 SE(3)-equivariant。

最小实现边界：

- 第一轮不要求完整等变张量特征，只要求 graph message passing 与显式 geometry branch 可训练、可测试。
- 保持输出形状为 `(batch, 129)` 且满足 Lorentz projection。

### 5. Pocket tower 升级

修改目标文件：

- `models/mf-encoders/humu_pocket_encoder/src/mf_encoders/humu_pocket/encoder.py`
- `models/mf-encoders/humu_pocket_encoder/pyproject.toml`
- `tests/unit/test_humu_training.py`

设计：

- 保留 `HUMUPocketEncoder` 类名和 `encode/encode_batch/forward` 接口。
- 基于 pocket atom coordinates 构建 radius graph：
  - `coords` shape 必须为 `(n_atoms, 3)`。
  - `elements` 和 `residue_types` 长度必须与坐标一致。
  - `radius_angstrom` 和 `max_neighbors` 必须真实生效。
- 用 geometry message passing 替代当前全局 pairwise 统计为主的特征：
  - node feature: element、residue type、hydrophobic/charge 类别。
  - edge feature: distance RBF、方向或相对坐标。
  - pooling: attention pooling 或 mean pooling。
- ESM2 fusion 保留：
  - inline `esm2_embedding` 优先。
  - 无 embedding 时按现有 ESM2 checkpoint 路径计算。
  - 仍执行 max sequence length 检查。
- 如果使用 e3nn，需要正式依赖声明；否则不能写“E(3)-GNN”，只能写“E(3)-invariant radius graph encoder”。

### 6. Route tower 升级

修改目标文件：

- `models/mf-encoders/humu_route_encoder/src/mf_encoders/humu_route/encoder.py`
- `models/mf-encoders/humu_route_encoder/pyproject.toml`
- `tests/unit/test_humu_training.py`

设计：

- 保留 `HUMURouteEncoder` 类名和接口。
- 将 route record 转换为 reaction step graph：
  - step node: reaction string、reactant count、product count、bond change、reaction type。
  - route edge: `parent_step_id` / `children` / `child_step_ids`。
  - 对当前 USPTO-MIT `steps=1` 的样本，构造单节点图。
- 第一阶段使用 Transformer encoder + tree-aware pooling：
  - token/feature projection
  - self attention
  - depth/branching feature injection
  - root pooling
- 只有在处理数据中出现多步 route tree 后，再实现双曲 TreeLSTM；当前 route manifest 和样本显示多数是一阶反应，直接做 TreeLSTM 没有足够数据收益。

### 7. 训练样本与采样

修改目标文件：

- `pipelines/humu_pretrain/src/humu_pretrain/data_loader.py`
- `tests/unit/test_humu_training.py`

设计：

- 保持 `PairedHUMUDataset` 为唯一 paired training dataset。
- 新增 pair_type 只在有真实 payload 时加入，但默认正式配置必须覆盖全数据集使用矩阵：
  - `mol_self`: 来自 `mol`，对同一分子的两个增强视图做分子自监督对齐。
  - `activity_pair`: 同一 target 下活性相近/差异大的 molecule pair，用于 supervised molecule objective。
  - `protac_component`: 已存在，继续使用。
  - `protac_ternary`: 来自 PROTAC-8K / PROTACpedia / RCSB 的 target pocket、E3 pocket、PROTAC/linker 组合。
  - `protein_interface`: 来自 RCSB/SKEMPI 的 WT/mutant interface 与 affinity label。
  - `route_template`: 来自 Retropath templates 和 route_eval。
  - `pdc_component`: 来自 PDCdb 的 peptide/linker/payload 合同。
- 引入目标配额采样：
  - 防止 BindingDB 3145942 条 activity 或 mol 2854636 条记录淹没 joint/protac 样本。
  - 每个 batch 需要记录 `pair_type_counts`，用于日志和验证。
- 删除 `joint_oversample_factor` 的主导地位。可以保留为 legacy 配置并在启用目标配额采样时忽略，最终从默认配置移除。

### 8. 损失函数升级

修改目标文件：

- `pipelines/humu_pretrain/src/humu_pretrain/pipeline.py`
- `tests/unit/test_humu_training.py`

设计：

- 保留现有 loss key：
  - `l_mol_pocket`
  - `l_mol_route`
  - `l_pocket_route`
  - `l_protac_component`
  - `l_curvature_reg`
- 新增真实可计算 loss key：
  - `l_mol_self`: molecule augmentation self-supervised contrastive objective。
  - `l_activity_supervised`: 同 target activity pair 的 supervised contrastive 或 margin ranking objective。
  - `l_hard_negative`: 基于当前 batch 或 memory queue 的 hard negative objective。
  - `l_protac_ternary`: POI pocket、E3 pocket、PROTAC/linker 的联合几何对齐 objective。
  - `l_protein_interface`: WT/mutant interface 与 affinity delta 的 ranking objective。
  - `l_route_template`: route embedding 与 reaction template 的对齐 objective。
  - `l_pdc_component`: peptide/linker/payload component 对齐 objective。
- `contrastive.negative_sampling` 要么真实支持 `hard_negative`，要么从配置中移除 `n_hard_negatives`，不能保留无效配置。
- 新增 loss 在向后兼容或开发 smoke 配置中可以默认权重为 0；正式 HUMU 预训练配置必须为全数据集使用矩阵中的 objective 显式配置非零权重。若配置启用某个 objective 但 batch 缺少对应 payload，训练必须 fail-fast。
- `total` 必须保持所有启用 loss 的加权和，validation 和 `_log_step` 同步显示。

### 9. 验证指标升级

修改目标文件：

- `pipelines/humu_pretrain/src/humu_pretrain/pipeline.py`
- `tests/unit/test_humu_training.py`

设计：

- 保留已有：
  - `retrieval_top1`
  - `positive_distance`
  - `negative_distance`
  - `distance_margin`
  - `embedding_variance`
  - `lorentz_norm_deviation`
  - `collapse_ratio`
  - `tree_distortion`
  - `cliff_separation_auroc`
- 新增：
  - `mol_pocket_retrieval_top1`
  - `mol_route_retrieval_top1`
  - `pocket_route_retrieval_top1`
  - `protac_component_retrieval_top1`
  - `protac_ternary_retrieval_top1`
  - `protein_interface_margin`
  - `activity_pair_margin`
  - `route_template_retrieval_top1`
  - `pdc_component_retrieval_top1`
  - `pair_type_counts`
  - `source_counts`
  - `source_repeat_rate`
  - `unique_source_coverage`
- 不声明 MOSES、PMO 或生成质量提升，除非后续真实运行对应 benchmark。

### 10. 大分子、PROTAC-8K 与 PDC 数据合同

为了满足 6.11 中“小分子、大分子、PROTAC、靶点口袋、E3 连接酶、linker 构象、三元复合体几何关系和合成路径统一到 HUMU”的目标，`rcsb_mmcif`、`interface_skempi2`、`pdcdb`、`protac8k` 不能只作为 preflight 报告存在。它们必须先补处理合同，再进入目标配额采样。

- RCSB mmCIF:
  - 解析 chain sequence。
  - 解析残基级或原子级坐标。
  - 输出 `protein_structure.jsonl` 或 pocket/interface sidecar。
  - 对 BioLiP/Propedia/SKEMPI/PROTAC 来源保留 source tag。
- SKEMPI2:
  - 将 `pdb_complex` 和 mutation 映射到可用结构。
  - 输出 wild-type interface 和 mutant interface payload。
  - 亲和力单位转换为数值 label。
- PDCdb:
  - 补 peptide sequence、linker SMILES、payload SMILES 或明确不可用状态。
- PROTAC-8K:
  - 从 archive index 展开 `target_pocket`、`ligase_pocket`、`target_ligand`、`ligase_ligand`、`protac` 和 `features`。
  - 输出 target pocket、E3 pocket、target ligand、E3 ligand、PROTAC、linker/features 的稳定配对。
- PROTAC ternary:
  - 需要 POI target、E3 ligase、PROTAC/linker、可用 pocket/structure 的稳定配对。
  - 合同通过前默认正式训练失败；开发 smoke 可以显式关闭 `protac_ternary`。

## 文件影响评估

### 主要修改文件

- `models/mf-encoders/humu_mol_encoder/src/mf_encoders/humu_mol/encoder.py`
  - 风险：高。改变核心 molecule embedding。
- `models/mf-encoders/humu_pocket_encoder/src/mf_encoders/humu_pocket/encoder.py`
  - 风险：高。涉及结构坐标、ESM2 和 batch 性能。
- `models/mf-encoders/humu_route_encoder/src/mf_encoders/humu_route/encoder.py`
  - 风险：中。现有 route 数据多为单步，需保证单步图仍可运行。
- `pipelines/humu_pretrain/src/humu_pretrain/data_loader.py`
  - 风险：高。影响 preflight、dataset、collate、validation split。
- `pipelines/humu_pretrain/src/humu_pretrain/pipeline.py`
  - 风险：高。影响训练、loss、validation、checkpoint。
- `configs/models/humu_pretrain.yaml`
  - 风险：中。只允许写入真实生效的配置。
- `tests/unit/test_humu_training.py`
  - 风险：低。扩展现有 HUMU 测试覆盖。

### 关联影响文件

- `models/mf-encoders/humu_mol_encoder/pyproject.toml`
- `models/mf-encoders/humu_pocket_encoder/pyproject.toml`
- `models/mf-encoders/humu_route_encoder/pyproject.toml`
- `uv.lock`
- `services/humu-encoder-svc/src/humu_encoder_svc/main.py`

`humu-encoder-svc` 默认不改。如果 encoder 构造参数需要从服务配置注入，必须单独审批。

### 不修改文件

- 不修改 CIG、HCIV、IntentCone、orchestrator、AMGE 生成器。
- 不更新 README，除非用户后续确认。
- 不新增 `intent_encoder`。
- 不新增并行 `humu_pretrain_v2` 文件。

## 实施阶段建议

### 阶段 1：全数据集合同与目标配额采样

目标：

- 为 `mol`、`pocket`、`route`、`route_eval`、`joint`、`activity`、`bindingdb_activity`、`protacpedia`、`protacdb`、`protac8k`、`rcsb_mmcif`、`interface_skempi2`、`pdcdb`、`retropath_templates` 建立统一 source registry。
- 补齐 `protac8k`、`rcsb_mmcif`、`interface_skempi2`、`pdcdb` 的可编码数据合同。
- 实现 `TargetRatioMultiSourceBatchSampler`。
- 实现 `objective_counts`、`source_counts`、`source_repeat_rate`、`unique_source_coverage` 日志和 validation 汇总。
- 默认正式配置必须覆盖全数据集使用矩阵；开发 smoke 可通过 `enabled_objectives` 使用子集。

验证：

```bash
uv run pytest tests/unit/test_humu_training.py::test_source_registry_requires_all_humu_datasets -q
uv run pytest tests/unit/test_humu_training.py::test_target_ratio_sampler_matches_configured_objective_mix -q
uv run pytest tests/unit/test_humu_training.py::test_default_config_rejects_missing_dataset_contracts -q
uv run pytest tests/unit/test_humu_training.py::test_training_batch_reports_source_coverage_stats -q
```

### 阶段 2：配置真实生效与测试基线

目标：

- encoder 构造真实消费 `hidden_dim`、`n_layers`、`n_heads`、`dropout`。
- 增加测试证明配置改变会改变 encoder 模块结构或参数数量。
- 保持现有 84 项 HUMU 单测通过。

验证：

```bash
uv run pytest tests/unit/test_humu_training.py tests/unit/test_learnable_curvature.py -q
PYTHONPYCACHEPREFIX=/tmp/mforge_pycache uv run python -m py_compile \
  models/mf-encoders/humu_mol_encoder/src/mf_encoders/humu_mol/encoder.py \
  models/mf-encoders/humu_pocket_encoder/src/mf_encoders/humu_pocket/encoder.py \
  models/mf-encoders/humu_route_encoder/src/mf_encoders/humu_route/encoder.py \
  pipelines/humu_pretrain/src/humu_pretrain/data_loader.py \
  pipelines/humu_pretrain/src/humu_pretrain/pipeline.py
```

### 阶段 3：Molecule 与 pocket 结构编码升级

目标：

- Molecule tower 加 graph message passing 和可选 3D geometry branch。
- Pocket tower 加 radius graph geometry encoder 和 ESM2 fusion。
- 保证输出 Lorentz 合法性、batch shape 和 invalid input fail-fast。

验证：

```bash
uv run pytest tests/unit/test_humu_training.py::test_molecule_encoder_consumes_geometry_branch -q
uv run pytest tests/unit/test_humu_training.py::test_pocket_encoder_uses_radius_graph_neighbors -q
uv run pytest tests/unit/test_humu_training.py::test_forward_paired_batch_computes_all_enabled_losses -q
```

### 阶段 4：Route graph/tree 编码升级

目标：

- 将 route record 转为 step graph。
- 单步 route 与多步 route 都能编码。
- 保持 `tree_distortion` 对无方差 route depth 返回空状态，不伪造深度。

验证：

```bash
uv run pytest tests/unit/test_humu_training.py::test_route_encoder_uses_step_graph_topology -q
uv run pytest tests/unit/test_humu_training.py::test_tree_distortion_remains_null_for_single_depth_routes -q
```

### 阶段 5：训练目标升级

目标：

- 实现 activity supervised objective。
- 实现 hard negative 或删除无效 hard negative 配置。
- 实现 `mol_self`、`protac_ternary`、`protein_interface`、`route_template`、`pdc_component` objective。
- 添加 per-pair retrieval 指标。

验证：

```bash
uv run pytest tests/unit/test_humu_training.py::test_mol_self_objective_uses_mol_source -q
uv run pytest tests/unit/test_humu_training.py::test_activity_supervised_loss_uses_same_target_pairs -q
uv run pytest tests/unit/test_humu_training.py::test_hard_negative_sampling_changes_negative_set -q
uv run pytest tests/unit/test_humu_training.py::test_protac_ternary_objective_uses_target_and_ligase_pockets -q
uv run pytest tests/unit/test_humu_training.py::test_protein_interface_objective_uses_skempi_affinity_delta -q
uv run pytest tests/unit/test_humu_training.py::test_route_template_objective_uses_retropath_templates -q
uv run pytest tests/unit/test_humu_training.py::test_pdc_component_objective_uses_peptide_linker_payload_contract -q
uv run pytest tests/unit/test_humu_training.py::test_validation_reports_per_pair_retrieval_metrics -q
```

### 阶段 6：真实配置 smoke 与 preflight

目标：

- 当前默认配置 `--preflight-only` 继续通过。
- `max_samples=2` smoke 能覆盖每个启用 objective 至少一个 batch。
- source coverage 报告必须包含全数据集使用矩阵里的每个数据集。
- 不声明训练质量提升，只报告链路通过。

验证：

```bash
timeout 180s uv run python pipelines/humu_pretrain/train.py \
  --config configs/models/humu_pretrain.yaml \
  --preflight-only

PYTHONPYCACHEPREFIX=/tmp/mforge_pycache uv run python - <<'PY'
from pathlib import Path
import sys
import yaml

ROOT = Path('/workspace/MForge/moleculeforge')
for rel in (
    'libs/mf-core/src',
    'libs/mf-humu/src',
    'models/mf-encoders/humu_mol_encoder/src',
    'models/mf-encoders/humu_pocket_encoder/src',
    'models/mf-encoders/humu_route_encoder/src',
    'pipelines/humu_pretrain/src',
):
    sys.path.insert(0, str(ROOT / rel))

from humu_pretrain.data_loader import create_dataloaders
from humu_pretrain.pipeline import _build_encoders, _forward_paired_batch
import torch

cfg = yaml.safe_load((ROOT / 'configs/models/humu_pretrain.yaml').read_text())
cfg['device'] = 'cpu'
cfg['max_samples'] = 2
cfg['batch_size'] = 8
cfg['data']['num_workers'] = 0
cfg['data']['shuffle'] = False
loaders = create_dataloaders(cfg)
batch = next(iter(loaders['paired']))
encoders = _build_encoders(cfg, torch.device('cpu'))
losses = _forward_paired_batch(encoders, batch, cfg)
print(sorted(losses.keys()))
print(float(losses['total'].detach().cpu()))
print(batch.get('source_dataset'))
PY
```

## 风险与缓解

- 风险：encoder 结构升级破坏旧 checkpoint 加载。
  - 缓解：保留 checkpoint key；`load_state_dict(strict=False)` 已存在，新增测试覆盖旧 state dict 恢复。
- 风险：PROTAC、joint、PDC、SKEMPI 与 BindingDB/mol 的数据量差异极大。
  - 缓解：使用 `TargetRatioMultiSourceBatchSampler`、固定 `steps_per_epoch`、source coverage 日志和 loss-by-objective 归一化，不再依赖 `joint_oversample_factor=40`。
- 风险：RDKit conformer 生成导致训练启动慢或失败。
  - 缓解：默认只在显式配置开启时使用；失败走清晰异常或 skip_bad_batches。
- 风险：ESM2 checkpoint 或序列超长导致 batch 失败。
  - 缓解：保留现有 max sequence length prefilter 和 manifest fast preflight。
- 风险：RCSB/SKEMPI/PDC/PROTAC-8K 合同不完整时强行训练会伪造大分子或三元复合体能力。
  - 缓解：默认正式配置要求这些合同通过；缺失时 fail-fast。只有开发 smoke 可以显式关闭对应 objective。
- 风险：引入 e3nn/torch-geometric 但未声明依赖。
  - 缓解：只要 production import 使用这些包，就必须修改对应 encoder `pyproject.toml` 并运行 `uv lock`。

## KISS 四问

1. 这是现实问题还是想象问题？
   - 现实问题。现有文档和代码均显示 HUMU 预训练还停留在三塔 baseline，且新数据已经接入但未充分利用结构/活性/宏分子信号。
2. 有没有更简单做法？
   - 有。先补齐所有已处理数据集的可编码合同，再用目标配额采样就地升级现有三塔和训练目标，不创建新入口，不改服务 API。
3. 会破坏什么？
   - 主要风险是 checkpoint 兼容、训练性能和 batch 合同。通过保持 key、默认关闭新增 loss、分阶段测试和 preflight 降低风险。
4. 当前项目真的需要这个功能吗？
   - 需要。用户明确要求在新数据接入验证后升级 HUMU 预训练，使其接近最初设想。

## 审批点

建议批准方案 B，并按阶段 1 到阶段 6 执行。阶段 1 必须先完成全数据集合同和目标配额采样，再进入 encoder 结构升级。这样可以保证每个已处理数据集都有明确训练或验证用途，并避免旧的 `joint * 40` 静态放大继续主导训练分布。

## 自检

- 已基于当前代码、manifest、样本字段和文档证据设计。
- 未把附件实验表述当作当前已实现结果。
- 未恢复 `intent_encoder`。
- 未创建并行 `v2` 训练入口。
- 未要求更新 README。
- 已将 `joint_oversample_factor=40` 从主策略降级为 legacy，方案主策略改为目标配额采样。
- 已要求每个已处理数据集都映射到训练或验证目标。
- 未承诺模型质量提升，只定义可验证的工程与训练链路目标。
