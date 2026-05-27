# CoreArchitecture v2 当前问题审阅 Todo

## 审阅范围

- 设计理念：`/workspace/MForge/MoleculeForge_CoreArchitecture_v2.md`
- 既有计划：`docs/todo/2026-05-13-06-20-corearchitecture-v2-completion-plan.md`
- 项目目录：`/workspace/MForge/moleculeforge`
- 旧后台训练日志：`logs/humu_pretrain/humu_4h200_20260515T033401Z.log`
- 当前后台训练日志：`logs/humu_pretrain/humu_4h200_fix_ddp_20260516T124616Z.log`
- 当前后台训练 PID：`logs/humu_pretrain/humu_4h200_fix_ddp_20260516T124616Z.pid`

## 当前结论

当前项目已经从纯空架构推进到部分 fail-fast 和真实数据预处理阶段，但还没有达到 CoreArchitecture v2 的可信闭环。HUMU 预训练已从独立三 loader 平均距离目标改为真实 paired dataset + in-batch contrastive loss，并完成一次正式四卡 HUMU 4h200 第 1 个 epoch 验证。旧训练日志仍只能作为异常诊断证据，当前 `humu_4h200_fix_ddp_20260516T124616Z` run 已进入 Epoch 2。

## 关键证据

- 当前分支：`feature/corearchitecture-v2-completion`，仓库仍无提交，所有文件处于 untracked 状态。
- 旧 HUMU 后台训练 PID：`3479878` 已通过 `pipelines/humu_pretrain/stop_humu_background.sh logs/humu_pretrain/humu_4h200_20260515T033401Z.pid` 停止；旧 run 仅保留为异常诊断证据。
- 当前 HUMU 后台训练 PID：`595054`，PID file `logs/humu_pretrain/humu_4h200_fix_ddp_20260516T124616Z.pid`，manifest `logs/humu_pretrain/humu_4h200_fix_ddp_20260516T124616Z.manifest.json`。
- 旧训练日志显示 Epoch 1 的 batch 1 到 50 内 loss 从 `3.3386` 降到 `0.1041`，仅完成 `50/2788` batch；该 run 仍按旧 loss 路径执行，只能作为异常诊断 run。
- 当前正式四卡 run 已完成 `Epoch 1/100: train_loss=4.9816, time=4881.1s`，随后进入 Epoch 2；复查日志未发现 `FAILED`、`Traceback`、`ValueError`、`ChildFailedError`、`Watchdog caught`、`RuntimeError`。
- 旧 run 每个 rank 下有约 48 个子进程，`num_workers=16` 分别用于 mol、pocket、route 三个 DataLoader，4 rank 总计约 192 个 worker；当前配置已降为 `num_workers: 4` 并改为单一 paired loader。
- `pipelines/humu_pretrain/src/humu_pretrain/pipeline.py` 中 `_compute_losses` 已改为 in-batch contrastive loss，使用 `contrastive.temperature` 和 `negative_sampling: in_batch`，不再计算 pocket-route 拉近项。
- `pipeline.py` 的训练循环已改为只消费 `paired` loader；CrossDocked pocket 使用 `ligand_smiles` 对齐 molecule tower，USPTO route 使用 `root_smiles` 对齐 molecule tower。
- `pipeline.py` 已修复 DDP tower 调用契约：pocket 和 route tower 在每个 rank 每个 batch 固定调用一次；空 tower batch 通过 zero-loss dummy forward 参与反传路径，避免不同 rank 条件调用 DDP-wrapped tower 导致 NCCL collective 顺序错位。
- 已处理数据存在：mol `2854636` 条、pocket `24242` 条、route `409035` 条、route_eval `70000` 条；当前 paired dataset 从 pocket 和 route 数据建立真实正样本契约。
- HUMU、index、feature store、ADMET、Dock、Boltz2、FEP、FTO、Supply、Retrosyn 生产路径已改为缺少真实 artifact/backend 时 fail-fast；`rg "np.random|torch.randn|random\\.Random|hash\\(" services models/mf-oracles models/mf-retrosyn` 当前无命中。
- `tests/e2e/test_kras_g12c_pilot.py` 和 `tests/e2e/test_audit_completeness.py` 仍全部 skip。
- `/workspace/mf-dki-bare` 存在，但不是当前项目内 Docker compose 入口；当前可验证入口仍是 `MForge/moleculeforge/infra/docker/docker-compose.test.yml`。
- `docker compose -f infra/docker/docker-compose.test.yml config` 通过；`up -d` 当前环境阻塞于 Docker layer 注册错误：`failed to register layer: unshare: operation not permitted`。

## P0-1：冻结当前 HUMU 训练结论，修正训练目标

目标：在继续消耗 H200 时间前，证明 HUMU 预训练目标不会通过表示坍塌获得低 loss。

涉及文件：

- `pipelines/humu_pretrain/src/humu_pretrain/pipeline.py`
- `pipelines/humu_pretrain/src/humu_pretrain/data_loader.py`
- `configs/models/humu_pretrain.yaml`
- `tests/unit/test_humu_training.py`
- `data/processing/humu_pretrain/manifest.json`

执行前问题：

- 当前 loss 是 `distance(mol, pocket) + distance(mol, route) + distance(pocket, route)`，优化目标直接鼓励所有 embedding 互相靠近。
- 三个数据源没有共同样本 id，按 batch index 拼接不代表真实正样本。
- 配置中声明的 `contrastive.temperature`、`negative_sampling`、`n_hard_negatives` 没有进入实现。
- 没有记录 embedding norm、batch 内距离方差、正负样本距离间隔、retrieval@k、collapse ratio。

Todo：

- [x] 停止把当前后台训练产物命名为可用 HUMU 权重；若保留该 run，只能标记为异常诊断 run。
- [x] 为训练数据建立真实配对样本契约：`mol_id`、`pocket_id`、`route_id`、`ligand_smiles`、`target_id`、`source_dataset`、`split`。
- [x] 将 CrossDocked pocket 的 `ligand_smiles` 与 molecule encoder 输入对齐，形成真实 mol-pocket 正样本。
- [x] 将 USPTO route 的 `root_smiles` 与 molecule encoder 输入对齐，形成真实 mol-route 正样本。
- [x] 实现 InfoNCE 或 supervised contrastive loss，至少包含 in-batch negatives。
- [x] 删除或禁用仅靠三塔平均距离互相拉近的训练目标。
- [x] 每个 log step 记录正样本距离、负样本距离、embedding 方差、Lorentz norm 偏差、collapse ratio。
- [x] 先跑单卡小样本 red-green：打乱配对后 loss 不应快速下降，恢复配对后 retrieval 指标应改善。
- [x] 四卡正式训练前先通过 1 epoch smoke 验收，并保存 manifest hash、配置、git 状态、日志和 checkpoint。

执行记录：

- 已新增单测覆盖真实 paired dataset、单 paired loader、in-batch contrastive loss、打乱配对惩罚、retrieval top-1 red-green 和无 pocket-route 拉近。
- 已将 `retrieval_top1` 纳入 HUMU loss stats；打乱配对时 top-1 retrieval 下降，恢复配对时 top-1 retrieval 为 1.0。
- 已执行单进程小样本 smoke：`max_samples=4`、`batch_size=4`、`epochs=1`、`device=cpu`、`num_workers=0`，2 个 batch 完成，最终 `train_loss=1.0465`，日志包含 Load/Fwd/Bwd 和 collapse 指标。
- 已尝试四卡 1 epoch smoke：`/tmp/humu_4gpu_smoke.yaml`、`NPROC_PER_NODE=4`、`CUDA_VISIBLE_DEVICES=0,1,2,3`、PID `3843410`、log `logs/humu_pretrain/humu_4gpu_smoke_20260515T081016Z.log`、manifest `logs/humu_pretrain/humu_4gpu_smoke_20260515T081016Z.manifest.json`；进程运行 7 分钟以上仅输出 NCCL 初始化，无 batch log 和 checkpoint，已用停止脚本清理，GPU 显存恢复 0 MiB。
- 已定位四卡 smoke 初始失败原因：DDP 下 batch 可能缺少某个 tower，且 DDP sampler 强制 shuffle，导致 tower 参数使用和 batch 组成不可控；已开启 `find_unused_parameters=True` 并让 DDP sampler 尊重原 loader 的 shuffle 配置。
- 已通过四卡 1 epoch ordered smoke：PID `3893878`，log `logs/humu_pretrain/humu_4gpu_smoke_ordered_20260515T084346Z.log`，manifest `logs/humu_pretrain/humu_4gpu_smoke_ordered_20260515T084346Z.manifest.json`，保存配置 `logs/humu_pretrain/humu_4gpu_smoke_ordered_20260515T084346Z.config.yaml`，manifest hash `08030c4d9746edce3e6324c3fe130b624c0ba7de14717af0942a3b87a57223a6`，config hash `6f4d823616011193bc5d97d450e374e1bdb12fa21e8d3d9621b147013b27ecf5`。
- 四卡 smoke 产物：`checkpoints/humu_4gpu_smoke/best_model.pt` hash `a068483c4b03476b198a8749268392cdc85a29bfd63f0a3f9b3bf6dc10826079`，`checkpoint_step_00000001.pt` hash `8de1eff90e056a3dbd545ef369b4208b89953a470b9301acfd1fd8596ddad625`，`checkpoint_epoch_0001.pt` hash `56adccaddb2824cb483bc1ad7674be98d8f3add77a1fa35675b0a742615e59f5`，`final_model.pt` hash `52675950ef294258658b38faa1bdcd00f47a2fdd058ff80764b3ad1bd0307f92`。
- 四卡 smoke 日志包含 Batch 1/1、`train_loss=1.0549`、Load/Fwd/Bwd、GPU/memory/util、collapse 指标；训练结束后 GPU 显存恢复 0 MiB。旧 PID `3479878` 已停止，旧 run 仅保留为异常诊断证据。
- 已修复真实数据中的 invalid SMILES 进入 molecule encoder 问题：paired collate 会过滤 RDKit 无法解析的 `ligand_smiles`，全无有效样本时 fail-fast；正式 run 已越过旧失败点 `Batch 144/424`。
- 已将 pocket 坐标 JSON 改为 lazy loading，避免 paired dataset 初始化时一次性加载全部 pocket 坐标。
- 已修复 DDP 条件 tower 调用导致的正式 run 最后 batch NCCL timeout：新增回归测试覆盖每个 tower 每 batch 单次调用、空 tower batch zero-loss dummy backward；小样本四卡 edge check `/tmp/humu_ddp_edge_check.yaml` 已完成 `Epoch 1/1`。
- 正式四卡 HUMU 4h200 run：PID `595054`，log `logs/humu_pretrain/humu_4h200_fix_ddp_20260516T124616Z.log`，manifest `logs/humu_pretrain/humu_4h200_fix_ddp_20260516T124616Z.manifest.json`；第 1 个 epoch 完成 `train_loss=4.9816`，`checkpoint_epoch_0001.pt` 和 `best_model.pt` 已在 `checkpoints/humu_4h200/` 写入，训练已进入 Epoch 2。

验收：

- 随机打乱 mol-pocket 或 mol-route 配对时，验证集 retrieval@k 明显退化。
- 正样本距离低于负样本距离，并且 batch 内 embedding 方差不接近 0。
- loss 下降必须伴随 retrieval、route consistency、tree distortion 或 activity cliff 指标改善。

## P0-2：修复 HUMU 训练进程与资源使用异常

目标：让训练过程可监控、可中断、可恢复，避免 GPU 显存占用但无有效计算。

涉及文件：

- `pipelines/humu_pretrain/run_humu_4h200_background.sh`
- `pipelines/humu_pretrain/src/humu_pretrain/pipeline.py`
- `configs/models/humu_pretrain.yaml`

执行前问题：

- 日志在 24 个 batch 后没有形成 epoch checkpoint。
- GPU 显存占用和 `pmon` 计算进程状态不一致。
- 每个 rank 启动 48 个 worker，4 rank 约 192 个 worker，数据加载开销过高。
- batch step 时间在 `69s` 到 `290s`，对当前输入规模不具备训练效率。

Todo：

- [x] 增加 heartbeat 日志：rank、epoch、step、dataloader wait time、forward time、backward time、GPU id、显存、util。
- [x] 将 `num_workers` 从 16 按 rank 降到可控值，先用 2 或 4 验证吞吐。
- [x] 避免为 mol、pocket、route 三个 loader 各自启动独立高 worker 池；改为单一 paired dataset loader。
- [x] 在 PID 文件旁写入 run manifest：命令、config path、config hash、start time、world size、output dir。
- [x] 增加安全停止脚本，只终止指定 PID 文件对应的 torchrun 进程树。
- [x] checkpoint 至少按固定 step 间隔保存诊断 checkpoint，不只等 epoch 结束。

执行记录：

- 旧 PID `3479878` 已通过安全停止脚本停止；脚本输出 `Force stopped process group: 3479878`。
- 正式 H200 四卡 run 已验证 heartbeat、batch timing、GPU stats、manifest、PID file 和 epoch checkpoint；`Batch 424/424` 的 backward time 为 `0.7s`，未复现上一轮 `600s` NCCL timeout。

验收：

- `tail -f` 能持续看到 heartbeat。
- `nvidia-smi pmon` 能看到对应 rank 的计算活动，或者日志明确处于 dataloader 阶段。
- 单 batch 时间降到可解释范围，并能拆分出数据加载和计算耗时。

## P0-3：同步并验证 `mf-dki-bare` Docker 变更

目标：让 DKI test stack 有一个当前项目内可复现、可验证的 Docker 入口。

涉及文件：

- `infra/docker/docker-compose.test.yml`
- `infra/docker/docker-compose.dki.yaml`
- `infra/docker/docker-compose.minimal.yml`
- `infra/docker/base/Dockerfile.*`
- `configs/services/*.yaml`

执行前问题：

- 当前 `/workspace/MForge/moleculeforge` 内没有定位到 `mf-dki-bare`。
- 当前 `infra/docker` 仍是项目内唯一可审阅 Docker 配置。
- 既有计划记录 Docker layer 注册曾被 `unshare: Operation not permitted` 阻塞，当前审阅未看到项目内已同步的替代方案。

Todo：

- [x] 将 `mf-dki-bare` 中已更新的 Docker 文件同步到项目内标准路径，或在 todo 中记录其绝对路径和使用方式。
- [x] 明确 test stack 的端口：Postgres `5433`、Neo4j `7475/7688`、Milvus `19531`、MinIO `9002`、NATS `4223`。
- [x] 在启动前执行端口占用检查，冲突时只处理对应历史进程。
- [x] 验证当前环境是否仍缺少 Docker layer 注册所需权限。
- [x] 若当前环境不能运行 Docker，输出不可运行的原始错误，并提供可运行环境要求。
- [ ] Dockerfile.base 需要验证 Python 3.12、uv venv、包安装和服务 entrypoint 是否真实可启动。

执行记录：

- `/workspace/mf-dki-bare` 已定位；当前项目内可验证 Docker 入口仍为 `infra/docker/docker-compose.test.yml`。
- `infra/docker/docker-compose.test.yml` 中 TimescaleDB 镜像已改为真实存在的 `timescale/timescaledb-ha:pg16`。
- 端口 `5433`、`7475`、`7688`、`19531`、`9002`、`4223` 启动前均为空闲。
- `docker compose -f infra/docker/docker-compose.test.yml config` 通过。
- `docker compose -f infra/docker/docker-compose.test.yml up -d` 当前阻塞于环境权限：`failed to register layer: unshare: operation not permitted`。
- 已执行 `docker compose -f infra/docker/docker-compose.test.yml down -v`，未发现残留容器。
- 已尝试 `docker build -f infra/docker/base/Dockerfile.base -t moleculeforge/base:codex-verify .` 验证 Python 3.12、uv venv 和包安装；legacy builder 仅输出 deprecation 提示，10 分钟无 build step 输出，已终止 PID `3917204`，未产生可验证镜像，因此 Dockerfile.base 启动验收仍未完成。
- 本轮再次执行 `docker compose -f infra/docker/docker-compose.test.yml config`，配置解析通过；再次执行 `timeout 120s docker build -f infra/docker/base/Dockerfile.base -t moleculeforge/base:codex-verify .`，legacy builder 仍只输出 deprecation 提示且 120 秒以上无 build step 输出，`timeout` 未能自动结束，已清理本轮启动的 `timeout` PID `4068119` 和 `docker build` PID `4068122`；Dockerfile.base 仍缺少真实构建验收证据。

验收：

- `docker compose -f infra/docker/docker-compose.test.yml up -d` 可启动 DKI test stack，或明确记录当前环境阻塞错误。
- Postgres、Neo4j、Milvus、MinIO、NATS 的健康检查能被测试读取。
- integration tests 不再通过无条件 skip 表示成功。

## P0-4：清理生产路径中的随机、hash、固定返回

目标：生产服务缺少真实后端时必须 fail-fast，不能返回看似成功的随机或固定结果。

涉及文件：

- `services/humu-encoder-svc/src/humu_encoder_svc/main.py`
- `services/humu-index-svc/src/humu_index_svc/main.py`
- `services/feature-store-svc/src/feature_store_svc/main.py`
- `services/admet-svc/src/admet_svc/main.py`
- `services/dock-svc/src/dock_svc/main.py`
- `services/boltz2-svc/src/boltz2_svc/main.py`
- `services/fep-svc/src/fep_svc/main.py`
- `services/fto-patent-svc/src/fto_patent_svc/main.py`
- `services/supply-oracle-svc/src/supply_oracle_svc/main.py`
- `services/retrosyn-svc/src/retrosyn_svc/main.py`

Todo：

- [x] HUMU encoder service 接入真实 HUMU checkpoint 和输入类型路由，缺 checkpoint 直接错误。
- [x] HUMU index service 接入 Milvus client，search/insert/stats 必须来自真实 collection。
- [x] Feature Store service 接入 Feast 或明确配置错误，删除 hash 特征生成。
- [x] ADMET/Dock/Boltz2/FEP service 接真实 runner 或 fail-fast。
- [x] FTO service 查询真实 patent index，不能默认 `patents_found: 0`。
- [x] Supply oracle 查询真实 catalog/index，不能用 hash 生成价格、库存、交期。
- [x] Retrosyn service 接真实 AiZynthFinder/RSGPT/UAlign runner，不能随机生成路线。

执行记录：

- HUMU encoder 已加载 `HUMU_CHECKPOINT_PATH` checkpoint 中的 `encoder_mol`、`encoder_pocket`、`encoder_route`，按 `input_type` 路由 molecule/pocket/route 输入并返回真实 embedding；缺 checkpoint 继续 fail-fast。
- HUMU index 已接入 Milvus client 路径，insert 要求显式 ids 并调用 `upsert`/`insert`，search/delete/stats 调用真实 client，不再返回 501 或伪 stats。
- Feature Store 已接入 `app.state.feast_store` 或 `feast.FeatureStore(repo_path=FEAST_REPO_PATH)`，online/batch/views/materialize 均委托 Feast client；缺 Feast repo 或缺 client 能力时明确错误。
- LaMGen generator 本体已删除 `torch.randn`、`hash()` 和固定 SMILES pool，改为委托真实 runner；缺 runner 直接错误。
- ADMET、Dock、Boltz2、FEP、FTO、Supply、Retrosyn 已改为真实 runner/index/catalog 缺失时 fail-fast。
- 扩展清理了同类服务伪结果路径：MMPT、CReM、HFM、LaMGen、FragFM、ICLM、EvoMol，以及 API Gateway 设计接口的默认固定 seed pool。
- 验收命令 `rg "np.random|torch.randn|random\\.Random|hash\\(" services models/mf-oracles models/mf-retrosyn --glob '!**/__pycache__/**'` 当前无命中。
- 已验证 `uv run pytest tests/unit/test_service_artifact_status.py -q` 通过 10 项；`uv run pytest tests/unit/test_generators.py::TestLaMGen3DGenerator -q` 通过 2 项；`uv run pytest tests/anti_degradation/test_no_degradation.py::TestAntiDegradation::test_p0_production_paths_do_not_generate_pseudo_results -q` 通过 1 项；对应 ruff 检查通过。

验收：

- 生产模式下全仓 `rg "np.random|torch.randn|random\\.Random|hash\\(" services models/mf-oracles models/mf-retrosyn` 不再命中伪结果路径。
- 缺少模型、索引、服务配置时返回明确错误，包含缺失配置名或 artifact path。

## P1-1：补齐 CIC 的真实语义解析和 learned HCIV

目标：CIC 默认路径从 heuristic/hash 过渡到真实 semantic parser 和 learned encoder。

涉及文件：

- `services/cig-compiler-svc/src/cig_compiler_svc/domain/compiler.py`
- `services/cig-compiler-svc/src/cig_compiler_svc/domain/stages/stage1_semantic.py`
- `services/cig-compiler-svc/src/cig_compiler_svc/domain/stages/stage1b_grounding.py`
- `services/cig-compiler-svc/src/cig_compiler_svc/domain/hciv_encoder.py`
- `models/mf-encoders/humu_intent_encoder/src/mf_encoders/humu_intent/encoder.py`

问题：

- `CIGCompiler` 已切到默认 `production_real` + `learned`，`local_demo` 才使用 `_heuristic_extract`、`hash`、`random`。
- `ground_knowledge` 已支持 UniProt、PDB、ChEMBL、SureChEMBL evidence 聚合，并要求每条 evidence 带 `source_timestamp`。
- learned HCIV 已从 `HCIV_CHECKPOINT_PATH` 加载 state_dict，checkpoint 缺失会直接错误；checkpoint id、feature schema version 和 provenance 仍未补齐。

Todo：

- [x] 增加 semantic parser adapter，区分 `production_real` 和 `local_demo`。
- [x] 生产路径要求 LLM/tool-call parser 配置齐全，缺失直接错误。
- [x] grounding evidence 覆盖 UniProt、PDB、ChEMBL、SureChEMBL，并记录 source timestamp。
- [x] learned HCIV 必须从 checkpoint 加载，checkpoint 缺失直接错误。
- [x] 保留 hash/random 仅用于显式 demo/test mode。

验收：

- KRAS G12C 输入生成含 target、activity、ADMET、FTO、synthetic constraints 的 CIG。
- learned HCIV 输出附带 checkpoint id、feature schema version 和 provenance。

执行记录：

- 已新增 `CompilerMode`，默认 `production_real`，显式 `local_demo` 才允许 heuristic/hash/random。
- 已新增 `ProductionSemanticParserAdapter`，生产路径缺 `CIG_SEMANTIC_PARSER_URI` 直接错误。
- 已新增 UniProt、RCSB PDB、ChEMBL、SureChEMBL grounding tools；生产 `CIGCompiler` 默认使用四源，`local_demo` 默认只走 UniProt。
- 已新增四源 grounding evidence 测试，要求 UniProt、PDB、ChEMBL、SureChEMBL evidence 均包含 `source_timestamp`。
- 已新增 learned HCIV checkpoint 加载，缺 `HCIV_CHECKPOINT_PATH` 或 checkpoint 不存在直接错误。
- 已删除 CIG service `Compile` 的固定模拟 CIG 和固定 `parse_confidence` 返回，改为调用 `CIGCompiler`；`Refine` 未配置 runner 时直接错误。
- 已验证 `uv run pytest tests/unit/test_cic_compiler.py tests/unit/test_cig_service.py tests/integration/cic/test_cic_end_to_end.py -q` 通过 43 项。
- 已验证 `uv run pytest tests/anti_degradation/test_no_degradation.py::TestAntiDegradation::test_p0_production_paths_do_not_generate_pseudo_results -q` 通过 1 项。
- 已验证相关文件 `ruff check --select F,I,E501` 通过。

## P1-2：生成器最小真实闭环

目标：先完成 HFM-3D、FragFM、CReM-3D 的最小真实生成，不同时铺开所有生成器。

涉及文件：

- `models/mf-generators/hfm_3d/src/mf_generators/hfm_3d/generator.py`
- `models/mf-generators/fragfm/src/mf_generators/fragfm/generator.py`
- `models/mf-generators/crem_3d/src/mf_generators/crem_3d/generator.py`
- `models/mf-generators/evomol_rl/src/mf_generators/evomol_rl/generator.py`
- `services/hfm-generator-svc/src/hfm_generator_svc/main.py`
- `services/fragfm-generator-svc/src/fragfm_generator_svc/main.py`
- `services/crem-generator-svc/src/crem_generator_svc/main.py`

问题：

- HFM-3D 生产路径已 fail-fast，但还没有 checkpoint、decoder artifact、3D conformer 输出。
- FragFM、CReM-3D、EvoMol 仍使用随机 fragment 或固定 SMILES pool。
- 服务层仍返回随机 score，不调用模型实现。

Todo：

- [x] HFM-3D 训练或加载真实 decoder artifact，输出合法 SMILES、3D conformer、latent 和 provenance。
- [x] FragFM 接真实 fragment vocabulary、组装规则和 validity check。
- [x] CReM-3D 接真实 matched molecular pair 或 pharmacophore mutation 数据。
- [x] 生成器服务调用对应模型对象，不再服务层随机返回。
- [x] TAR/KD 只能消费真实 Oracle feedback，不能用字符串启发式 teacher 作为生产路径。

验收：

- 生成候选可被 RDKit 解析并带 3D conformer。
- provenance 记录 generator name、checkpoint、input cone、sampling seed、decode artifact。

执行记录：

- 已为 `HFM3DGenerator` 增加 `decoder_path`，从 JSON decoder artifact 读取带 latent 的 SMILES 条目，生产生成要求 checkpoint、decoder artifact 和 `sampling_seed`，按 latent 最近邻解码。
- HFM-3D 输出现在包含 RDKit 3D conformer `sdf_bytes`，并在 metadata 记录 `generator_name`、`checkpoint`、`decode_artifact`、`decoder_entry_id`、`sampling_seed`、`input_cone`、`latent`。
- HFM generator service runtime requirement 已同步为 `HFM_CHECKPOINT_PATH` + `HFM_DECODER_PATH`，`models/artifacts/manifest.json` 已补 `hfm_checkpoint`。
- 已验证 `uv run pytest tests/unit/test_humu_training.py::test_hfm3d_generator_save_load tests/unit/test_humu_training.py::test_hfm3d_generate_smoke tests/unit/test_humu_training.py::test_hfm3d_production_requires_decoder_artifact tests/unit/test_humu_training.py::test_hfm3d_generation_uses_decoder_artifact_conformer_and_provenance -q` 通过 4 项。
- 已将 `FragFMGenerator` 生产默认改为必须提供 `vocab_path`，从 JSON artifact 读取 `fragments` 和 `assembly_rules`，所有 product 经 RDKit canonical/validity check 后输出，并记录 `generator_name`、`fragment_vocabulary`、`assembly_rule_id`。
- 已将 `CReM3DGenerator` 生产默认改为必须提供 `mmp_db_path`，从 JSON artifact 读取 `mutations`，按 `seed_smiles` 过滤真实 mutation，所有 product 经 RDKit canonical/validity check 后输出，并记录 `generator_name`、`mmp_database`、`mutation_id`。
- 已验证 `uv run pytest tests/unit/test_generators.py::TestFragFMGenerator tests/unit/test_generators.py::TestCReM3DGenerator -q` 通过 6 项；对应 ruff 检查通过。
- 已为 HFM、FragFM、CReM 三个 generator service 增加模型对象注入路径；artifact 校验通过且 generator 已配置时，`Generate()` 调用 `generator.generate()` 并序列化真实模型输出，不在服务层生成随机 score 或候选。
- 已验证 `uv run pytest tests/unit/test_service_artifact_status.py -q` 通过 5 项；`uv run ruff check --select F,I,E501 services/hfm-generator-svc/src/hfm_generator_svc/main.py services/fragfm-generator-svc/src/fragfm_generator_svc/main.py services/crem-generator-svc/src/crem_generator_svc/main.py tests/unit/test_service_artifact_status.py` 通过。
- 已将 `CrossParadigmKDLayer` 默认模式设为 `production_real`，`update_teacher_scores()` 只接受带 `oracle_name` 和 `normalized_score` 的 Oracle feedback；直接传 SMILES 字符串会抛出 `TypeError`。
- 已保留显式 `local_demo` 模式下的 `WeakTeacher`，避免启发式 teacher 成为生产默认路径。
- 已验证 `uv run pytest tests/unit/test_cross_paradigm_kd.py -q` 通过 9 项；`uv run ruff check --select F,I,E501 libs/mf-core/src/mf_core/routing/cross_paradigm_kd.py tests/unit/test_cross_paradigm_kd.py` 通过。

## P1-3：真实 Oracle Cascade

目标：让 L0-L2 至少形成真实可运行筛选链，L3/L4 作为显式 slow/gpu 路径。

涉及文件：

- `models/mf-oracles/admet_ai/src/mf_oracles/admet_ai/oracle.py`
- `models/mf-oracles/diffdock_l/src/mf_oracles/diffdock_l/oracle.py`
- `models/mf-oracles/boltz2/src/mf_oracles/boltz2/oracle.py`
- `models/mf-oracles/openfe/src/mf_oracles/openfe/oracle.py`
- `services/admet-svc/src/admet_svc/main.py`
- `services/dock-svc/src/dock_svc/main.py`
- `services/boltz2-svc/src/boltz2_svc/main.py`
- `services/fep-svc/src/fep_svc/main.py`

Todo：

- [x] L0 接 RDKit descriptor 和真实 ADMET 模型，输出 uncertainty。
- [x] L1 接 GNINA 或 DiffDock runner，记录 input artifact hash 和 stderr path。
- [x] L2 接 Boltz runner，记录 model version 和 runtime。
- [x] L3 接 OpenFE runner，缺依赖时明确跳过 slow path。
- [x] Oracle cascade 实现升级策略：L0 过滤、L1 docking、L2 affinity、L3 RBFE。

验收：

- KRAS Pilot 至少跑通 L0-L2。
- 每个 Oracle 输出都可追溯到输入 artifact、模型版本、配置和日志。

执行记录：

- `ADMETAIOracle` 现在先用 RDKit 生成 `mol_wt`、`logp`、`tpsa`、HBD/HBA、rotatable bonds、ring count、QED、Lipinski violations descriptor rows，再交给 ADMET runner；`predict_with_uncertainty()` 返回 runner 提供的不确定性。
- `GninaOracle` 和 `DiffDockLOracle` 要求 runner 结果包含 `input_artifact_hash` 与 `stderr_path`；缺字段直接失败。
- `Boltz2Oracle` 已调整为 L2，并要求 runner 结果包含 `model_version` 与 `runtime_ms`；缺字段直接失败。
- `OpenFEOracle(skip_when_unavailable=True)` 在缺 runner 时返回显式 skipped slow path 和 skip reason，不伪造 RBFE。
- `ValidationAgent` 已实现注入式 Oracle cascade：L0 按 `admet_score` 阈值过滤，L0 失败停止升级；L0 通过后按 L1 docking、L2 affinity、L3 RBFE 执行，L3 skip 不视为伪通过。
- 已验证 `uv run pytest tests/unit/test_l0_oracle.py tests/unit/test_validation_agent.py -q` 通过 17 项；对应 ruff 检查通过。

## P1-4：DKI 服务与测试从 skip 变为真实读写

目标：Postgres、Neo4j、Milvus、MinIO、NATS 成为实际数据层，而不是文档中的目标。

涉及文件：

- `libs/mf-core/src/mf_core/db/*`
- `services/humu-index-svc/src/humu_index_svc/main.py`
- `services/feature-store-svc/src/feature_store_svc/main.py`
- `tests/integration/test_dki_postgres.py`
- `tests/integration/test_dki_neo4j.py`
- `tests/integration/test_dki_milvus.py`

Todo：

- [ ] Postgres fixture 使用 `TEST_DATABASE_URL` 或 test stack DSN，执行真实 CRUD。
- [ ] Neo4j fixture 使用 `NEO4J_URI`、`NEO4J_USER`、`NEO4J_PASSWORD`，写入 CRG 边并查询。
- [ ] Milvus test 使用真实 collection insert/search/delete，验证维度和 metric。
- [ ] MinIO artifact store test 覆盖 put/get/hash。
- [ ] NATS event test 覆盖 publish/subscribe 和 trace_id 传播。

验收：

- integration tests 没有无条件 skip。
- DKI 服务健康检查返回真实 backend 状态和版本。

执行记录：

- 已将 PostgreSQL integration fixture 改为读取 `TEST_DATABASE_URL`；变量存在时创建 ORM schema 并对 `molecules`、`runs`、`oracle_calls`、`pareto_fronts` 执行真实写入/查询。
- 已将 Neo4j integration fixture 改为读取 `NEO4J_URI`、`NEO4J_USER`、`NEO4J_PASSWORD`；变量存在时通过 `GraphRepository` 写入 `TRANSFORMS_TO`、`COVERED_BY`、`PRODUCED`、`HAS_BELIEF` 并查询。
- 已将 Milvus integration fixture 改为读取 `MILVUS_HOST`/`MILVUS_PORT`；变量存在时创建真实 collection，执行 insert/search/delete/drop，并验证 `z_humu` 129 维向量字段和 IP metric。
- 已把 `MinIOStorageClient` 从 stub 改为 aiobotocore S3-compatible put/get/head 封装，新增 `object_sha256()`；单测覆盖 put/get/hash。
- 已把 `NATSBus` 改为连接失败或超时时回退到进程内 pub/sub，单测覆盖 publish/subscribe 中 `trace_id` 传播。
- 当前环境没有 `TEST_DATABASE_URL`、`NEO4J_URI`、`MILVUS_HOST`、`MINIO_ENDPOINT_URL`、`NATS_URL`，因此 `uv run pytest tests/integration/test_dki_postgres.py tests/integration/test_dki_neo4j.py tests/integration/test_dki_milvus.py -q` 结果为 8 skipped，不能作为真实后端通过证据；本节 checkbox 暂不勾选。
- 已验证 `uv run pytest tests/unit/test_vector_store.py tests/unit/test_object_store.py tests/unit/test_nats_bus.py -q` 通过 13 项；对应 ruff 检查通过。
- 本轮复查环境变量仍为空：`TEST_DATABASE_URL`、`NEO4J_URI`、`MILVUS_HOST`、`MINIO_ENDPOINT_URL`、`NATS_URL` 均未设置；再次运行 DKI integration 结果仍为 8 skipped，因此 Postgres/Neo4j/Milvus/MinIO/NATS 真实后端项继续保持未完成。

## P1-5：MARB、CRG、Provenance 与 E2E 解锁

目标：让 orchestrator、审计链、trace 和 E2E 测试成为完成证据。

涉及文件：

- `agents/orchestrator/src/orchestrator/workflow/graph_builder.py`
- `services/orchestrator-svc/src/orchestrator_svc/main.py`
- `services/provenance-svc/src/provenance_svc/*`
- `libs/mf-telemetry/src/mf_telemetry/tracing/opentelemetry.py`
- `tests/e2e/test_kras_g12c_pilot.py`
- `tests/e2e/test_audit_completeness.py`

问题：

- graph builder 返回 dict，不是 LangGraph `StateGraph`。
- orchestrator status 是固定统计。
- provenance 的 Sigstore 集成仍有本地 hash 模拟痕迹。
- E2E 全部 skip，不能证明核心架构完成。

Todo：

- [x] 按锁定版本和官方文档实现真实 LangGraph 状态机。
- [x] 状态迁移覆盖 PLANNING、GENERATING、VALIDATING、REFINING、ESCALATING。
- [ ] CRG 写 Neo4j，belief/event 写 Postgres，artifact 写 MinIO。
- [ ] OpenTelemetry trace_id 从 API Gateway 贯穿 generator、Oracle、FTO、Retrosyn、SRB。
- [x] Sigstore 不可用时显式 `local_dev_signature`，不能伪称 Sigstore。
- [ ] 解锁 KRAS Pilot 和 Audit completeness E2E，缺依赖时使用环境标记而非无条件 skip。

验收：

- 每个 pipeline step 至少产生一个可验证 AuditEvent。
- E2E 输出 run_id、trace_id、artifact hash、模型版本和服务版本。

执行记录：

- 已将 orchestrator `WorkflowGraph.build()` 从 dict 改为真实 `langgraph.graph.StateGraph(...).compile()`，并实现 PLANNING、GENERATING、VALIDATING、REFINING、ESCALATING 状态迁移；VALIDATING 失败时按 `max_refinements` 路由到 REFINING 或 ESCALATING。
- 已新增 MVP 测试覆盖 LangGraph `ainvoke()` 和验证失败后升级到 ESCALATING。
- 已将 Sigstore unavailable fallback 改为显式 `local_dev_signature`；不再伪造 Fulcio certificate 或 Rekor URL，`get_rekor_entry()` 对本地签名返回 `None`。
- 已将 KRAS Pilot 和 Audit completeness E2E 的无条件 `pytest.skip()` 改为 `RUN_KRAS_G12C_E2E`/`RUN_AUDIT_E2E` 环境标记；当前环境未设置标记，运行结果仍为 11 skipped，不能作为 E2E 完成证据。
- 已验证 `uv run pytest tests/test_mvp_pipeline.py::TestMVPPipeline::test_orchestrator_graph tests/test_mvp_pipeline.py::TestMVPPipeline::test_orchestrator_graph_escalates_after_validation_failure -q` 通过 2 项，`uv run pytest tests/unit/test_provenance.py -q` 通过 11 项；对应 ruff 检查通过。
- 本轮复查 `RUN_KRAS_G12C_E2E` 和 `RUN_AUDIT_E2E` 仍未设置；再次运行 `uv run pytest tests/e2e/test_kras_g12c_pilot.py tests/e2e/test_audit_completeness.py -q` 结果为 11 skipped，因此 E2E 解锁项继续保持未完成。

## P1-6：FTO、供应链、Retrosyn、SRB 真实路径

目标：候选分子必须有真实专利、供应链和合成路线证据，SRB 不能从固定反应类型生成。

涉及文件：

- `pipelines/patent_indexing/src/patent_indexing/pipeline.py`
- `pipelines/reaction_indexing/src/reaction_indexing/pipeline.py`
- `services/fto-patent-svc/src/fto_patent_svc/main.py`
- `services/supply-oracle-svc/src/supply_oracle_svc/main.py`
- `services/retrosyn-svc/src/retrosyn_svc/main.py`
- `models/mf-retrosyn/*/src/*`
- `agents/srb_agent/src/srb_agent/compiler.py`
- `wetlab/xdl-compiler/src/xdl_compiler/compiler.py`

Todo：

- [x] 补 SureChEMBL、Google Patents、Enamine REAL、Reaxys 或替代离线数据路径。
- [x] Patent indexing 支持真实结构、patent id、claim evidence 和 vector index 写入。
- [x] Reaction indexing 从 USPTO/RetroPath 产出可复现 reaction template manifest。
- [x] Retrosyn runner 输出真实 reaction、reactants、conditions、building blocks。
- [x] SRB 消费 retrosyn route，不再轮询固定 `REACTION_TYPES`。
- [x] XDL step 必须能追溯到 SSP step id 和 retrosyn route step id。

验收：

- FTO verdict 有 patent evidence。
- Supply result 有 catalog source 和 timestamp。
- SSP/XDL 每一步可追溯到 retrosyn route。

执行记录：

- 已将 `compile_ssp()` 改为要求 `retrosyn_route.route_id` 和 `retrosyn_route.steps`，并从 route step 的 `reaction`、`reaction_type`、`reactants`、`reagents`、`conditions`、`yield`、`yield_uncertainty`、`purification` 生成 SSP，不再用固定 `REACTION_TYPES` 轮询生成步骤。
- SSP step 的 `parameters` 现在记录 `retrosyn_route_step_id` 和 `retrosyn_reaction`。
- XDL compiler 现在在每个 XDL step attributes 中写入 `ssp_step_id`，并透传 `retrosyn_route_step_id`。
- 已验证 `uv run pytest tests/unit/test_srb_agent.py -q` 通过 11 项；对应 ruff 检查通过。
- Patent indexing 现在解析 `smiles`、`patent_id`、`claim_evidence`、`source`，通过 `humu_encoder` 生成 `z_humu` 后写入 Milvus insert/upsert；`search_patent_similarity()` 调用真实 search client，不再固定返回低风险。
- Reaction indexing 现在保留 USPTO/RetroPath template source provenance，并在配置 `manifest_path` 时写出包含 `source_hashes`、`template_smarts`、`content_sha256` 的可复现 manifest。
- FTO service 支持 `PATENT_INDEX_URI=file://...` 离线专利索引，`SearchPatents()` 返回 `patent_evidence`、claim evidence 和 source；Supply service 支持 `SUPPLY_CATALOG_URI=file://...` 离线 catalog，返回 catalog source、timestamp、price 和 lead time。
- AiZynth、RSGPT、UAlign wrappers 现在校验 runner 返回的 route 必须包含 `route_id`、非空 `steps`，每个 step 必须包含 `reaction`、`reactants`、`conditions`、`building_blocks`。
- 已验证 `uv run pytest tests/unit/test_indexing_pipelines.py -q` 通过 9 项，`uv run pytest tests/unit/test_service_artifact_status.py -q` 通过 7 项；对应 ruff 检查通过。

## P2-1：模型权重、依赖和 artifact 清单

目标：把缺失依赖从隐性失败变成启动前可检查的 artifact manifest。

Todo：

- [x] 建立 `models/artifacts` 或现有 checkpoint 目录的统一 manifest，记录 HUMU、HFM、FragFM、ADMET、Boltz、OpenFE 等 artifact。
- [x] 每个服务启动时校验所需 artifact，不存在则拒绝启动。
- [x] 对命令行工具和 Python 包做启动前检查：GNINA、Boltz、OpenFE、OpenBabel、RDKit、Sigstore。
- [x] 区分系统 PATH 工具和 `.venv/bin` 工具，避免误判工具不存在或服务启动后找不到。

执行记录：

- 已新增 `models/artifacts/manifest.json`，记录 HUMU、HFM、FragFM、CReM、MMPT、LaMGen、ICLM、EvoMol、ADMET、Boltz、DiffDock、FTO、Supply、Retrosyn、Feast、Milvus 等 artifact/backend 要求。
- 已新增 `mf_core.artifacts`，支持 file、directory、path、uri、PATH executable、env executable、Python package 的统一检查。
- 已将启动前 runtime requirement 接入 HUMU encoder、HUMU index、Feature Store、ADMET、Dock、Boltz2、FEP、HFM、FragFM、CReM、MMPT、LaMGen、ICLM、EvoMol、FTO、Supply、Retrosyn 服务。
- Feature Store 和 HUMU Index 的 `/health` 返回结构化 `artifact_status`；gRPC 服务提供 `runtime_status()` 并在 `serve()` 前拒绝缺失依赖。
- 已补单测覆盖 artifact 缺失、存在文件、URI、PATH 工具、Python package、manifest 加载、服务 artifact status。

验收：

- 任何模型服务的 `/health` 包含 artifact status。
- 缺 artifact 的服务不能返回伪健康。

## P2-2：代码清理与质量门

目标：让代码库从调试开发状态转为可审阅状态。

Todo：

- [x] 清理已生成的 `__pycache__`、`.pytest_cache`、`.ruff_cache`，并确认 `.gitignore` 覆盖。
- [x] anti-degradation test 扩展到 service/model/pipeline 生产路径，阻断随机、hash、fixed pool、placeholder。
- [x] 统一生产模式和 demo/test mode 命名，避免默认走 demo。
- [x] 所有新增 TODO 都必须关联本文件或 issue，不在业务代码中写历史记录式注释。

执行记录：

- 已清理本次测试和历史测试产生的 `__pycache__`、`.pytest_cache`、`.ruff_cache`。
- 已增加 P0 生产路径 anti-degradation 覆盖，阻断 `np.random`、`torch.randn`、`torch.rand`、`random.Random`、`random.gauss`、`random.random`、`hash(` 伪结果路径。
- 已验证 `rg -n "TODO|FIXME" services libs models pipelines agents --glob '!**/__pycache__/**' --glob '!**/.venv/**'` 当前无命中。
- 本轮清理了 `services`、`libs`、`models`、`pipelines`、`agents`、`tests`、`wetlab` 下的 `__pycache__`，并删除 `.pytest_cache`、`.ruff_cache`；`rg -n "TODO|FIXME" services libs models pipelines agents tests --glob '!**/__pycache__/**' --glob '!**/.venv/**'` 当前无命中。
- 本轮 `rg -n "placeholder|Simulate|hash\\(|torch\\.randn|random\\.Random|np\\.random"` 在当前处理的生产路径中只剩 `models/mf-generators/lamgen_3d/src/mf_generators/lamgen_3d/model/multi_target_attention.py:9` 的 `nn.Parameter(torch.randn(...))`，该项是模型参数初始化，不是服务或生成器输出伪结果；LaMGen generator 输出路径已由 anti-degradation 测试覆盖并通过。

验收：

- `rg "TODO|FIXME|placeholder|Simulate|random|hash\\("` 的生产路径命中项被分类处理。
- 基线测试、integration tests、E2E 的 skip 数量被显式记录并持续下降。

## 建议执行顺序

1. P0-1：先修 HUMU 训练目标，否则当前训练越久越难判断权重价值。
2. P0-2：修进程、日志和资源可观测性。
3. P0-3：同步 `mf-dki-bare` Docker 并验证 DKI test stack。
4. P0-4：生产路径随机/hash/fixed 返回全部 fail-fast。
5. P1-1 到 P1-6：按 CIC、生成器、Oracle、DKI、MARB/E2E、FTO/SRB 顺序补真实闭环。
6. P2-1 到 P2-2：补 artifact manifest 和清理质量门。

## Linus 四问

1. 这是现实问题还是想象问题？
   - 是现实问题。loss 异常、训练目标坍塌风险、服务随机返回、E2E skip 都有实际文件和日志证据。
2. 有没有更简单的做法？
   - 有。先把 HUMU 训练目标和最小真实闭环修正，不同时扩展全部生成器和全部 Oracle。
3. 会破坏什么？
   - 主要风险是生产路径 fail-fast 后 demo 体验变差。处理方式是保留显式 `local_demo`，生产默认必须真实或失败。
4. 当前项目真的需要这个功能吗？
   - 需要。CoreArchitecture v2 的核心价值依赖 joint manifold、真实 Oracle、DKI、FTO 和审计链，不是单个 demo 能替代的。
