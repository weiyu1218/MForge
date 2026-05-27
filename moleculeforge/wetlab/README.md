# Wetlab Integration Layer (Reserved)

> **状态**：待实施（Phase 3）
> **预计启动**：2026-Q4

## 概述

Wetlab 层将 MoleculeForge 的虚拟分子设计结果连接到实体实验室自动化设备，实现设计→合成→测试的闭环。

## 预留子模块

| 模块 | 说明 | 依赖 |
|---|---|---|
| `xdl-compiler/` | 将 SSP 编译为 XDL 2.0 化学实验描述语言 | SSP schema |
| `sila2-adapter/` | SiLA2 实验室自动化标准适配器 | xdl-compiler |
| `hardware-drivers/` | 硬件驱动（Chemputer/Opentrons/Chemspeed/ECL/Strateos） | sila2-adapter |
| `eln-integrations/` | 电子实验记录本集成（Benchling/IDBS/Dotmatics） | REST API |

## 预留接口

| 接口 | 说明 |
|---|---|
| `POST /api/v1/wetlab/synthesis` | 提交合成任务（含 SSP + XDL） |
| `GET /api/v1/wetlab/synthesis/{id}` | 查询合成状态 |
| `POST /api/v1/wetlab/assay` | 提交生物活性测试 |
| `GET /api/v1/wetlab/results/{id}` | 查询测试结果（反馈到 HUMU） |
