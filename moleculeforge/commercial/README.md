# Commercial Layer (Reserved)

> **状态**：待实施（Phase 3）
> **预计启动**：2026-Q4

## 概述

商业化层提供多租户 SaaS 部署、计费、合规认证等功能。

## 预留子模块

| 模块 | 说明 |
|---|---|
| `multi-tenancy/` | 租户隔离（DB schema / K8s namespace per tenant） |
| `billing/` | 使用量计费（GPU 小时、Oracle 调用次数、存储量） |
| `compliance/21cfr-part11/` | FDA 21 CFR Part 11 电子记录/签名合规 |
| `compliance/eu-annex-11/` | EU GMP Annex 11 计算机化系统合规 |
| `compliance/gdpr/` | GDPR 数据保护合规 |
| `compliance/china-dsl/` | 中国数据安全法合规 |
| `customer-deployments/` | 客户私有化部署配置 |

## 计费模型（规划）

| 资源 | 单位 | 说明 |
|---|---|---|
| GPU 推理 | GPU-小时 | 生成器/Oracle 推理 |
| Oracle 调用 | 千次 | Boltz-2 / DiffDock-L / FEP |
| 存储 | GB-月 | MinIO 对象存储 |
| 用户席位 | 席位-月 | 每用户每月 |
