# Commercial Architecture (Planned)

> **阶段**：Phase 3 设计规划

## 多租户架构

```
┌──────────────────────────────────────────────┐
│  Tenant Router (API Gateway middleware)       │
│  JWT claim → tenant_id → namespace routing   │
└──────────────┬───────────────────────────────┘
               │
   ┌───────────┼───────────┐
   ▼           ▼           ▼
Tenant-A    Tenant-B    Tenant-C
(K8s ns)   (K8s ns)   (K8s ns)
   │           │           │
   ▼           ▼           ▼
PostgreSQL schema isolation (per-tenant)
MinIO bucket prefix (tenant-{id}/)
Milvus collection prefix (tenant_{id}_)
```

## 合规矩阵

| 法规 | 要求 | MoleculeForge 实现 |
|---|---|---|
| 21 CFR Part 11 | 电子签名 + 审计追踪 | Sigstore + CRG provenance chain |
| EU Annex 11 | 计算机化系统验证 | 不可变审计日志 + 定期备份 |
| GDPR | 数据删除权 + 处理记录 | 租户数据隔离 + DSR API |
| China DSL | 数据本地化 + 安全审查 | 中国区域 MinIO + 加密传输 |

## 计费系统

```
Usage Events (NATS)
        │
        ▼
Usage Aggregator (Redis Streams → TimescaleDB)
        │
        ▼
Billing Engine (CronJob: hourly rollup)
        │
        ▼
Invoice Generator (Stripe / custom)
```
