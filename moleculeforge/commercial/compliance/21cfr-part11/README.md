# 21 CFR Part 11 合规模块 — 待实施

## 已满足的要求（核心架构实现）
- ✅ 电子记录：所有分子推理链存 Neo4j
- ✅ 电子签名：Sigstore 为每个审计事件签名
- ✅ 审计追踪：`provenance-svc` 提供不可篡改的完整日志
- ✅ 访问控制：api-gateway 的 OIDC + RBAC

## 待实施
- 文档生成：`provenance-svc` REST API 输出 Part 11 格式 PDF 报告
- 周期性报告自动化
- 监管机构接口
