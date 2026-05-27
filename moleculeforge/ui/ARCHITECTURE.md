# UI Architecture (Planned)

> **阶段**：Phase 2 设计规划
> **依赖**：`services/api-gateway`

## 架构概览

```
┌─────────────────────────────────────────────────────┐
│  Browser (Next.js 14)                               │
│  ┌──────────┐ ┌──────────┐ ┌────────────────────┐   │
│  │ 分子编辑器 │ │ Pareto图  │ │ FTO 报告面板      │   │
│  │ (Ketcher) │ │ (Plotly)  │ │ (Markdown+表格)   │   │
│  └──────────┘ └──────────┘ └────────────────────┘   │
│         │            │               │               │
│         └────────────┼───────────────┘               │
│                      │ gRPC-Web + REST               │
└──────────────────────┼──────────────────────────────┘
                       │
              ┌────────▼────────┐
              │  API Gateway    │
              │  (FastAPI)      │
              └────────┬────────┘
                       │ NATS/gRPC
              ┌────────▼────────┐
              │  Orchestrator   │
              └─────────────────┘
```

## 组件树

```
App
├── Layout
│   ├── Sidebar (项目列表)
│   └── TopBar (用户/设置)
├── Pages
│   ├── /projects → ProjectListPage
│   ├── /projects/:id → ProjectDetailPage
│   │   ├── DesignPanel (CIG 构建器)
│   │   ├── MoleculeGrid (分子卡片网格)
│   │   ├── ParetoChart (前沿可视化)
│   │   ├── FTOReport (专利分析)
│   │   ├── RetrosynTree (逆合成树)
│   │   └── AuditLog (审计追踪)
│   └── /settings → SettingsPage
└── Hooks
    ├── useProject(id)
    ├── useDesignStream(wsUrl)
    ├── useMoleculeList(projectId)
    └── useParetoData(projectId)
```

## 数据流

1. 用户定义 CIG（Chemical Intent Graph）→ POST /api/v1/projects/{id}/design
2. API Gateway → Orchestrator (NATS)
3. Orchestrator → 生成/验证/FTO 流程
4. 中间结果通过 WebSocket 流式推送到 UI
5. 最终 SSP（Structured Synthesis Protocol）展示给用户

## 安全

- OIDC 认证（Keycloak）
- CSRF token via cookie
- CSP headers 限制脚本来源
