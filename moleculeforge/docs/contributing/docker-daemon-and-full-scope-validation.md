# Docker Daemon 不可用时的功能边界

Docker daemon 不可用时，项目仍然可以进行代码开发和部分本机测试，但不能完成完整平台级功能验证。

不可实现或不可验收的功能如下：

- 不能使用 `make run-dev` 启动统一开发服务栈。
- 不能通过 Docker Compose 拉起前后端完整平台。
- 不能自动启动 `postgres`、`neo4j`、`qdrant`、`minio`、`redis` 等基础设施。
- 不能统一启动 `api-gateway`、`orchestrator-svc`、生成器服务、验证服务、逆合成服务、供应服务、critic 服务和 provenance 服务。
- 不能通过 `http://127.0.0.1:8000` 验证浏览器到后端服务栈的完整链路。
- 不能完成 production/full scope 的真实多服务编排验证。
- 不能声明 production/full scope 全流程已经跑通。

因此，Docker daemon 不可用时，只能验证本机代码逻辑和部分轻量级链路，不能验证完整平台运行状态。
