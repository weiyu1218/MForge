# mf-agents

Reusable agent framework: BaseAgent, Redis bus wrapper, lineage tracker,
multi-provider LLM client, and LangGraph helpers.

## Layout

```
src/mf_agents/
├── base/          BaseAgent + tool/memory abstractions
├── messaging/     Redis message bus wrapper, signed envelopes, routing
├── lineage/       Sigstore signer, Neo4j writer, span tracker
├── llm/           Multi-provider client (Claude, OpenAI, DeepSeek)
├── crg/           Chemical Reasoning Graph operations + persistence
└── workflow/      LangGraph helpers, simple state-machine framework
```
