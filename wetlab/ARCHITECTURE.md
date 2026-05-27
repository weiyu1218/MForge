# Wetlab Architecture (Planned)

> **阶段**：Phase 3 设计规划

## 闭环设计

```
MoleculeForge 虚拟设计
        │
        ▼
XDL Compiler (SSP → XDL 2.0)
        │
        ▼
SiLA2 Adapter (XDL → 硬件指令)
        │
        ▼
Hardware Drivers (Chemputer / Opentrons / Chemspeed)
        │
        ▼
自动化合成 + 纯化 + 表征
        │
        ▼
生物活性测试 (ELN 记录)
        │
        ▼
结果反馈 → Oracle 验证 → HUMU 更新 → 下一轮设计
```

## 硬件驱动接口

```python
class BaseHardwareDriver:
    """所有硬件驱动的抽象接口"""
    async def connect(self) -> None: ...
    async def execute_protocol(self, xdl: str) -> dict: ...
    async def get_status(self) -> dict: ...
    async def disconnect(self) -> None: ...
```

## 数据标准

- **实验描述**：XDL 2.0 (Chemical Description Language)
- **实验室通信**：SiLA2 (Standardisation in Lab Automation)
- **分析数据**：AnIML / JCAMP-DX
- **ELN 格式**：Allotrope Data Format (ADF)
