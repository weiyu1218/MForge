# Generator Image CI 修复 Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 补齐 generator runtime 的 RDKit random generator 依赖，使生成器镜像导入检查能够成功。

**Architecture:** 保留现有 GitHub Actions、镜像构建脚本和 Dockerfile 导入检查，只修正根项目 `generator-runtime` optional dependency 的闭包。回归测试通过 `uv export` 解析真实锁文件，验证运行时依赖实际包含 RDKit random generator。

**Tech Stack:** Python 3.12、uv、pytest、Docker、GitHub Actions

## Global Constraints

- 只修改 generator runtime 依赖闭包，不删除 Dockerfile 中的模块导入检查。
- 不修改生成器业务逻辑、服务接口或部署配置。
- 使用现有 `mf-generators-rdkit-random` workspace 包。
- 先验证测试 RED，再实施最小修复并验证 GREEN。
- 功能开发分支为 `feature/fix-generator-image-ci`。

---

### Task 1: 补齐 generator runtime 依赖闭包

**Files:**
- Modify: `tests/unit/test_service_artifact_status.py`
- Modify: `pyproject.toml`
- Modify: `uv.lock`

**Interfaces:**
- Consumes: 根项目的 `project.optional-dependencies.generator-runtime` 和 `uv.lock`
- Produces: 能解析并安装 `mf-generators-rdkit-random` 的 generator runtime

- [x] **Step 1: 写入失败回归测试**

```python
def test_generator_runtime_resolves_rdkit_random_package() -> None:
    completed = subprocess.run(
        [
            "uv",
            "export",
            "--frozen",
            "--extra",
            "generator-runtime",
            "--no-dev",
            "--no-hashes",
            "--no-editable",
            "--format",
            "requirements.txt",
        ],
        cwd=ROOT,
        capture_output=True,
        check=False,
        text=True,
    )

    assert completed.returncode == 0, completed.stderr
    resolved_requirements = {
        line.partition(" ; ")[0].removeprefix("-e ").strip()
        for line in completed.stdout.splitlines()
        if line and not line.lstrip().startswith("#")
    }
    assert "./models/mf-generators/rdkit_random" in resolved_requirements
```

- [x] **Step 2: 运行测试并确认 RED**

Run:

```bash
uv run pytest tests/unit/test_service_artifact_status.py::test_generator_runtime_resolves_rdkit_random_package -q
```

Expected: FAIL，导出的 generator runtime requirements 不包含 `models/mf-generators/rdkit_random`。

- [x] **Step 3: 实施最小依赖修复**

在 `pyproject.toml` 的 `generator-runtime` 列表中加入：

```toml
"mf-generators-rdkit-random",
```

- [x] **Step 4: 更新锁文件**

Run:

```bash
uv lock
```

Expected: `uv.lock` 中根项目 `generator-runtime` extra 包含 `mf-generators-rdkit-random`。

- [x] **Step 5: 运行测试并确认 GREEN**

Run:

```bash
uv run pytest tests/unit/test_service_artifact_status.py::test_generator_runtime_resolves_rdkit_random_package -q
```

Expected: PASS。

- [x] **Step 6: 执行回归验证**

Run:

```bash
uv lock --check
uv run pytest --maxfail=20
uv run lint-imports --no-cache
git diff --check
```

Expected: 命令全部退出为 0；需要外部模型、数据集或数据库的测试可以保持显式跳过。

- [ ] **Step 7: 提交并推送修复分支**

Run:

```bash
git add moleculeforge/pyproject.toml moleculeforge/uv.lock moleculeforge/tests/unit/test_service_artifact_status.py moleculeforge/docs/todo/2026-07-30-18-38-generator-image-ci-方案.md
git commit -m "fix: 补齐生成器镜像运行依赖"
git push -u origin feature/fix-generator-image-ci
```

Expected: 远端分支提交哈希与本地 HEAD 一致。
