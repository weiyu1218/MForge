# ADR-0001: 选择 Lorentz 模型而非 Poincaré 作为双曲流形表示

**状态**：已采纳
**日期**：2025-Q1
**决策者**：MoleculeForge 架构团队

## 背景

HUMU（Hyperbolic Unified Molecular Understanding）需要将分子、口袋、合成路径编码到统一的双曲流形上。双曲几何有两种主要的等距模型：

- **Poincaré 球模型**：在单位球内表示，距离公式简单但有数值不稳定性（边界附近精度丢失）
- **Lorentz 超boloid 模型**：在 R^{d+1} 上，约束 <x,x>_L = -1/c，数值更稳定，与狭义相对论的闵可夫斯基空间同构

## 决策

**选择 Lorentz 模型** 作为 HUMU 的唯一流形表示。

## 理由

1. **数值稳定性**：Lorentz 模型的 exp/log map 涉及 cosh/sinh，在 GPU 上（float32/float16）比 Poincaré 的分数线性变换更稳定
2. **梯度流动**：Lorentz 约束 <x,x>_L = -1/c 可以通过显式投影保证，反向传播更平滑
3. **曲率泛化**：Lorentz 模型对可学习曲率 c 的缩放公式更简洁（expmap: cosh(√c·|v|)·x + sinh(√c·|v|)/(√c·|v|)·v）
4. **与 Flow Matching 兼容**：HFM-3D 需要在切丛上进行 ODE 求解，Lorentz 切空间投影公式更直接
5. **文献依据**：Nickel & Kiela (2018), Law et al. (2019) 均推荐 Lorentz 用于高维双曲学习

## 后果

- 所有 HUMU 操作必须在 Lorentz 模型上实现（不能混用 Poincaré）
- 实现 `libs/mf-humu/src/mf_humu/manifold/lorentz.py` 作为唯一流形类
- Encoder 输出必须投影到 Lorentz 约束（通过 `LorentzManifold.project()`）
- 数据库中的向量存储使用 Lorentz 坐标（d+1 维）

## 验证

- `test_lorentz_manifold.py` 验证 exp/log 互逆性、Lorentz 约束、曲率缩放
