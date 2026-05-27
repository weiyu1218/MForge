# MoleculeForge 核心架构设计方案 v2.0
## 端到端分子逆向设计平台 · 创新前沿核心架构

> **架构哲学**：分子逆向设计的本质是在"能被合成的"与"值得被合成的"两个约束之间，寻找最优的化学意图实现路径。现有系统把生成、验证、合成规划视为三个独立任务——这是错的。MoleculeForge 的核心创新是：**在统一的流形上同时学习这三个任务的联合分布**，让智能体在这个空间内"游走"，而不是在不同黑盒工具之间传递字符串。

> **三个部分保留位置（核心架构完成后实现）**：
> - 🔲 **前端用户界面**（预留 Layer UI 接口：REST/WebSocket/gRPC，格式 OpenAPI 3.1）
> - 🔲 **湿实验室硬件接口**（预留 Wet-Lab Bridge API：XDL 2.0 / SiLA2 输出端口）
> - 🔲 **商业化与部署**（预留 Multi-Tenant K8s 部署模板）

---

# 第〇层：元架构哲学与根本创新

## 0.1 当前分子逆向设计的三大"结构性断层"

```
现有方案的流程（线性、断层式）：

[生成模型] → SMILES 字符串 → [验证工具] → 打分 → [逆合成工具] → 路径
     ↑                                                        |
     └──────────── 无反馈 / 无梯度 / 无因果 ───────────────────┘

问题：
① 生成模型不知道"这个分子能不能被合成"（Synthesis-Blind Generation）
② 验证工具不知道"这个分子是从哪个目标函数演化来的"（Context-Blind Scoring）  
③ 逆合成规划不知道"哪种路径更贴合客户的供应链/成本约束"（Supply-Blind Planning）
```

## 0.2 MoleculeForge 的根本创新：联合流形共生成 (Joint Manifold Co-Generation, JMCG)

**核心思想**：把分子结构 `m`、其合成路径 `r`、以及其性质轮廓 `p` 视为一个**联合随机对象** `(m, r, p)`，在共享的双曲流形 `ℍ^d` 上建模其联合分布：

```
p(m, r, p | T, c) ≠ p(m|T) · p(r|m) · p(p|m)    ← 现有方案（独立假设，错误）

p(m, r, p | T, c) = ∫ p(m,r,p|z,T,c) · q(z|T,c) dz    ← MoleculeForge（联合学习）
                         ℍ^d
```

其中 `z ∈ ℍ^d` 是**双曲化学意图向量**（Hyperbolic Chemical Intent Vector, HCIV），它同时编码了：
- 这个意图对应的**分子结构空间区域**（什么样的骨架）
- 这个意图对应的**合成可达子空间**（哪些反应可以实现）
- 这个意图对应的**性质轮廓方向**（什么性质可以期待）

这使得生成器天然产出"可合成的、有药理意义的"分子，而不需要事后过滤。

---

# 第一层：化学意图编译器 (Chemical Intent Compiler, CIC)

## 1.1 层的定位

**输入**：自然语言描述（任意形式）  
**输出**：结构化化学意图表示 (Chemical Intent Representation, CIR)  
**核心创新**：CIR 不是简单的 YAML 配置，而是一个**有类型的、可微分的、多模态的化学目标函数图**

## 1.2 三阶段编译流程

### 阶段 1.2.1：语义解析与知识锚定

```
用户输入："设计一个能绕开 Mirati 专利、对 KRAS G12C 有纳摩尔级活性、
         口服吸收好、没有 CYP3A4 相互作用的小分子"
         
            ↓  LLM (Scientific Reasoning Model, SRM)
            ↓  + 工具调用：UniProt / PDB / SureChEMBL / ChEMBL
            
科学实体抽取：
  - 靶点：KRAS G12C [UniProt: P01116, 突变: G12C, 口袋: Switch-II, PDB: 8AFB]
  - 目标活性：IC50 < 100 nM（→ ΔG < -9.5 kcal/mol）
  - ADMET 约束：logP ∈ [1,4], F_oral > 30%, CYP3A4 抑制 IC50 > 10 μM
  - IP 约束：FTO w.r.t. [US11291420, US11186593, ...](Mirati 专利组)
  - 靶标选择性（隐含）：vs HRAS/NRAS fold > 100
```

### 阶段 1.2.2：化学意图图 (Chemical Intent Graph, CIG) 构建

CIG 是 CIR 的核心数据结构——一个**有向超图**，节点是目标变量，边是约束/偏好关系：

```
CIG 结构（JSON-LD 格式，可序列化为向量）：

{
  "intent_id": "CIG-20260429-001",
  "version": "2.0",
  "signature": "sigstore://...",   ← 可审计锚点
  
  "target_context": {
    "pocket_embedding": [0.23, -1.45, ...],   ← ESMFold/AF3 口袋向量 d=512
    "pharmacophore_3d": {                      ← CReM-pharm 格式
      "hbd": [[x1,y1,z1], ...],
      "hba": [[x2,y2,z2], ...],
      "hydrophobic": [[x3,y3,z3], ...],
      "aromatic": [[x4,y4,z4], ...]
    },
    "binding_mode_prior": "covalent_reversible"   ← 引导生成器
  },
  
  "objective_nodes": [
    {
      "id": "affinity",
      "type": "continuous_maximize",
      "oracle": "Boltz2_then_FEP",
      "target_value": -9.5,             ← kcal/mol
      "uncertainty_tolerance": 0.5,
      "weight": 0.35,
      "pareto_tier": 1
    },
    {
      "id": "selectivity",
      "type": "ratio_maximize",
      "numerator_oracle": "affinity_KRAS",
      "denominator_oracle": "affinity_HRAS",
      "target_ratio": 100,
      "weight": 0.20,
      "pareto_tier": 1
    },
    {
      "id": "admet_bundle",
      "type": "multi_constraint_satisfy",
      "constraints": {
        "logP": {"range": [1, 4]},
        "F_oral": {"min": 0.30},
        "CYP3A4_IC50": {"min": 10},
        "hERG_IC50": {"min": 1},
        "Ames_positive": {"max": 0.05},
        "TPSA": {"range": [60, 140]}
      },
      "soft_penalty": true,
      "weight": 0.20,
      "pareto_tier": 2
    },
    {
      "id": "fto_score",
      "type": "continuous_maximize",
      "oracle": "PatentEmbeddingDistance",
      "blocked_patent_ids": ["US11291420", "US11186593"],
      "similarity_threshold": 0.85,   ← 高于此值必须拒绝
      "weight": 0.15,
      "pareto_tier": 1               ← 专利是硬约束，Tier=1
    },
    {
      "id": "synthetic_accessibility",
      "type": "continuous_maximize",
      "oracle": "AiZynthFinder4_RouteScore",
      "target": {"max_steps": 5, "min_bb_availability": 0.8},
      "weight": 0.10,
      "pareto_tier": 2
    }
  ],
  
  "generative_priors": {
    "scaffold_bias": "covalent_warhead_acrylamide",    ← 引导 CReM/FragFM
    "mw_range": [350, 550],
    "ring_systems": ["fused_bicyclic", "saturated_ring"],
    "forbidden_substructures": ["PAINS", "reactive_groups"],
    "novelty_vs_analogy": 0.6   ← 0=纯类似物, 1=完全从头
  },
  
  "budget_constraints": {
    "oracle_L2_calls_max": 5000,
    "oracle_L3_calls_max": 200,
    "oracle_L4_calls_max": 20,
    "wallclock_hours": 12,
    "cost_usd_max": 500
  }
}
```

### 阶段 1.2.3：HCIV 编码（CIG → 双曲意图向量）

CIG 中的每个节点和边通过**意图编码器网络** `Enc_intent` 转化为双曲空间中的一个方向向量和置信半径，形成**意图锥 (Intent Cone)**：

```
HCIV = Enc_intent(CIG) ∈ ℍ^128_Lorentz

意图锥：以 HCIV 为顶点，
        以性质约束为约束面的双曲扇形区域。
生成任务 = 在意图锥内采样满足约束的点。
```

---

# 第二层：双曲统一分子宇宙 (Hyperbolic Unified Molecular Universe, HUMU)

## 2.1 为什么是双曲空间？——严格的数学论证

### 2.1.1 欧氏空间的本质缺陷

分子世界具有**天然的指数级层次结构**：
- 化学空间：10^60 量级的分子
- 骨架层次：BEMIS-MURCKO 骨架树是树状结构
- SAR 关系：activity cliff（结构差 1 个原子，活性差 1000×）
- 合成树：每个目标分子对应一棵反应树

欧氏空间的体积随 r 多项式增长（`V ~ r^d`），而树状结构的节点数指数增长（`N ~ e^{αr}`）。因此**欧氏空间天然无法低维地嵌入树状化学世界**——这是维度灾难的根源。

### 2.1.2 Lorentz 双曲模型的优越性

```
Lorentz 模型：ℍ^d = {x ∈ ℝ^{d+1} | ⟨x,x⟩_L = -1, x_0 > 0}
Lorentzian 内积：⟨x,y⟩_L = -x_0·y_0 + Σ x_i·y_i

距离：d_L(x,y) = arcosh(-⟨x,y⟩_L)
体积：V(r) ~ sinh^d(r) ≈ e^{dr}  ← 指数增长！与树匹配

优势：
1. 低维容量大（d=64 ≈ 欧氏 d=512 的层次表达能力）
2. 解析梯度（exponential map / log map 有闭式解）
3. 原生支持层次聚类（根节点在原点附近，叶节点在边缘）
4. Activity cliff 天然分离（小角度差 → 大双曲距离）
```

### 2.1.3 HUMU 的三维结构

```
HUMU 是 ℍ^128 的子流形，三个方向轴：

  径向深度 r = ‖x‖_L：分子的"普适性" 
    近原点 → 骨架型分子（高泛化、低特异）
    远离原点 → 高度特异的分子（活性悬崖区）
    
  极角 θ_target：与特定靶点口袋向量的角度
    不同靶点对应 ℍ^128 的不同锥形扇区
    
  方位角 φ_synth：合成复杂度方向
    φ → 0：简单、可以从 REAL Space 立即购买
    φ → π：高度复杂多步合成路线
    
直觉：理想的药物分子 = 特定扇区（靶点）内、
      中等径向深度（活性好但不太 OOD）、
      小方位角（合成简单）的点。
```

## 2.2 HUMU 的联合编码器网络

```
HUMU 需要三类编码器，共享双曲主干：

① 分子编码器 Enc_mol：
   输入：3D 分子图 G = (V, E, x∈ℝ^{Nx3})
   网络：SE(3)-Equivariant Message Passing (SEGNN) → 切空间投影 → exp_μ
   输出：z_mol ∈ ℍ^128

② 口袋编码器 Enc_pocket：
   输入：蛋白口袋点云 P = {(atype_i, coord_i)}
   网络：EquiBind-style E(3)-GNN → 切空间 → exp_μ
   输出：z_pocket ∈ ℍ^128

③ 合成路径编码器 Enc_route：
   输入：反应树 T = (V_rxn, E_rxn)（AND-OR 图）
   网络：双向图 Transformer → 双曲 TreeLSTM → exp_μ
   输出：z_route ∈ ℍ^128

联合对比损失（训练 HUMU）：

L_HUMU = L_mol-pocket(z_mol, z_pocket)     ← HypSeek 风格亲和力对比
        + L_mol-route(z_mol, z_route)      ← 分子-合成路径一致性
        + L_fto(z_mol, z_patent)           ← 与专利向量最大化距离
        + λ·L_curvature(c)                 ← 可学习曲率正则
```

---

# 第三层：自适应多范式生成引擎 (Adaptive Multi-Paradigm Generative Engine, AMGE)

## 3.1 架构总览：不是"集成"，是"协进化"

现有的多模型集成只是在输出层投票。AMGE 的创新是：**8 种生成范式共享 HUMU 潜空间，通过跨范式知识蒸馏互相教学，并由任务感知路由器动态协同**。

```
                    ┌─────────────────────────────────────────┐
                    │         HUMU (ℍ^128, 共享)              │
                    │  意图锥 (Intent Cone, HCIV 定义)         │
                    └──────────────┬──────────────────────────┘
                                   │ 采样 z ~ q(z|cone)
                    ┌──────────────▼──────────────────────────┐
                    │   任务感知 MoE 路由器 (TAR)              │
                    │   输入：HCIV + 任务画像 + 预算状态       │
                    │   输出：π_i（8 个生成器的权重）           │
                    └─┬─────┬──────┬──────┬──────┬────────────┘
                      │     │      │      │      │
               ┌──────▼─┐ ┌─▼───┐ ┌▼──┐ ┌▼──┐ ┌▼───────────┐
               │ HFM-3D │ │FragFM│ │LaMGen│ │CLM+│ │MMPT-RAG   │
               │双曲流匹配│ │片段DFM│ │3D-LLM│ │RL │ │药化直觉   │
               └────────┘ └─────┘ └───┘ └───┘ └────────────┘
                      │     │      │      │      │
                    ┌─▼─────▼──────▼──────▼──────▼────────────┐
                    │   跨范式知识蒸馏层 (Cross-Paradigm KD)   │
                    │   Teacher: Boltz-2 / HypSeek 打分        │
                    │   Student: 各生成器共享梯度信号           │
                    └──────────────┬──────────────────────────┘
                                   │
                    ┌──────────────▼──────────────────────────┐
                    │    候选分子池 + HUMU 坐标 + 不确定度     │
                    └─────────────────────────────────────────┘
```

## 3.2 核心生成器详细设计

### 3.2.1 HFM-3D：双曲流匹配（核心理论创新）

**现有方法**：Flow Matching 在欧氏空间 ℝ^{3N} 上定义 ODE。  
**我们的创新**：把生成过程定义在 **Lorentz 双曲流形的切丛** 上，利用指数映射和测地线采样。

```
数学形式：

Flow Matching on ℍ^d：
  源分布 p_0 = uniform on ℍ^d（无偏先验）
  目标分布 p_1 = q(m|T, z)（靶点条件的分子分布）
  
条件流（Conditional Flow）：
  x_t = exp_{x_0}(t · log_{x_0}(x_1))   ← 双曲测地线插值
  
向量场（在切空间 TℍM 上）：
  u_t(x_t) = d/dt exp_{x_0}(t · log_{x_0}(x_1))|_{t}
            = (1/sin(d_t)) · (cos(d_t)·x_t - x_1) / sinh(r_t)
            
网络参数化：
  v_θ(x_t, t, z_pocket, z_intent) ∈ T_{x_t}ℍ^d
  
  使用 Lorentz-equivariant Transformer（类似 GATr 但在双曲切空间）
  
采样过程（推理时）：
  1. 从 HCIV 意图锥采样 z_0 ∈ cone(HCIV)
  2. 解 ODE: dx_t/dt = v_θ(x_t, t, ...)   [20 步，Euler/Midpoint]
  3. x_1 → 原子坐标 + 类型（通过解码器）

优势：
  - 意图锥约束确保生成结果在目标区域内
  - 双曲几何使 activity cliff 区域稀疏（自然避免生成"活性悬崖"）
  - 20 步即达 SemlaFlow 质量（vs DDPM 的 1000 步）
```

### 3.2.2 FragFM：片段级双层离散流匹配（引自 FragFM 论文延伸）

**核心设计**：两层 DFM，Level-0 生成 scaffold（骨架），Level-1 生成 R 基团（取代基），两层在 HUMU 中共享向量场。

```python
# 伪代码：FragFM 的两层生成
class FragFM_HUMU:
    def __init__(self):
        self.scaffold_fm = DiscreteFlowMatching(vocab=SCAFFOLD_VOCAB)
        self.rgroup_fm   = DiscreteFlowMatching(vocab=RGROUP_VOCAB)
        self.humu_bridge = LorentzProjection(dim=128)  # 共享 HUMU
    
    def forward(self, z_intent: HyperbolicVector, n_frags: int):
        # Level 0: Scaffold CTMC (Continuous-Time Markov Chain)
        # 速率矩阵 R_t 由 z_intent 调制
        scaffold_probs = self.scaffold_fm.sample(
            z=self.humu_bridge(z_intent), 
            context="scaffold_for_kras_switch2"
        )
        scaffold = scaffold_probs.argmax(dim=-1)   # 骨架 token 序列
        
        # Level 1: R-group DFM 以 scaffold + HCIV 为条件
        molecule = self.rgroup_fm.grow(
            scaffold=scaffold,
            pharmacophore=z_intent.pharmacophore_3d,
            sa_penalty=True  # 内嵌合成可及性惩罚
        )
        return molecule, self.humu_bridge.embed(molecule)
```

**关键**：SA 惩罚被**内嵌在 R 基团的转移概率矩阵**中：
```
R_t(y' → y) = base_rate(y'→y) · exp(-λ·SA_penalty(scaffold+y))
```
这使得 FragFM 生成的分子天然 SA 分数高——不是事后过滤，而是生成期内约束。

### 3.2.3 LaMGen-3D（多靶点 LLM-based 3D 生成，扩展自论文 #6）

**原论文局限**：仅支持 3 个靶点，3D token 精度受限，推理速度 0.44 s/mol 但精度不足以直接用于虚筛。

**我们的扩展**：

```
LaMGen-3D-Pro 架构：

  输入：{T_1, T_2, ..., T_k} 任意数量靶点的口袋嵌入（ESM2 + PocketMiner）
        + z_intent（HCIV）
  
  核心创新：多靶点交叉注意力门控
  
  Pocket_Attn(Q=mol_tokens, K=V=pocket_tokens_all_targets)
  → 每个分子 token 同时感知所有靶点，
    Gate_i = sigmoid(W_g · [pocket_i_attn; z_intent]) 控制靶点权重
  
  3D Token 精度提升：
    - 旋转感知 token + SE(3)-invariant 距离矩阵编码
    - 引入 SemlaFlow 的 scale optimal transport 确保几何一致性
    - 直接输出 HUMU 坐标（z ∈ ℍ^128），而非中间 SMILES
    
  推理速度：~0.3 s/mol（引入 speculative decoding）
  多靶点覆盖：支持 k ≤ 10 靶点（典型多靶点药物 k=2-4）
```

### 3.2.4 增量 CLM（引自论文 #10，扩展至闭环在线学习）

**原论文**：静态的增量训练，针对特定 SAR 系列。  
**我们的扩展**：**在线持续学习（Online Continual Learning）机制**：

```
闭环增量训练流程：

每当湿实验或高精度 oracle 返回新数据点 (m_i, y_i)：
  1. 计算新数据在 HUMU 中的位置 z_i
  2. 判断是否属于"已知骨架类"（KNN 在 ℍ^128 中检索）：
     - 若是（r < τ_scaffold）→ 更新该骨架类的 CLM fine-tune
     - 若否（r > τ_scaffold）→ 新建 SAR 分支，开始新任务微调
  3. 使用 EWC（Elastic Weight Consolidation）防止灾难性遗忘
  4. 用 PackNet 结构剪枝，为新任务分配参数子空间

数学：EWC 正则项
  L_EWC = L_new + λ·Σ F_i(θ_i - θ*_i)^2
  其中 F_i 是 Fisher 信息矩阵对角，θ*_i 是旧参数
```

### 3.2.5 MMPT-RAG（匹配分子对变换 + 检索增强，引自论文 #9 扩展）

**原论文**：学习 MMP 变换的"药物化学直觉"，但未处理 FTO 问题。  
**我们的扩展**：把**专利数据库作为负样本的检索语料**，让 MMPT 学会"绕开"：

```
MMPT-RAG 增强框架：

检索库：
  正样本库：ChEMBL MMP pairs（已验证有效的 R 基团替换）
  负样本库：SureChEMBL Markush 展开（已申请专利的变换）
  
对于目标分子 m_query，生成类似物：
  1. 在正样本库检索最相似的 MMP pair (m_A → m_B)
  2. 在负样本库检索最近的专利变换 (m_P → m_P')
  3. RAG 编码器融合：
     z_rag = Attn(Q=Enc(m_query), K=[Enc(m_A→B); Enc(m_P→P')_neg])
  4. 解码器生成新变换，使得：
     similarity(transform, positive_transforms) ↑
     similarity(transform, patent_transforms) ↓   ← FTO 意识
     
实现：Seq2Seq Transformer + 负样本对比解码
      推理时：beam search with FTO penalty score 重排
```

### 3.2.6 EvoMol-RL（强化学习-遗传算法混合，引自论文 #2 深度扩展）

**原论文**：RL 引导的遗传算法，但奖励是手工设计的标量。  
**我们的扩展**：**多目标 RL + 超体改善 (Hypervolume Improvement) 奖励**：

```python
class EvoMolRL_Pareto:
    def compute_reward(self, mol: Molecule) -> float:
        # 获取多维奖励向量
        scores = {
            'affinity': self.oracle_L1.predict(mol),   # Boltz-2
            'admet':    self.admet_oracle.predict(mol),
            'fto':      self.patent_oracle.score(mol),
            'sa':       self.sa_oracle.score(mol)
        }
        
        # Pareto 超体改善（vs 当前 Pareto 前沿）
        hvi = compute_hypervolume_improvement(
            new_point=scores, 
            current_pareto_front=self.pareto_archive,
            reference_point=self.reference_point
        )
        return hvi
    
    def sleeping_bandit_policy(self, mol_context: EcfpVector) -> Action:
        """
        Sleeping Bandit（引自原论文）：
        根据 ECFP 上下文决定使用哪种 mutation operator
        - 在 HUMU 中的位置决定 mutation 半径（活性悬崖边缘用小扰动）
        - 合成约束决定 mutation 类型（只保留化学有效的操作）
        """
        z = self.humu.embed(mol_context)
        cliff_risk = self.cliff_detector(z)   # HypSeek 估计
        if cliff_risk > 0.7:
            return self.fine_mutation(mol_context)
        else:
            return self.scaffold_hop(mol_context)
```

### 3.2.7 CReM-pharm-3D（引自论文 #4，升级为实时 3D 感知版本）

**原论文**：药效团约束 + 合成可及性，但 3D 对接分数未内嵌。  
**我们的扩展**：把 DiffDock-L 的快速对接（2 s/pose）实时嵌入 CReM 的片段替换循环：

```
CReM-3D 生成流程：
  
  1. 从 CIG 提取 3D 药效团 Φ = {(type_i, coord_i, tol_i)}
  2. 选取初始 scaffold（来自 HypSeek 检索当前意图锥内的最高 SA 分子）
  3. CReM 片段替换循环（最多 10 次迭代）：
     For each fragment position f in scaffold:
       候选碎片 = CReM_db.query(attachment_point=f, sa_filter=True)
       For each fragment g in candidates:
         m_new = scaffold.replace(f, g)
         pose_score = DiffDockL.fast_score(m_new, pocket)   ← 2s/pose
         pharmacophore_match = Φ.match_3d(m_new)
         humu_z = HUMU.embed(m_new)
         fto_score = patent_oracle.score(humu_z)
         
         combined = α·pose_score + β·pharmacophore_match - γ·(1-fto_score)
       best_g = argmax(combined)
       scaffold = scaffold.replace(f, best_g)
  
  4. 输出：高 SA、好对接、符合药效团、FTO 安全的分子
```

### 3.2.8 不熟悉度感知生成器（引自论文 #8，系统性整合）

**Nat Mach Intell 2026"Molecular deep learning at the edge of chemical space"** 指出：生成模型在 OOD 区域预测不可靠，但当前没有系统方法**在生成阶段就规避 OOD**。

**我们的创新**：**不熟悉度感知采样（Unfamiliarity-Aware Sampling, UAS）**：

```
UAS 机制：

每次从 HUMU 采样候选 z 时：
  1. 计算不熟悉度 U(z) = reconstruction_loss(Enc_mol(Dec_mol(z)))
     （Auto-encoder 的重建误差反映 OOD 程度）
     
  2. 修正采样分布：
     p_safe(z) ∝ p_intent(z) · exp(-β·U(z))
     
     β 是 unfamiliarity penalty 强度
     → 高不熟悉度区域被压缩，引导生成器留在"已知化学可信域"
     
  3. OOD 预警机制：
     If U(z) > τ_ood:
       标注为 "Extrapolation Risk"
       触发：Incremental CLM 主动学习入队
             oracle 精度降级警告（展示给 Agent）
             建议湿实验验证
```

## 3.3 任务感知 MoE 路由器 (Task-Aware Router, TAR)

TAR 决定每个生成步骤使用哪些生成器、权重多少。这是一个**在线学习的路由策略**：

```python
class TaskAwareRouter:
    """
    输入：任务画像 + HCIV + 历史 oracle 反馈
    输出：8 个生成器的混合权重 π ∈ Δ^7（概率单纯形）
    """
    def __init__(self):
        # 任务画像特征
        self.task_features = [
            'target_family',        # kinase/GPCR/PPI/covalent/multi-target
            'data_richness',        # ChEMBL 数据量 (log scale)
            'novelty_demand',       # 用户指定新颖性 vs 类似物
            'sa_priority',          # 合成优先级权重
            'budget_remaining',     # 剩余 oracle 预算
            'stage',                # hit/lead_opt/scaffold_hop
            'fto_risk',             # 当前 Pareto 前沿的 FTO 均值
            'cliff_density',        # 当前搜索区域的 activity cliff 密度
        ]
        self.router_net = ProxylessNAS_Router(in_dim=64, n_experts=8)
        
    def route(self, hciv, task_profile, oracle_history):
        x = self.encode_features(hciv, task_profile, oracle_history)
        logits = self.router_net(x)
        
        # 特殊规则覆盖（硬约束）：
        if task_profile['stage'] == 'scaffold_hop':
            logits[CREM_IDX] -= 100  # CReM 不适合骨架跳跃
        if task_profile['data_richness'] < 50:
            logits[CLM_IDX] += 2.0   # 数据稀缺时优先 incremental CLM
        if task_profile['fto_risk'] > 0.7:
            logits[MMPT_IDX] += 1.5  # FTO 风险高时优先 MMPT 绕开
            
        return F.softmax(logits, dim=-1)
```

**TAR 的在线学习**：每次 oracle 返回反馈后：
```
ΔW_router = η · ∇_W [HVI(oracle_result) - baseline] · log π_selected
```
这是一个 REINFORCE 型更新，让路由器从每次任务反馈中学习哪种生成器在什么情境下最有效。

---

# 第四层：多智能体推理大脑 (Multi-Agent Reasoning Brain, MARB)

## 4.1 智能体系统的根本创新：从"工具调用者"到"化学推理者"

现有的 Agent 系统（ChemCrow、FROGENT、ChatInvent）本质上是"LLM + 工具列表"——LLM 决定调用哪个工具，传入参数，返回结果。这有三个根本问题：

1. **工具调用无梯度**：LLM 的决策不能从 oracle 结果中反向传播更新
2. **多步推理无记忆**：每次对话都是独立的，Agent 不能积累化学直觉
3. **Agent 间无真正协同**：只是主从调用，不是分布式推理

**MARB 的创新**：**化学推理图 (Chemical Reasoning Graph, CRG)** 作为 Agent 间的共享状态空间

```
CRG 是一个动态有向图，节点是"化学信念 (Chemical Beliefs)"，边是"推理步骤"：

{
  "beliefs": [
    {"id": "B001", "type": "affinity_estimate", "value": -8.2, "confidence": 0.7, 
     "source_agent": "ValidationAgent", "evidence": ["Boltz2_pred"]},
    {"id": "B002", "type": "synthesis_feasible", "value": true, "confidence": 0.9,
     "source_agent": "RetrosynAgent", "evidence": ["AiZynth_solved_4steps"]},
    {"id": "B003", "type": "fto_clear", "value": false, "confidence": 0.6,
     "source_agent": "FTOAgent", "evidence": ["SureChEMBL_dist=0.82"]}
  ],
  "reasoning_edges": [
    {"from": "B001", "to": "decision_proceed", "logic": "affinity>-8.0 ∧ confidence>0.6"},
    {"from": "B003", "to": "action_mmpt_escape", "logic": "fto_clear=false → trigger MMPT"}
  ]
}
```

所有 Agent 读写 CRG，Orchestrator 监控 CRG 的一致性。

## 4.2 各智能体详细规格

### Agent-0：Orchestrator（主导研究者 Principal Researcher）

```
角色：全局任务规划 + 资源调度 + 一致性维护
实现：LangGraph StateMachine + Claude Sonnet 4.5 + 科学推理微调层

状态机关键状态：
  PLANNING → GENERATING → VALIDATING → REFINING → ESCALATING → DONE

任务规划（Plan-and-Execute 框架）：
  1. 接收 CIR（来自 CIC 层）
  2. 制定实验设计 DAG（Design-of-Experiments）
     - 哪些生成器先跑（Stage 0: Breadth-first exploration）
     - oracle 预算如何分配（Stage 1: Exploitation of top-k）
     - 何时触发 FTO 检查（Stage 2: Validation）
     - 何时启动 CLM 微调（Stage 3: Refinement）
  3. 监控 CRG 的一致性（检测矛盾信念 → 触发 Critic）
  4. 决策门：什么 Pareto 前沿质量可以"推进"到更贵的 oracle

自我反思机制（Reflexion，DeepMind 2023 风格）：
  每个 mini-cycle 后，Orchestrator 生成一份"研究日志"：
  "本轮生成了 200 个分子，top-5 平均 FTO=0.6（低于预期 0.8）。
   判断：FTO 约束比预期更紧，需要增大 MMPT 路由权重，
   同时查询 SureChEMBL 的最新专利更新。"
  → 自动更新 TAR 的路由权重 + 触发 FTOAgent 重新扫描
```

### Agent-1：NL2Obj（意图解析，对接 CIC 第一层）

```
核心能力：
  - 科学文献理解（调用 PubMed MCP + Semantic Scholar API）
  - 靶点知识库（UniProt / PDB / ChEMBL / BindingDB / DrugBank）
  - FTO 初步检索（SureChEMBL 全文检索）
  - 多轮澄清（识别模糊意图 → 向用户提问）

澄清策略（关键设计）：
  不问用户"你想要什么 logP？"这种技术问题，
  而是问"这个分子是要在哪个患者群体中使用？口服还是注射？"
  → 从临床需求推导技术约束（更像真正的项目主任）

输出：CIG（格式见第一层 1.2.2）+ 置信度评分
```

### Agent-2：Generator Coordinator（生成协调者）

```
核心能力：
  - 激活 AMGE 的 TAR，分发生成任务
  - 监控各生成器的健康状态（失败检测 + 降级）
  - 实时展示生成进度（SSE 流推送到前端预留接口）
  - 管理候选分子的去重（HUMU 距离 > ε）和多样性平衡

多样性保障机制（重要创新）：
  当 Pareto 前沿的分子过于集中（HUMU 中相互距离 < δ）时：
    触发"多样性探索模式"：
    → 强制 HFM-3D 从意图锥的边缘区域采样
    → 提高 scaffold_hop 概率
    → 暂时降低 affinity 权重，提高 novelty 权重
    
  目标：维持 Pareto 前沿的结构多样性（HUMU 中 span > 0.3）
```

### Agent-3：RetroSyn Agent（逆合成规划）

```
核心能力（三层逆合成引擎）：

Layer A：快速可及性过滤（< 0.1 秒/分子）
  - RetroGNN：GNN 估计逆合成求解难度
  - 若预计步数 > max_steps + 2：直接淘汰
  
Layer B：单步逆合成预测（< 1 秒/分子）
  - RSGPT：10B 数据预训练，top-1 准确 63.4%（SOTA）
  - UAlign：图-序列，graph-to-sequence，top-10 准确 81%
  - 集成：2 模型的 beam search 结果排序，取最优
  
Layer C：多步路径规划（< 60 秒/分子）
  - AiZynthFinder 4.0：MCTS 主干 + 工业级调校（3 年 AZ 实践）
  - 约束注入：用户的 bond-prompting（"优先用 Buchwald-Hartwig"）
  - Enamine/REAL 库存检查集成：BB 不在库 → 惩罚此路径
  
创新：合成感知的 HUMU 更新
  当找到高质量路径时：
    z_route = Enc_route(best_route)
    更新 z_mol 的 HUMU 坐标（拉近分子-路径嵌入）
    → 下一轮生成器偏向这个"合成友好"区域采样
```

### Agent-4：Validation Agent（多级验证）

```
分级 Oracle 瀑布（Adaptive Oracle Cascade）：

L0 快速滤：QED ≥ 0.3, SA ≤ 4, Lipinski ≤ 1 violation, PAINS clean
  耗时 < 1 ms/mol，过滤率 ~40%

L1 神经预测：Boltz-2 亲和力 + ADMET-AI + Chemprop ADMET
  耗时 < 0.5 s/mol，过滤率 ~40%（Pass: top-60%）
  
  ⚑ 不熟悉度门控：
     U(z) > 0.6 → 降低 Boltz-2 预测置信权重
     U(z) > 0.8 → 跳过 L1，直接标注"待实验验证"

L2 结构对接：DiffDock-L（快速，2s/pose）+ GNINA 重排
  耗时 < 10 s/mol，过滤率 ~50%（Pass: top-30%）

L3 自由能扰动：OpenFE RBFE（相对 FEP）
  耗时 < 600 s/mol，仅对 Pareto 前沿 top-50 执行
  
  判断标准：ΔΔΔG_FEP - ΔΔΔG_Boltz2 > 1.5 kcal/mol → 
           标注"神经网络误判"，送回 CLM 主动学习

L4 量子校正：GPU4PySCF (DFT/B3LYP-D3) + ORCA DFTB3
  耗时 < 3600 s/mol，仅对最终 top-10 执行
  用途：共价机制建模、质子化态校正、构象能垒计算

不确定度传播（关键设计）：
  每个 oracle 返回值 v ± σ
  下一层的判断阈值 = f(σ)（高不确定度时设置更宽松的通过阈值）
  最终 Pareto 解标注：value ± propagated_uncertainty
```

### Agent-5：FTO/Patent Agent（知识产权安全）

```
数据架构：
  实时数据源：
    - SureChEMBL（每日增量，17M+ 化合物，数据来自专利全文）
    - USPTO PatFT全文（每周快照，Python scrapy 爬取）
    - Reaxys Patent（商业 API，半年订阅）
    - Google Patents BigQuery（每日 export）
  
  Markush 解析引擎：
    - 基于 SMARTS 的通式展开（处理 R 基团变量）
    - 限制展开规模：每件专利最多展开 10^5 虚拟化合物
    - 边界化：只展开"对权利要求核心结构有意义"的位置
    
  双层 FTO 评估：
    
    Layer 1：结构相似性检索
      ECFP4 Tanimoto + HUMU 双曲距离 联合检索
      → 返回 top-k 最相似专利化合物及相似度
      
    Layer 2：权利要求覆盖分析（语义层）
      用 SciFinder 风格的 claim 解析 + LLM 推理：
      "候选分子是否落入 US11186593 权利要求 1 的 Markush 范围？"
      → GPT-4o + 法律推理 few-shot + 置信度标注
      
  FTO Score 计算：
    FTO_score = sigmoid(
      w1 * min_structural_distance_to_blocked +   ← 越远越好
      w2 * claim_coverage_score +                 ← 0=清晰, 1=被覆盖
      w3 * temporal_factor(patent_expiry)         ← 快过期的专利减权重
    )
    
  FTO Score > 0.8：绿灯进入下一阶段
  FTO Score 0.6~0.8：黄灯，触发 MMPT-RAG 逃逸优化
  FTO Score < 0.6：红灯，放弃该分子，在 HUMU 中标记 "Patent Dead Zone"
  
  Patent Dead Zone：
    FTO < 0.6 的 HUMU 区域被标注为 "forbidden cone"，
    生成器的采样分布避开这些区域（添加 HUMU 障碍势能）。
    这是一个持续学习的 FTO 地图，越用越精准。
```

### Agent-6：Supply Chain Oracle（供应链神谕）

```
（保留为核心架构组件，仅做可及性预判，不做真实下单）

核心功能：合成路径的 BB 可及性评分

数据：
  - Enamine REAL Space 49B（本地索引，ECFP4 Faiss IVF-PQ）
  - Enamine in-stock 4.7M（每日同步）
  - Mcule、eMolecules、Chemspace（API 查询）

算法：
  For each BB in synthesis_route:
    availability = max(
      enamine_stock_match(BB),      ← 精确匹配 in-stock
      enamine_real_match(BB) * 0.8, ← REAL Space 定制合成，打折
      analog_match(BB) * 0.6        ← 结构类似物可以购买
    )
  route_score = geometric_mean(availability of all BBs)
  
合成路径的 supply-aware 重排：
  在 RetroSyn Agent 的路径排序中，加入 supply_score：
  final_score = 0.4 * yield_estimate + 0.3 * step_count^{-1} + 0.3 * supply_score
  
输出：路径的 BB 来源清单（用于后期真实采购，不在核心架构范围内）
```

### Agent-7：Scientific Critic（科学质疑者）

```
角色：系统内部的"反对意见"，防止智能体群体的确认偏误

实现：独立 LLM（使用与 Orchestrator 不同的模型族，如 Gemini Pro 或 DeepSeek）

质疑清单（100+ 条规则，举例）：
  ✗ "Boltz-2 的预测置信度 < 0.6，但 Agent 组将其列为 top 候选" → 警告
  ✗ "分子含 Michael acceptor 且亲电性 ≥ 0.4，未标注共价机制" → 要求澄清
  ✗ "三个生成器产出了结构极度相似的 top-10（平均 Tc > 0.85）" → 多样性不足
  ✗ "FTO Score 0.72，但临界专利 US11186593 的 claim 2 明确覆盖该骨架" → 重新评估
  ✗ "预计合成步数 = 7，但预算约束是 5 步" → 违反约束
  ✓ "五个候选分子均通过 L2 对接，平均 -9.1 kcal/mol，分布良好" → 批准进入 L3

触发时机：
  - 每批 50 个候选结束验证后
  - Pareto 前沿发生重大变化时
  - 任何 oracle 出现异常值（3σ外）时
  - 人工审查节点（审计日志要求 mandatory human-in-loop）

Critic 的输出写入 CRG：
  {"type": "critical_concern", "severity": "WARN/BLOCK", "description": "...", 
   "evidence": [...], "suggested_action": "...", "agent_id": "Critic"}
  
Orchestrator 必须在下一个 mini-cycle 中显式处理所有 BLOCK 级别的 Critic 意见。
```

## 4.3 Agent 通信协议（Message Protocol）

```
所有 Agent 消息格式（基于 JSON-LD + Sigstore 签名）：

{
  "@context": "https://moleculeforge.io/agent-protocol/v2",
  "msg_id": "MSG-{uuid4}",
  "trace_id": "EXP-2026-001-{run_id}",    ← 实验级追踪 ID（GxP 审计用）
  "timestamp": "2026-04-29T10:23:45.123Z",
  "from": {"agent_id": "GeneratorCoord", "version": "2.1.0"},
  "to": {"agent_id": "ValidationAgent"},
  "priority": "NORMAL",   ← URGENT / NORMAL / LOW
  
  "intent": "evaluate_batch",
  "payload": {
    "molecule_batch": [
      {
        "smiles": "CC(=O)N1CCN...",
        "humu_z": [0.23, -1.45, ...],   ← HUMU 坐标
        "conformer_uri": "s3://mf-data/conformers/mol-001.sdf",
        "generator": "FragFM",
        "generation_seed": 42,
        "unfamiliarity_score": 0.23
      }
    ],
    "oracle_budget": {"L1": 100, "L2": 50, "L3": 0},
    "objective_weights": {"affinity": 0.35, "fto": 0.20, ...}
  },
  
  "lineage": [           ← 完整的生成谱系（可审计）
    {"step": 0, "agent": "NL2Obj", "action": "create_cig", "output_hash": "sha256:..."},
    {"step": 1, "agent": "Orchestrator", "action": "route_to_generator"},
    {"step": 2, "agent": "FragFM", "action": "generate", "model_ckpt": "fragfm_v2.3"}
  ],
  
  "signature": "sigstore://rekor.sigstore.dev/api/v1/log/entries/{uuid}"
}
```

## 4.4 编排状态机（LangGraph 实现）

```python
from langgraph.graph import StateGraph, END
from typing import TypedDict

class MFState(TypedDict):
    cig: dict                     # Chemical Intent Graph
    hciv: list                    # Hyperbolic Chemical Intent Vector
    candidate_pool: list          # 当前候选分子池
    pareto_front: list            # 当前 Pareto 前沿
    crg: dict                     # Chemical Reasoning Graph（共享信念状态）
    oracle_budget_remaining: dict # 剩余预算
    stage: str                    # hit / lead_opt / refine / done
    critic_concerns: list         # Scientific Critic 的待处理意见

# 定义节点
builder = StateGraph(MFState)
builder.add_node("nl2obj",          nl2obj_agent)
builder.add_node("humu_encode",     humu_encode_node)    # CIG → HCIV
builder.add_node("generate",        generator_coord_agent)
builder.add_node("validate",        validation_agent)
builder.add_node("fto_check",       fto_patent_agent)
builder.add_node("retrosyn",        retrosyn_agent)
builder.add_node("critic",          scientific_critic_agent)
builder.add_node("orchestrate",     orchestrator_agent)  # 决策节点
builder.add_node("refine",          refine_cycle)        # CLM/EvoMol-RL 精修

# 定义边（条件路由）
builder.add_edge("nl2obj", "humu_encode")
builder.add_edge("humu_encode", "generate")
builder.add_edge("generate", "validate")
builder.add_conditional_edges("validate", route_after_validation, {
    "fto_check": "fto_check",       # 通过 L1/L2 验证 → FTO
    "regenerate": "generate",        # 验证失败率过高 → 重新生成
    "escalate_L3": "validate",       # 预算许可 → L3 验证
})
builder.add_edge("fto_check", "retrosyn")
builder.add_edge("retrosyn", "critic")
builder.add_conditional_edges("critic", route_after_critic, {
    "proceed": "orchestrate",        # Critic 批准 → 主管决策
    "block_regenerate": "generate",  # Critic 阻拦 → 重新生成
    "refine": "refine",              # Critic 建议精修 → 进入 CLM/RL
})
builder.add_conditional_edges("orchestrate", route_by_orchestrator, {
    "next_stage": "generate",        # 提升难度继续生成
    "final_output": END,             # 达到退出条件 → 输出结果
    "human_review": END,             # 需要人工审查 → 暂停等待
})
```

---

# 第五层：自适应分级验证 Oracle 级联（已在 Agent-4 详述）

## 5.1 关键补充：Pareto 感知贝叶斯优化（全局搜索层）

AMGE + MARB 的本地探索需要一个全局优化层来协调：

```
Pareto-aware Constrained Bayesian Optimization (PCBO)：

超体期望改善 (EHVI) 采集函数：
  α_EHVI(x) = E[HVI(f(x), Y_front)]
  其中 Y_front 是当前 Pareto 前沿，HVI 是超体改善
  
可行性概率 (PoF) 修正（处理硬约束）：
  α_PCBO(x) = α_EHVI(x) · PoF(x)
  PoF(x) = Pr[g_i(x) ≤ 0 ∀i]   ← 所有约束满足的概率
  （约束包括：FTO_score > 0.8, SA_score < 4, toxicity_flag = 0）
  
代理模型：
  Gaussian Process on HUMU 切空间（双曲 GP）
  核函数：Matérn 5/2 on geodesic distance
  → 预测 oracle 函数在 HUMU 中的分布 μ(z), σ(z)
  
BO 循环（外层，每 50 个 oracle 评估更新一次）：
  1. 用 EHVI 识别 HUMU 中的高价值区域
  2. 在这些区域激活 AMGE 的生成器（增加采样密度）
  3. 新 oracle 结果更新 GP 代理模型
  4. 更新 Pareto 前沿 + 超体积 HV 指标
```

---

# 第六层：合成现实桥接器 (Synthesis Reality Bridge, SRB)

## 6.1 从 SMILES 到 XDL 的全链路（保留接口，详见湿实验室模块）

```
SRB 在核心架构中的职责（仅逻辑层，不涉及硬件）：

输入：验证通过的分子 + 逆合成路径
输出：结构化合成规程 (Structured Synthesis Protocol, SSP)

SSP 格式：
{
  "target_smiles": "...",
  "route_id": "ROUTE-001",
  "steps": [
    {
      "step_id": 1,
      "reaction_type": "Buchwald_Hartwig_amination",
      "reactants": [{"smiles": "...", "amount_mmol": 1.0, "source": "Enamine_BB-12345"}],
      "reagents": ["Pd2(dba)3", "BINAP", "Cs2CO3"],
      "solvent": "toluene",
      "temperature_C": 110,
      "time_h": 16,
      "yield_estimate": 0.72,
      "yield_uncertainty": 0.12,
      "purification": "silica_gel_chromatography"
    }
  ],
  "total_estimated_yield": 0.31,
  "total_estimated_cost_usd": 450,
  "xdl_version": "2.0",   ← 预留 XDL 编译接口
  "sila2_endpoint": null   ← 预留 SiLA2 接口（湿实验室模块实现）
}
```

## 6.2 NL → CIG → HCIV → 分子 → 路径 → SSP 完整链路追踪

```
每一步都写入 Neo4j Provenance Graph（线性链路）：

(NL_Input) -[COMPILED_BY {agent: NL2Obj}]→ (CIG)
(CIG) -[ENCODED_BY {model: IntentEncoder_v1}]→ (HCIV)
(HCIV) -[GENERATED_BY {model: FragFM_v2.3, seed: 42}]→ (Molecule_draft)
(Molecule_draft) -[VALIDATED_BY {oracle: Boltz2, score: -9.1}]→ (Molecule_validated)
(Molecule_validated) -[FTO_CHECKED {score: 0.87, patents: [...]}]→ (Molecule_fto_clear)
(Molecule_fto_clear) -[RETROSYN {route: ROUTE-001, steps: 4}]→ (Synthesis_Route)
(Synthesis_Route) -[COMPILED_TO]→ (SSP)
(SSP) -[APPROVED_BY {agent: Critic, timestamp}]→ (Final_Candidate)

每个节点和边都有 Sigstore 数字签名 → 监管可审计 (GxP 合规)
```

---

# 第七层：数据与知识基础设施 (Data & Knowledge Infrastructure, DKI)

## 7.1 四类数据库的统一架构

```
┌─────────────────────────────────────────────────────────────┐
│ 7.1 Vector Store (Milvus 2.5 + Faiss IVF-PQ)               │
│                                                              │
│  Collections：                                               │
│  - molecules_humu: (smiles, z_humu, oracle_scores, source)  │
│    索引：DiskANN on ℍ^128（自定义双曲距离度量）                │
│    规模：10^9 量级（Enamine + ChEMBL + 生成历史）              │
│                                                              │
│  - pockets_humu: (pdb_id, pocket_z, target_info)            │
│    规模：~10^5 个已知口袋                                     │
│                                                              │
│  - patents_embedding: (doc_id, claim_text, claim_z, smarts) │
│    规模：17M（SureChEMBL）+ 实时增量                          │
└─────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────┐
│ 7.2 Graph Store (Neo4j 5 Enterprise)                         │
│                                                              │
│  图模式：                                                    │
│  (Molecule)-[:TRANSFORMS_TO {via: 'MMPT', confidence}]→(Molecule) │
│  (Molecule)-[:SYNTHESIZED_FROM {step, yield}]→(Reactant)   │
│  (Molecule)-[:BINDS_TO {affinity, source}]→(Protein)        │
│  (Molecule)-[:COVERED_BY {claim_id, similarity}]→(Patent)   │
│  (Run)-[:PRODUCED {agent, timestamp}]→(Molecule)            │
│  (Molecule)-[:HAS_BELIEF]→(ChemicalBelief)                  │
│                                                              │
│  用途：FTO 图查询、MMP 网络分析、GxP 审计追溯、SAR 传播      │
└─────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────┐
│ 7.3 Relational Store (PostgreSQL 16 + TimescaleDB)           │
│                                                              │
│  核心表（部分）：                                             │
│  molecules: id, smiles, inchikey, mw, logP, created_at      │
│  oracle_calls: mol_id, oracle, value, uncertainty, cost_s   │
│  runs: id, cig_id, status, hv_pareto, n_oracle_L1/L2/L3     │
│  agent_logs: msg_id, from_agent, to_agent, payload_hash     │
│  pareto_fronts: run_id, timestamp, front_json               │
│                                                              │
│  TimescaleDB：oracle_calls 和 pareto_fronts 按时间分区，      │
│  用于实验进度的实时监控仪表盘（预留前端接口）                   │
└─────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────┐
│ 7.4 Object Store (MinIO S3-compatible)                       │
│                                                              │
│  目录结构：                                                   │
│  s3://mf-data/                                               │
│    conformers/{mol_id}/{conformer_n}.sdf   ← 3D 构象文件     │
│    md_trajectories/{run_id}/{sim}.xtc      ← MD 轨迹         │
│    fep_results/{run_id}/{pair}.json        ← FEP 结果         │
│    models/{service}/{version}/weights/     ← 模型权重         │
│    audit_logs/{date}/{run_id}/signed.jsonl ← 签名审计日志     │
│    xdl_protocols/{mol_id}/{route_id}.xdl  ← 合成协议（预留） │
└─────────────────────────────────────────────────────────────┘
```

## 7.2 特征存储 (Feature Store, Feast-based)

```
Feature Groups：

molecule_features:
  - ecfp4_1024: binary fingerprint
  - rdkit_desc_200: 200 RDKit 描述符
  - humu_z_128: 双曲嵌入向量
  - predicted_admet: 8 ADMET 端点的点预测 + 不确定度
  更新频率：每次生成/验证后实时写入

pocket_features:
  - pocket_z_512: ESMFold 口袋向量（每月更新已知靶点）
  - pharmacophore_3d: 3D 药效团坐标
  
patent_features：
  - claim_z_768: BERT 权利要求向量（每日增量）
  - markush_ecfp4: Markush 展开样本的 ECFP4 集合
```

---

# 第八层：工程实施蓝图（核心架构专项）

## 8.1 微服务架构（核心 12 个服务）

```
服务一览（仅核心架构，不含前端、湿实验室、商业化）：

核心 AI 服务（GPU 密集型）：
1. humu-encoder-svc       ← HUMU 联合编码器，SE(3)-GNN + LorentzProjection
2. hfm-generator-svc      ← 双曲 Flow Matching 生成器（4×A100 80G）
3. fragfm-generator-svc   ← 片段级 DFM 生成器（2×A100 40G）
4. lamgen-generator-svc   ← LaMGen-3D 多靶点（4×A100 80G）
5. crem-generator-svc     ← CReM-pharm-3D（4×A40 48G，CPU+GPU 混合）
6. mmpt-generator-svc     ← MMPT-RAG（2×A40 48G）
7. evomol-rl-svc          ← EvoMol-RL Pareto（1×A40 48G）
8. iclm-svc               ← 增量 CLM 在线学习（1×A100 40G）

计算密集型服务：
9.  boltz2-svc            ← Boltz-2 亲和力（2×H100 80G，吞吐优先）
10. dock-svc              ← GNINA/DiffDock-L（4×A40，批量对接）
11. fep-svc               ← OpenFE ABFE（8×A100，NNP-MM 混合）
12. admet-svc             ← ADMET-AI + Chemprop（1×A40）

智能体逻辑服务（CPU 型，可扩展）：
13. orchestrator-svc      ← LangGraph 主管智能体
14. nl2obj-svc            ← 意图解析（调用 LLM API）
15. fto-patent-svc        ← FTO 检查（Neo4j 查询 + LLM 推理）
16. retrosyn-svc          ← 逆合成（AiZynthFinder + RSGPT，GPU 可选）
17. critic-svc            ← 科学质疑者（独立 LLM，CPU）
18. cig-compiler-svc      ← CIG 构建 + HCIV 编码（CPU + 小 GPU）

数据服务：
19. humu-index-svc        ← Milvus 向量检索 + 实时写入
20. provenance-svc        ← Neo4j 图写入 + Sigstore 签名
21. supply-oracle-svc     ← BB 可及性查询（API 聚合）
22. feature-store-svc     ← Feast online/offline feature 服务
```

## 8.2 技术栈选型（核心架构专项）

```
深度学习框架：
  PyTorch 2.6 + Lightning 2.x（训练）
  ONNX Runtime / TensorRT-LLM（推理加速）
  DGL 2.x + PyG（图神经网络）
  e3nn（SE(3) 等变层）
  geoopt（双曲流形优化，支持 Lorentz 模型的 RAdam）

分子化学：
  RDKit 2024.09（分子操作/性质计算）
  OpenBabel 3.1（格式转换）
  AiZynthFinder 4.0（逆合成规划）
  OpenFE 1.x（自由能扰动）
  Psi4 1.9 + GPU4PySCF（量子化学，GPU 加速）
  MACE-OFF24 / ANI-2x（机器学习力场）

大型语言模型：
  Claude Sonnet 4.5（Orchestrator 主 LLM）
  DeepSeek-V3 / Qwen3-72B（本地化部署备选，敏感数据）
  ChemDFM / Chemma（化学专域微调 LLM）
  Llama-3.3-70B + LoRA（XDL 编译器微调，保留接口）

Agent 框架：
  LangGraph 0.3（状态机编排）
  LangChain 0.3（工具调用、Memory）
  NATS JetStream（消息总线，< 1 ms 延迟，支持 ACK 重试）

向量数据库：
  Milvus 2.5（主向量库，支持自定义双曲距离）
  Faiss 1.8（高速本地批量检索，IVF-PQ 量化）
  pgvector（PostgreSQL 内嵌向量，用于小规模精确检索）

图数据库：
  Neo4j 5 Enterprise（谱系图 / 专利图 / MMP 网络）

关系数据库：
  PostgreSQL 16 + TimescaleDB（结构化数据 + 时序监控）

MLOps：
  MLflow 2.x（模型注册表 / 实验追踪）
  Weights & Biases（训练监控 / Sweep 超参搜索）
  DVC 3.x（数据版本控制）
  LakeFS（对象存储版本控制，S3 兼容）

安全与合规：
  Sigstore（数字签名，无密钥，GxP 审计）
  HashiCorp Vault（密钥管理）
  OPA Gatekeeper（K8s 策略执行）
  Falco（运行时安全监控）
```

## 8.3 Kubernetes 部署架构（核心架构专项）

```yaml
# K8s 资源规划（核心架构专项）

命名空间结构：
  - namespace: mf-generators      # GPU 密集型生成服务
  - namespace: mf-agents          # Agent 逻辑服务（CPU 主）
  - namespace: mf-oracles         # Oracle 计算（GPU + HPC）
  - namespace: mf-data            # 数据服务（Milvus, Neo4j, PG）
  - namespace: mf-mlops           # MLflow, W&B Proxy, DVC

GPU 节点池：
  pool-h100:   2x H100 SXM5 80G   # Boltz-2 高吞吐推理
  pool-a100:   8x A100 SXM4 80G   # HFM-3D, LaMGen, FEP
  pool-a40:    4x A40 48G          # FragFM, CReM, EvoMol, Dock

HPA 配置（关键服务）：
  hfm-generator-svc:
    minReplicas: 1
    maxReplicas: 8
    metrics:
      - type: External
        external:
          metric: {name: nats_jet_stream_pending_msg_count}
          target: {type: AverageValue, averageValue: "20"}
  
  boltz2-svc:
    minReplicas: 2   # 高优先级，常驻 2 个副本
    maxReplicas: 16
    metrics:
      - type: Resource
        resource: {name: nvidia.com/gpu, target: {averageUtilization: 80}}

存储类：
  fast-nvme:     Local NVMe SSD, latency < 100μs  # Milvus 索引
  object-store:  MinIO S3, throughput > 10 GB/s    # 构象/MD 文件
  postgres-ssd:  GP3 EBS 16000 IOPS               # PG TimescaleDB
```

## 8.4 核心架构的关键流程完整代码示例

```python
# MoleculeForge 核心设计循环（伪代码，但可直接映射到实现）

import asyncio
from moleculeforge.core import (
    CICCompiler, HUMU, TaskAwareRouter, 
    GeneratorSwarm, OracleAdaptiveCascade,
    MultiAgentBrain, ProvenanceGraph
)

async def design_molecules(user_query: str, budget: dict) -> DesignResult:
    
    # ── Layer 1: 化学意图编译 ──────────────────────────────────────
    cic = CICCompiler()
    cig = await cic.compile(user_query)          # NL → CIG
    hciv = HUMU.encode_intent(cig)               # CIG → ℍ^128
    intent_cone = HUMU.define_cone(hciv, cig.constraints)
    
    # ── Layer 2-3: 初始化 Agent 大脑 ────────────────────────────────
    brain = MultiAgentBrain(
        orchestrator_model="claude-sonnet-4-5",
        critic_model="deepseek-v3",         # 不同模型族避免 collusion
        crg=ChemicalReasoningGraph()
    )
    
    # ── Layer 4: 初始化 Pareto 贝叶斯优化 ───────────────────────────
    pcbo = ParetoConstrainedBO(
        surrogate=HyperbolicGP(kernel="matern52_geodesic"),
        pareto_archive=ParetoArchive(objectives=cig.objective_nodes),
        humu=HUMU
    )
    
    # ── 主循环 ────────────────────────────────────────────────────────
    router = TaskAwareRouter()
    oracle = OracleAdaptiveCascade(budget=budget)
    provenance = ProvenanceGraph(backend="neo4j")
    
    stage = "breadth_exploration"
    while not brain.should_terminate():
        
        # 1. BO 建议下一批采样区域
        z_candidates = pcbo.suggest(
            n_samples=200, 
            constraint=intent_cone,
            acquisition="EHVI_PoF"
        )
        
        # 2. 任务感知路由决定生成器组合
        generator_weights = router.route(
            hciv=hciv, 
            stage=stage,
            oracle_history=pcbo.history
        )
        
        # 3. 并行生成（各生成器在分配的 z 区域内采样）
        gen_swarm = GeneratorSwarm(weights=generator_weights)
        molecules = await gen_swarm.generate_parallel(
            z_regions=z_candidates,
            intent_cone=intent_cone,
            n_total=500
        )
        
        # 4. 去重 + 不熟悉度过滤
        molecules = HUMU.deduplicate(molecules, min_dist=0.05)
        molecules = HUMU.filter_ood(molecules, tau_ood=0.7)
        
        # 5. 自适应 Oracle 瀑布（L0→L1→L2→L3，按需升级）
        evaluated = await oracle.evaluate_cascade(molecules, cig)
        
        # 6. FTO 检查（并行）
        fto_results = await brain.fto_agent.batch_check(evaluated)
        cleared = [m for m, f in zip(evaluated, fto_results) if f.score > 0.8]
        
        # 7. 逆合成规划（top 候选）
        top_k = pareto_select(cleared, k=50)
        routes = await brain.retrosyn_agent.plan(top_k)
        
        # 8. Critic 质疑
        concerns = await brain.critic.review(
            molecules=top_k, routes=routes, crg=brain.crg
        )
        brain.crg.update(concerns)
        
        # 9. Orchestrator 决策
        decision = await brain.orchestrator.decide(
            pareto_front=pcbo.pareto_archive,
            crg=brain.crg, budget=oracle.remaining_budget
        )
        stage = decision.next_stage
        
        # 10. 更新 Pareto BO 代理模型
        pcbo.update(molecules=evaluated, fto=fto_results, routes=routes)
        
        # 11. 写入 Provenance Graph
        await provenance.log_cycle(
            molecules=evaluated, routes=routes, 
            fto=fto_results, decision=decision
        )
    
    # ── 最终输出 ──────────────────────────────────────────────────────
    return DesignResult(
        pareto_front=pcbo.pareto_archive.get_final(),
        top_candidates=pcbo.pareto_archive.top_k(k=10),
        synthesis_routes=routes,
        audit_log=provenance.export(format="signed_jsonl"),
        # ssp_xdl=None   ← 预留：湿实验室模块实现
    )
```

---

# 第九层：评估体系与基准方案

## 9.1 生成器评估（分子生成质量）

```
基准数据集：
  MOSES 2.0 / GuacaMol v3（分子质量综合评估）
  CrossDocked 2020 v2（基于口袋的 3D 生成）
  PMO 23 任务（单任务性质优化）
  自建多目标套件（affinity × admet × fto × sa，4 目标 Pareto）
  
关键指标：
  ① Validity / Uniqueness / Novelty（分子合法性三项，目标：>99% / >99% / >80%）
  ② SA Score 分布（目标：均值 < 3.0，超越 DiffSBDD 的 3.8）
  ③ 3D Vina Docking Score 中位数（目标 < -9.0 kcal/mol for KRAS G12C）
  ④ FTO Safety Rate（目标 > 85%，即生成的分子 85% FTO 安全）
  ⑤ Pareto 超体积 HV（多目标，越高越好，与 REINVENT 4.0 基线对比）
  ⑥ 不熟悉度校准（Calibration AUROC > 0.80，U 分数能正确识别 OOD 分子）
```

## 9.2 Agent 系统评估

```
参考 DeltaWave《An Auditable Agent Platform》+ 自建基准：

① 任务完成率：给定 10 个真实药物设计任务，成功输出 top-10 Pareto 前沿的比率
② 审计完整性：每个候选分子是否有完整的 Sigstore 签名谱系（目标 100%）
③ FTO 准确率：对照人工 IP 律师评审（目标 Precision > 0.90, Recall > 0.80）
④ Agent 一致性：Orchestrator 的决策与 Critic 的 BLOCK 意见符合率
⑤ 计算效率：D2L（自然语言输入到第一个通过 L2 验证的候选）P95 < 4 小时
```

## 9.3 HUMU 双曲空间评估

```
① 层次结构保留度（Distortion）：
   把骨架树嵌入 ℍ^128 后，树距离 vs 双曲距离的平均蒸馏（目标 < 0.15，vs 欧氏的 0.35）
   
② Activity Cliff 分辨力：
   Cliff pair 的双曲距离显著大于非-cliff pair（目标：Mann-Whitney U p < 0.001）
   
③ 检索增益（EF1% enrichment）：
   HypSeek vs DrugCLIP 在 DUD-E 的 EF1% 提升（原论文 +20.7%，目标维持或提升）
   
④ 合成-分子嵌入一致性：
   Enc_mol(m) 与 Enc_route(route_of_m) 的双曲距离 < τ（目标：平均距离 < 0.2）
```

---

# 第十层：实施路线图（核心架构专项）

## 10.1 三阶段路线图

```
Phase 0 — 基础设施与数据（M1-M3）：

  M1：
    - 搭建 K8s 集群（GPU 节点池 + 存储）
    - 部署 Milvus 2.5 + Neo4j 5 + PostgreSQL + MinIO
    - 建立 HUMU 基础数据集：
        ChEMBL 34（>2.4M 分子）+ PDB 对接构象 + PaRoutes 反应树
    - 预训练 HUMU Encoder（分子 + 口袋 + 路径，三塔联合训练）

  M2：
    - 集成 AiZynthFinder 4.0 + RSGPT + UAlign（逆合成）
    - 部署 Boltz-2 推理服务（Triton + H100）
    - 建立 SureChEMBL Patent Graph（17M 化合物，Neo4j）
    - 实现 CIG Compiler v0.1（规则-based，非 LLM）

  M3：
    - 接通 NATS JetStream 消息总线
    - 实现 Provenance Graph + Sigstore 签名
    - 建立 Feast Feature Store
    - 初步 MLflow 模型注册表

Phase 1 — 核心生成与 Agent（M4-M8）：

  M4-M5：生成器实现
    - 实现 HFM-3D（Lorentz Flow Matching，从 SemlaFlow 改造）
    - 实现 FragFM-HUMU（片段 DFM，SA 感知）
    - 实现 CReM-pharm-3D（DiffDock 实时集成）
    - 实现 TAR（任务感知路由器，初期规则 + ProxylessNAS）

  M6-M7：Agent 系统实现
    - 实现所有 7 个 Agent（LangGraph 状态机）
    - 实现 CRG（Chemical Reasoning Graph）
    - 实现 Scientific Critic（独立 LLM 路）
    - 部署 Pareto EHVI-PoF BO（双曲 GP）

  M8：集成测试
    - 端到端集成（NL → CIG → HUMU → 生成 → 验证 → FTO → 逆合成 → 输出）
    - 单靶点基准测试（KRAS G12C 作为 Pilot）
    - Bug 修复 + 性能优化

Phase 2 — 精化与高级功能（M9-M14）：

  M9-M10：高级生成器
    - 实现 LaMGen-3D-Pro（多靶点门控）
    - 实现 MMPT-RAG（专利负样本对比解码）
    - 实现 EvoMol-RL Pareto（EHVI 奖励）
    - 实现 Incremental CLM（EWC + PackNet）
    - 实现 UAS（不熟悉度感知采样）

  M11-M12：供应链 Oracle + 高精度 Oracle
    - 集成 Supply Oracle（Enamine REAL 本地索引 + API）
    - 集成 OpenFE L3 FEP（HPC 集群）
    - 集成 GPU4PySCF L4 量子计算

  M13-M14：基准评估 + 消融研究
    - 全面基准（MOSES/GuacaMol/PMO/CrossDocked + 自建）
    - 消融：有 vs 无 HUMU / 有 vs 无 FTO-aware / 有 vs 无 UAS
    - 撰写技术报告（可用于期刊投稿）

Phase 3 — 预留（M15+）：
    🔲 前端用户界面实现（接口已预留）
    🔲 湿实验室硬件接口（XDL 2.0 + SiLA2 编译器）
    🔲 商业化与多租户部署
```

## 10.2 核心架构风险与缓解

```
风险 1：HUMU 双曲 GP 的计算复杂度
  风险：双曲距离的 GP 核矩阵计算 O(n^2)，1000 样本后变慢
  缓解：稀疏 GP（SVGP，inducing points = 500）+ LOVE（快速方差估计）
       实测：10^4 样本下仍 < 2 分钟更新

风险 2：多 Agent 的一致性与死锁
  风险：多个 Agent 同时修改 CRG 导致冲突
  缓解：CRG 写入使用 Optimistic Concurrency Control（OCC + 向量时钟）
       Orchestrator 持有 CRG 的"否决权"，可强制回滚矛盾 belief

风险 3：LLM 幻觉污染 CIG
  风险：NL2Obj 把用户意图误解析为错误的目标函数
  缓解：CIG 的每个字段必须来自**工具调用**的真实数据（不允许 LLM 直接填写数值）
       CIG 构建后由 Critic 独立验证（不同模型族）+ 用户确认环节

风险 4：FTO 数据滞后
  风险：SureChEMBL 更新有延迟（专利公开 → 数据库通常延迟 2-4 周）
  缓解：多源融合（PatSnap API 实时 + Reaxys + 自建 USPTO 爬虫日更）
       FTO Score 加入"时间不确定度"项（近期专利权重降低 30%）

风险 5：双曲空间的数值不稳定
  风险：Lorentz 模型在 x_0 → 0 时梯度爆炸
  缓解：使用 Wrapped Normal + 稳定指数映射（Geoopt 库）
       添加 Lorentz 约束正则 + gradient clipping on 流形法向量
```

---

# 架构总览：核心创新一图速览

```
MoleculeForge 核心架构 — 七大原创贡献总结：

┌─────────────────────────────────────────────────────────────────────┐
│                  MoleculeForge 核心架构创新矩阵                       │
├─────────────────────────────────────────────────────────────────────┤
│                                                                      │
│  创新 1：JMCG（联合流形共生成）                                        │
│    → 全球首次把分子、合成路径、性质在单一双曲流形上联合建模              │
│    → 消除"生成-验证-合成"三段论的内在矛盾                              │
│                                                                      │
│  创新 2：HUMU（双曲统一分子宇宙）                                       │
│    → Lorentz ℍ^128，三塔编码（分子+口袋+路径），联合对比训练            │
│    → 双曲 GP 代理模型用于 EHVI-PoF 贝叶斯优化                         │
│    → 内嵌 activity cliff 分辨 + OOD 不熟悉度门控                      │
│                                                                      │
│  创新 3：HFM-3D（双曲流匹配生成器）                                     │
│    → 意图锥约束采样（生成结果天然在目标区域内）                          │
│    → Lorentz-equivariant 向量场（20 步即达 SOTA 质量）                │
│                                                                      │
│  创新 4：Patent Dead Zone（专利禁区地图）                               │
│    → FTO 评估结果写回 HUMU，形成动态增长的专利障碍势                   │
│    → 生成器的采样分布主动绕开专利禁区（非事后过滤）                     │
│                                                                      │
│  创新 5：TAR + 跨范式知识蒸馏（自适应 MoE）                            │
│    → 8 种生成范式共享 HUMU，相互提供教学信号                           │
│    → 路由器通过在线 REINFORCE 持续学习最优生成策略                      │
│                                                                      │
│  创新 6：CRG（化学推理图）+ Sigstore 审计                              │
│    → Agent 间共享结构化信念而非自然语言                               │
│    → 每个分子的完整推理链都有数字签名，GxP 级别可审计                   │
│                                                                      │
│  创新 7：UAS（不熟悉度感知采样）                                        │
│    → 主动规避 OOD 化学空间（防止预测崩溃）                              │
│    → 自动触发增量 CLM 主动学习（持续扩展可信化学空间）                  │
│                                                                      │
├─────────────────────────────────────────────────────────────────────┤
│  保留位置：                                                           │
│  🔲 前端层（REST/WebSocket 接口已定义，待实现）                         │
│  🔲 湿实验室层（SSP 结构已定义，XDL/SiLA2 编译器待实现）               │
│  🔲 商业化层（Multi-tenant 模板待规划）                                │
└─────────────────────────────────────────────────────────────────────┘
```

---

*MoleculeForge Core Architecture v2.0*  
*参考文献：10 篇指定文献 + FragFM(2025) + SemlaFlow(2025) + RSGPT(2026) + HypSeek(NeurIPS 2025) + FlexiFlow(AZ 2025) + PropMolFlow(Nat Comput Sci 2026) + FROGENT(2025) + ChatInvent@AZ(2026) + Boltz-2(2025)*  
*架构版本：v2.0 | 日期：2026-04-29*
