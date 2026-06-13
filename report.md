# ERRNet 模型结构与 Gate 消融分析报告

## 一、整体框架

ERRNet 是一个单图反射去除网络。完整框架由三个模块组成：

```
输入图像 I (3×H×W)
    │
    ├──→ Laplacian Pyramid ──→ L (12×H×W)  ───────────────┐
    │                                                       │
    ├──→ RDNet ──────────────→ M (1×H×W) 反射掩码 ────────┬─┤
    │                                                       │ │
    └──→ ERRNet (DRNet) ─────────────────────────────────→ 输出 T (3×H×W)
                            ↑
                     Gate 调制（可选）
```

- **Laplacian Pyramid**：提取 4 个尺度的拉普拉斯边缘图（3×4=12 通道），提供多尺度结构先验
- **RDNet**：反射检测网络，预测反射置信图 $M$
- **ERRNet (DRNet)**：主去反射网络，编码器-残差块-解码器结构
- **Gate**：将 $M$（及可选的 $L$）转化为特征调制信号

---

## 二、RDNet 网络结构

### 2.1 输入

默认输入为 RGB + Laplacian Pyramid 共 **15 通道**（`--rdnet_no_laplacian` 时退化为 3 通道 RGB）。

### 2.2 网络架构

```
Input (15×H×W)
    │
    Conv2d(15→32, k3, p1) + ReLU          ← Head
    │
    ResBlock(32) × 3                        ← Body
    │
    Conv2d(32→16, k3, p1) + ReLU           ← Tail
    Conv2d(16→1, k3, p1)
    Sigmoid
    │
    Output M (1×H×W)
```

ResBlock 结构：
```
x → Conv3×3 → ReLU → Conv3×3 → + → ReLU → out
     └──────────────────────────┘
```

### 2.3 训练监督

RDNet 使用**梯度比较掩码**作为监督信号：

$$\text{Mask}_{\text{GT}} = \mathbb{1}\big[\nabla I_{\text{blend}} > \nabla I_{\text{clean}}\big]$$

即混合图中梯度强于干净图的区域标记为反射区域。

### 2.4 损失函数

$$\mathcal{L}_{\text{RDNet}} = \lambda_{\text{BCE}} \cdot \text{BCE}(M, M_{\text{GT}}) + \lambda_{\text{TV}} \cdot \text{TV}(M)$$

---

## 三、Concat 结构

### 3.1 原理

将 RDNet 输出的反射掩码 $M$ 直接拼接为 ERRNet 的第 4 个输入通道：

$$I_{\text{concat}} = [I_{\text{RGB}}, M] \in \mathbb{R}^{4 \times H \times W}$$

### 3.2 信息流

```
输入 I ──→ Laplacian Pyramid ──→ RDNet ──→ M
                                              │
输入 I ────────────────────────→ [I, M] ──→ ERRNet ──→ 输出 T
```

### 3.3 特点

- 最直接的融合方式，将掩码作为额外输入通道
- ERRNet 第一层卷积自行学习如何利用 $M$ 信息
- 无额外可学习参数，完全依赖主网络对 $M$ 的隐式利用
- **局限**：掩码仅在输入层注入，深层特征无法直接感知反射信息

---

## 四、Simple Gate 结构

### 4.1 原理

在 ERRNet 瓶颈处（残差块之后、解码器之前）用 $M$ 生成逐通道空间注意力图，对标量 $\alpha$ 进行调制：

$$A = \sigma\big(\text{Conv}_{1\times1}(M, 1 \to 256)\big) \in \mathbb{R}^{256 \times \frac{H}{2} \times \frac{W}{2}}$$

$$F' = F \cdot (1 + \alpha \cdot A), \quad \alpha \in \mathbb{R}$$

### 4.2 信息流

```
输入 I ──→ Laplacian Pyr ──→ RDNet ──→ M ──→ Conv1×1(1→256) ──→ Sigmoid ──→ A
                                                                              │
输入 I ──→ Encoder ──→ ResBlocks×13 ──→ F ──→ [×] ──→ F' ──→ Decoder ──→ 输出
                                            α ──→ [×]
```

### 4.3 特点

| 属性 | 值 |
|------|-----|
| Gate 输入 | $M$ (1 通道，来自 RDNet) |
| 可学习参数 | Conv1×1: 1×256+256=512，α: 1 |
| α 初始化 | `softplus(-2.25) ≈ 0.095`（接近恒等映射） |
| 调制位置 | 瓶颈层（$H/2 \times W/2$） |
| 调制方式 | 逐像素、所有通道共享同一空间注意力 + 统一标量强度 |

### 4.4 设计优势

1. **极简参数**（~512），不易过拟合
2. **保守初始化**，训练初期 gate 接近恒等映射，网络先学会基础去反射再逐步利用掩码
3. **标量 α** 对所有通道统一约束，避免逐通道自由度导致的训练不稳定
4. $M$ 已经过 RDNet 的层次化处理，蕴含了从 Laplacian 多尺度结构到反射置信度的语义压缩

---

## 五、Structure-Aware Gate 结构（Mlocal Gate）

### 5.1 原理

在 Simple Gate 基础上引入 **Laplacian 多尺度结构先验** $L$（绕过 RDNet），与 $M$ 联合生成**逐通道**空间注意力：

$$L_f = \text{ReLU}\big(\text{Conv}_{1\times1}(L, 12 \to 256)\big)$$

$$A = \sigma\big(\text{Conv}_{1\times1}(\text{ReLU}(\text{Conv}_{1\times1}([L_f, M], 257 \to 64)), 64 \to 256)\big)$$

$$F' = F \cdot (1 + \alpha_c \cdot A_c), \quad \alpha_c \in \mathbb{R}^{256}$$

### 5.2 信息流

```
输入 I ──→ Laplacian Pyr ──→ L ──→ Conv1×1(12→256) ──→ L_f ──┐
                                                                 ├──→ [L_f, M] ──→ Gate Conv ──→ A
输入 I ──→ Laplacian Pyr ──→ RDNet ──→ M ─────────────────────┘        (257→64→256)
                                                                              │
输入 I ──→ Encoder ──→ ResBlocks×13 ──→ F ──→ [×] ──→ F' ──→ Decoder ──→ 输出
                                            α_c ──→ [×]
```

### 5.3 与 Simple Gate 的关键差异

| 维度 | Simple Gate | Structure-Aware Gate |
|------|-------------|---------------------|
| Gate 输入 | $M$ (1ch) | $[L_f, M]$ (257ch) |
| Laplacian 信息 | 仅经 RDNet 压缩后间接使用 | 直接输入 Gate |
| α 维度 | 标量 (1) | 逐通道向量 (256) |
| α 初始化 | softplus(-2.25) ≈ 0.095 | softplus(1.0) ≈ 1.313 |
| Gate 参数量 | ~512 | ~36,480 |
| 训练初期调制强度 | ~5% | ~65% |

---

## 六、Structure-Aware Gate 效果不佳的原因分析

实验结果表明，structure_aware gate 在所有基准数据集（CEILNet、real20、SIR²）上均不如 simple gate。以下从多个维度进行分析：

### 6.1 信息冗余：Laplacian 结构信息已被 RDNet 压缩到 M 中

这是最根本的原因。RDNet 的输入本身就是 $[I, L]$（RGB + Laplacian Pyramid，共 15 通道）。经过 RDNet 的三层 ResBlock 层次化处理后，Laplacian 提供的多尺度结构信息已经被**提取、筛选、压缩**到单通道反射置信图 $M$ 中：

- $M$ 中高响应区域 = 反射边缘（RDNet 学会区分反射边缘与场景边缘）
- $M$ 中低响应区域 = 场景自身结构

此时在 Gate 中再次注入原始 Laplacian $L$，相当于向已经包含结构信息的 $M$ 中重复添加**未经筛选的原始边缘信号**。$L$ 包含了所有边缘——既有反射边缘也有场景纹理边缘——而 Gate 模块需要从 257 维输入中重新学习区分，这在有限数据下是低效的。

### 6.2 初始化不当导致训练初期过度调制

Simple gate 的 $\alpha$ 初始化为 $\text{softplus}(-2.25) \approx 0.095$，训练初期调制幅度约 5%，网络行为接近无 gate 的基线——先在像素空间学会去反射，再逐步利用 gate 进行精细调制。

Structure-aware gate 的 $\alpha_c$ 初始化为 $\text{softplus}(1.0) \approx 1.313$，配合随机初始化的 gate conv（sigmoid 输出约 0.5），训练初期调制幅度约 **65%**——网络从第一步起就面临剧烈的逐通道特征调制，而此时的 gate 权重还是随机的，相当于在瓶颈特征上施加了强随机扰动。这导致：

1. 编码器-解码器的预训练权重在训练初期被严重破坏
2. 优化器需要同时学习有用的 gate 行为和恢复被破坏的特征表示
3. 训练轨迹更长、更不稳定，容易陷入局部最优

### 6.3 参数冗余与过拟合

Structure-aware gate 有 ~36K 参数（vs simple gate 的 ~512），增加了约 70 倍。在有限的真实训练数据（250 张非对齐 + 89 张对齐）下，这些额外参数主要用于：

- 将 12 通道 Laplacian 投影到 256 通道（3,072 参数）
- 257→64→256 的两层 gate conv（32,896 参数）
- 逐通道 α（256 参数）

这些参数需要从数据中学习"哪些 Laplacian 边缘是反射、哪些是场景"，而这本质上正是 RDNet 已经完成的任务。因此，额外参数不仅没有提供新信息，反而引入了过拟合风险。

### 6.4 逐通道 α 的自由度并未带来收益

Simple gate 的标量 α 对所有 256 个通道使用统一的调制强度。Structure-aware gate 的逐通道 α 允许每个通道学习不同的调制强度。然而，实验结果表明这种额外自由度并未转化为性能提升。

可能的原因：
- 瓶颈层 256 个通道中，许多通道编码的是底层纹理/颜色信息，与反射无关，不需要被 gate 调制
- 标量 α 作为"全局调制强度"已经足够表达反射区域的抑制程度
- 逐通道 α 在没有足够约束的情况下，某些通道可能学到极端调制值，破坏特征的一致性

### 6.5 联合训练中的梯度冲突

在 joint training 模式下（`rdnet_freeze=False`），RDNet 和 ERRNet 同时更新。对于 simple gate：

- RDNet 的梯度主要来自 Mask L1 loss（合成数据）+ 通过 Gate 回传的梯度
- Gate 的梯度路径短：$M \to \text{Conv1×1} \to \text{Sigmoid} \to F'$

对于 structure_aware gate：

- $L$ 通过 `lap_proj` 也向 Gate 传递梯度
- $L$ 的梯度与 $M$ 的梯度在 Gate 输入处叠加
- 由于 $L_f$ 有 256 通道而 $M$ 仅 1 通道，$L$ 相关的梯度在数值上占主导
- 这导致 Gate 更关注 Laplacian 原始边缘特征，而非 RDNet 学到的反射语义

---

## 七、总结

| 方案 | Gate 输入 | α 维度 | 参数量 | 效果 |
|------|-----------|--------|--------|------|
| Concat | $M$ (输入层拼接) | — | 0 | 基线 |
| Simple Gate | $M$ (瓶颈调制) | 标量 | ~512 | **最优** |
| Per-Channel Gate | $M$ (瓶颈调制) | 向量 (256) | ~512 | 次优 |
| Structure-Aware Gate | $[L, M]$ (瓶颈调制) | 向量 (256) | ~36K | 最差 |

**核心结论**：RDNet 已将 Laplacian 结构信息充分压缩到反射掩码 $M$ 中，再次注入原始 Laplacian 不会提供额外有效信息，反而引入冗余参数、不稳定的初始化以及联合训练中的梯度冲突。**Simple Gate** 以最少的参数和最稳健的初始化，在标量 α 约束下实现了最优的反射感知特征调制。
