# SAM 源码分析理解

## 概述

SAM（Segment Anything Model）示例是 GGML 中**规模最大、架构最复杂**的示例。它实现了 Meta AI 的 Segment Anything 模型，用于图像分割任务。用户可以通过点提示（point prompt）或框提示（box prompt）指定目标区域，模型会自动生成对应的分割掩码。该示例完整展示了如何用 GGML 构建和推理一个包含 ViT 图像编码器、Prompt 编码器和 Mask 解码器的复合视觉模型。

## 文件结构

| 文件 | 功能 |
|------|------|
| `sam.cpp` | 完整的 SAM 推理实现（约 2300 行） |
| `convert-pth-to-ggml.py` | PyTorch 模型到 GGML 格式转换 |
| `CMakeLists.txt` | 构建配置 |

## 模型架构

SAM 由三个主要组件构成：

### 1. 图像编码器 (Image Encoder) — ViT 架构

```
输入图像 [1024, 1024, 3]
  → patch_embed: Conv2D(16x16, stride=16) → [64, 64, 768]
  → + pos_embed (位置编码)
  → 12x Transformer Encoder Blocks:
      → LayerNorm → Window/Global Self-Attention → 残差连接
      → LayerNorm → MLP (4x expansion) → 残差连接
  → neck: Conv2D → LayerNorm → Conv2D → LayerNorm
  → embd_img [64, 64, 256]
```

关键设计：
- **混合注意力**：部分层使用全局注意力（global attention），部分使用窗口注意力（window attention，14x14 窗口），通过 `global_attn_indices` 指定
- **相对位置编码**：`ggml_get_rel_pos` + `ggml_add_rel_pos_inplace` 实现 2D 相对位置偏置
- **窗口分区/反分区**：`ggml_win_part` / `ggml_win_unpart` 将特征图划分为窗口，在窗口内做局部注意力

### 2. Prompt 编码器 (Prompt Encoder)

支持两种提示类型：
- **点提示**：将 (x, y) 坐标通过高斯位置编码映射为嵌入向量
- **框提示**：将 (x1, y1, x2, y2) 两个角点的坐标嵌入

位置编码使用 `ggml_sam_sin` / `ggml_sam_cos` 自定义操作（通过 `ggml_map_custom1` 注册），将坐标乘以 2π 后取 sin/cos 拼接。

### 3. Mask 解码器 (Mask Decoder)

```
prompt embeddings + image embeddings
  → 2x Two-Way Transformer Layers:
      → self_attn (token 自注意力)
      → cross_attn_token_to_img (token 到图像的交叉注意力)
      → MLP
      → cross_attn_img_to_token (图像到 token 的交叉注意力)
  → final_attn_token_to_image
  → output_upscaling (反卷积上采样)
  → hypernetwork MLPs (每个 mask token 独立的 MLP)
  → masks [256, 256] + IoU predictions
```

## 核心流程分析

### 1. 模型加载 (`sam_model_load`)

SAM 使用**自定义二进制格式**（非 GGUF），加载流程：
1. 验证 magic number `0x67676d6c`
2. 读取超参数（encoder state/layer/head 大小等）
3. 计算上下文大小（精确计算所有张量所需内存）
4. 创建 ggml 上下文（`no_alloc=false`，直接分配内存）
5. 创建所有权重张量并建立名称映射
6. 从文件逐个读取张量数据

权重存储精度：
- 大部分权重以 `GGML_TYPE_F16` 存储（节省空间）
- 偏置和 LayerNorm 参数以 `GGML_TYPE_F32` 存储（保持精度）

### 2. 图像预处理 (`sam_image_preprocess`)

```
原图 → 缩放最长边到 1024 → 双线性插值 → ImageNet 标准化 → 1024x1024 padding
```

- 缩放比例：`scale = max(nx, ny) / 1024.0`
- 归一化：`(pixel - mean) / std`，使用 ImageNet 标准参数
- 手动实现双线性插值（与 PyTorch 的 `F.interpolate` 等价）

### 3. 图像编码 (`sam_encode_image`)

构建并执行 ViT 编码器计算图：

1. **Patch Embedding**：`ggml_conv_2d_sk_p0` 实现 stride=kernel_size 的卷积
2. **Transformer Block** 循环：
   - LayerNorm → QKV 投影 → 多头注意力（含相对位置编码）
   - 窗口注意力：`ggml_win_part` → 注意力计算 → `ggml_win_unpart`
   - MLP：`ggml_mul_mat` → `ggml_gelu` → `ggml_mul_mat`
3. **Neck**：Conv2D + LayerNorm2d
4. 结果存入 `state.embd_img`

关键技巧：使用 `ggml_disconnect_node_from_graph` 将输出节点从图中断开，使其数据保持不变。

### 4. Mask 解码 (`sam_decode_mask`)

解码过程构建在 `sam_build_fast_graph` 中：

1. 编码 prompt → sparse/dense embeddings
2. 生成 dense positional encoding（高斯随机矩阵投影 → sin/cos）
3. Two-Way Transformer 解码
4. 上采样 + Hypernetwork MLPs 生成掩码
5. IoU 预测头生成质量分数

### 5. 掩码后处理 (`sam_write_masks`)

```
低分辨率掩码 [256, 256]
  → 双线性插值上采样到 1024x1024
  → 去除 padding
  → 缩放到原始图像尺寸
  → 阈值化（mask_threshold）
  → 稳定性分数过滤（stability_score_threshold）
  → IoU 过滤（iou_threshold）
  → 保存为 PNG
```

使用**稳定性分数**（stability score）进行二次过滤：
```
stability_score = count(mask > threshold + offset) / count(mask > threshold - offset)
```

## 自定义操作

SAM 注册了两个自定义操作：
- `ggml_sam_sin`：逐元素 sin 计算，多线程并行
- `ggml_sam_cos`：逐元素 cos 计算，多线程并行

通过 `ggml_map_custom1` 注册，使用 `GGML_N_TASKS_MAX` 实现最大并行度。

## 展示的 GGML 关键特性

1. **ViT 架构实现**：完整的 Vision Transformer 编码器，含窗口注意力和全局注意力
2. **相对位置编码**：2D 相对位置偏置的实现
3. **窗口分区操作**：`ggml_win_part` / `ggml_win_unpart` 用于高效局部注意力
4. **交叉注意力**：Two-Way Transformer 中的双向交叉注意力机制
5. **自定义操作注册**：`ggml_map_custom1` 注册 sin/cos 操作
6. **`ggml_gallocr` 精确内存分配**：根据计算图自动分配恰好所需的内存
7. **混合精度权重**：F16 权重 + F32 偏置的混合存储策略
8. **多阶段推理**：图像编码（慢）与 prompt 解码（快）分离执行
9. **`ggml_graph_compute_helper`**：简化计算图执行的辅助函数
10. **图节点断开**：`ggml_disconnect_node_from_graph` 保留中间结果
