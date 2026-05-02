# GPT-J 源码分析理解

## 一、项目概览

本目录是 ggml 框架下 GPT-J 6B 模型的完整推理实现。GPT-J 是一个 60 亿参数的开源 Transformer 语言模型，本目录通过纯 C/C++ 在 ggml 张量库上实现了该模型的本地 CPU 推理，无需 GPU。

目录构成：

| 文件 | 作用 |
|------|------|
| `main.cpp` | 核心推理：模型加载、前向计算、自回归生成 |
| `quantize.cpp` | 模型量化工具：将 f16/f32 模型压缩为低精度格式 |
| `convert-h5-to-ggml.py` | 模型转换：将 HuggingFace PyTorch 格式转为 ggml 二进制格式 |
| `download-ggml-model.sh` | 下载已转换好的 ggml 格式模型 |
| `download-model.sh` | 下载原始 HuggingFace 模型的指引脚本 |
| `CMakeLists.txt` | 构建配置，编译两个可执行文件 `gpt-j` 和 `gpt-j-quantize` |

---

## 二、模型架构（main.cpp）

### 2.1 超参数结构体 `gptj_hparams`

```
n_vocab = 50400    // 词表大小
n_ctx   = 2048     // 最大上下文长度（即 KV cache 容量上限）
n_embd  = 4096     // 嵌入维度 / 隐藏层维度
n_head  = 16       // 注意力头数（每头维度 = 4096/16 = 256）
n_layer = 28       // Transformer 层数
n_rot   = 64       // 旋转位置编码（RoPE）的旋转维度
ftype   = 1        // 权重存储格式（1 = float16）
eps     = 1e-5f    // LayerNorm 的 epsilon
```

GPT-J 6B 的关键架构特征：使用 **Rotary Position Embedding（RoPE）** 而非传统的绝对位置编码，旋转维度 `n_rot=64` 意味着每个注意力头中前 64 维参与旋转编码。

### 2.2 层结构体 `gptj_layer`

每一层 Transformer 包含以下权重张量：

**层归一化：**
- `ln_1_g` / `ln_1_b`：LayerNorm 的 scale（gamma）和 bias（beta），float32 存储

**自注意力（Self-Attention）：**
- `c_attn_q_proj_w`：Query 投影矩阵，形状 [n_embd, n_embd]
- `c_attn_k_proj_w`：Key 投影矩阵，形状 [n_embd, n_embd]
- `c_attn_v_proj_w`：Value 投影矩阵，形状 [n_embd, n_embd]
- `c_attn_proj_w`：输出投影矩阵，形状 [n_embd, n_embd]

与 GPT-2 将 Q/K/V 合并为单个矩阵不同，GPT-J 将 Q/K/V 拆分为三个独立投影矩阵。

**前馈网络（Feed-Forward Network）：**
- `c_mlp_fc_w`：第一层全连接权重，形状 [n_embd, 4*n_embd]（4096 → 16384）
- `c_mlp_fc_b`：第一层偏置，形状 [4*n_embd]
- `c_mlp_proj_w`：第二层全连接权重，形状 [4*n_embd, n_embd]（16384 → 4096）
- `c_mlp_proj_b`：第二层偏置，形状 [n_embd]

注意：GPT-J 的 FFN 扩展比为 4x，激活函数为 **GELU**（而非 ReLU）。

### 2.3 模型结构体 `gptj_model`

顶层模型还包含：
- `ln_f_g` / `ln_f_b`：最终的 LayerNorm
- `wte`：词元嵌入矩阵（token embedding），形状 [n_embd, n_vocab]
- `lmh_g`：语言模型头（输出投影），形状 [n_embd, n_vocab]
- `lmh_b`：语言模型偏置，形状 [n_vocab]
- `memory_k` / `memory_v`：KV cache，形状均为 [n_embd * n_layer * n_ctx]，以 float16 存储

关键设计：`memory_k` 和 `memory_v` 是预分配的固定大小张量，用于缓存所有层、所有历史位置的 Key 和 Value，避免重复计算。它们在模型加载时一次性分配，推理过程中原地更新。

---

## 三、模型加载流程（`gptj_model_load`）

### 3.1 ggml 二进制文件格式

文件布局：

```
[magic: 0x67676d6c ("ggml")]
[n_vocab] [n_ctx] [n_embd] [n_head] [n_layer] [n_rot] [ftype]
[n_vocab 个词表条目：每个为 (len, bytes)]
[若干张量：每个为 (n_dims, name_len, type, dims..., name, data)]
```

### 3.2 加载过程

1. **验证魔数**：确认文件以 `0x67676d6c`（"ggml"的十六进制）开头
2. **读取超参数**：7 个 int32 值，并从中提取量化版本号 `qntvr = ftype / GGML_QNT_VERSION_FACTOR`
3. **加载词表**：逐个读取 token 字符串，建立 `token_to_id` 和 `id_to_token` 的双向映射
4. **计算内存需求**：精确计算所有权重张量 + KV cache 所需的总字节数，用 `ggml_row_size` 根据数据类型（wtype）计算每行大小
5. **创建 ggml 上下文**：一次性 `ggml_init` 分配全部内存
6. **分配张量并建立名称映射**：先创建所有张量对象，再通过 `model.tensors[name] = tensor` 建立从权重名到张量指针的映射
7. **逐个加载权重**：读取每个张量的元数据（维度、名称、类型），验证形状匹配后，将数据直接读入张量的内存区域

**核心设计理念**：所有内存在模型加载阶段一次性分配，推理期间零内存分配。权重以连续方式存储在 ggml context 管理的内存池中。

### 3.3 权重名称映射

加载时通过字符串匹配将文件中的权重名映射到代码中的张量变量：

| 文件中的名称 | 代码变量 |
|-------------|---------|
| `transformer.wte.weight` | `model.wte` |
| `transformer.ln_f.weight` | `model.ln_f_g` |
| `transformer.ln_f.bias` | `model.ln_f_b` |
| `lm_head.weight` | `model.lmh_g` |
| `lm_head.bias` | `model.lmh_b` |
| `transformer.h.{i}.ln_1.weight` | `layer.ln_1_g` |
| `transformer.h.{i}.attn.q_proj.weight` | `layer.c_attn_q_proj_w` |
| `transformer.h.{i}.attn.k_proj.weight` | `layer.c_attn_k_proj_w` |
| `transformer.h.{i}.attn.v_proj.weight` | `layer.c_attn_v_proj_w` |
| `transformer.h.{i}.attn.out_proj.weight` | `layer.c_attn_proj_w` |
| `transformer.h.{i}.mlp.fc_in.weight` | `layer.c_mlp_fc_w` |
| `transformer.h.{i}.mlp.fc_in.bias` | `layer.c_mlp_fc_b` |
| `transformer.h.{i}.mlp.fc_out.weight` | `layer.c_mlp_proj_w` |
| `transformer.h.{i}.mlp.fc_out.bias` | `layer.c_mlp_proj_b` |

---

## 四、前向推理核心（`gptj_eval`）

这是整个项目最核心的函数，通过 ggml 的**计算图（computation graph）**机制实现 GPT-J 的完整前向传播。

### 4.1 内存管理策略

```cpp
static size_t buf_size = 256u*1024*1024;  // 初始 256MB
static void * buf = malloc(buf_size);       // 静态分配，跨调用复用
```

推理缓冲区使用 `static` 变量 + `realloc` 策略：
- 首次调用分配 256MB
- 后续调用根据 `mem_per_token * N`（每个 token 的推理内存 × 当前 token 数）判断是否需要扩容
- 扩容时额外增加 10% 以容纳 ggml 对象开销
- 每次 `gptj_eval` 结束时 `ggml_free(ctx0)` 释放计算图，但底层缓冲区 `buf` 被复用

这意味着：推理阶段的临时张量内存是**可复用的**，不会反复 malloc/free。

### 4.2 Transformer 前向计算图构建

对每一层（共 28 层），依次执行：

#### 步骤 1：LayerNorm

```
cur = LayerNorm(inpL)
cur = ln_1_g * cur + ln_1_b    // gamma * normalized + beta
```

通过 `ggml_norm` → `ggml_mul` → `ggml_add` 构建。

#### 步骤 2：Self-Attention

```
Qcur = RoPE(reshape_3d(Q_proj_w @ cur, [n_embd/n_head, n_head, N]), positions)
Kcur = RoPE(reshape_3d(K_proj_w @ cur, [n_embd/n_head, n_head, N]), positions)
Vcur = transpose(V_proj_w @ cur)
```

关键操作：
- **矩阵乘法**：`ggml_mul_mat` 实现 Q/K/V 的线性投影
- **RoPE 旋转位置编码**：`ggml_rope_inplace` 对 Q 和 K 应用旋转位置编码，维度 [n_embd/n_head, n_head, N]，旋转维度 n_rot=64
- **KV cache 存储**：通过 `ggml_view_1d` / `ggml_view_2d` 在预分配的 `memory_k` / `memory_v` 中定位当前层和当前位置的偏移，用 `ggml_cpy` 将新的 K/V 写入

KV cache 的偏移计算：
```cpp
// Key 偏移 = (层数 * n_ctx + n_past) * n_embd * element_size
// Value 偏移 = 层数 * n_ctx * n_embd * element_size + n_past * element_size
```

#### 步骤 3：Attention Score 计算

```
Q = permute(Qcur, [0, 2, 1, 3])    // [head_dim, N, n_head]
K = permute(Kmem, [0, 2, 1, 3])    // [head_dim, n_past+N, n_head]
KQ = K @ Q                          // [n_past+N, N, n_head]
KQ_scaled = KQ / sqrt(head_dim)
KQ_masked = mask_past(KQ_scaled)    // 因果掩码：遮蔽未来位置
KQ_soft_max = softmax(KQ_masked)
```

通过 `ggml_diag_mask_inf_inplace` 实现因果注意力掩码——将当前位置之后的位置设为负无穷，softmax 后变为零。

#### 步骤 4：Attention 输出

```
V = view_3d(memory_v, [n_past+N, head_dim, n_head])
KQV = V @ KQ_soft_max               // 加权求和
KQV_merged = permute(KQV, [0, 2, 1, 3])
cur = contiguous(KQV_merged)         // [n_embd, N]
cur = out_proj_w @ cur               // 输出投影（无偏置）
```

#### 步骤 5：Feed-Forward Network

```
// 注意：FFN 的输入是 inpSA（LayerNorm 后的输出），而非 Attention 的输出
cur = fc_w @ inpSA + fc_b
cur = GELU(cur)
cur = proj_w @ cur + proj_b
```

**架构关键特征**：GPT-J 的 FFN 和 Self-Attention 接收**相同的输入**（LayerNorm 后的结果），这意味着二者在理论上可以**并行计算**。代码中保留了 `inpSA` 作为 FFN 的输入来体现这一点。

#### 步骤 6：残差连接

```
cur = FFN_output + Attention_output     // 两个分支相加
inpL = cur + inpL                       // 与该层输入做残差连接
```

### 4.3 最终输出

所有 28 层计算完毕后：
1. 最终 LayerNorm：`inpL = ln_f_g * LayerNorm(inpL) + ln_f_b`
2. 语言模型头：`logits = lmh_g @ inpL + lmh_b`
3. 只取最后一个 token 的 logits：`(float *)data + n_vocab * (N-1)`

### 4.4 计算图执行

```cpp
ggml_build_forward_expand(gf, inpL);       // 将最终节点加入计算图
ggml_graph_compute_with_ctx(ctx0, gf, n_threads);  // 多线程执行
```

ggml 的计算图是**惰性求值**的：所有操作只是构建计算图的节点，`ggml_graph_compute_with_ctx` 才真正执行计算，并自动利用多线程并行化。

---

## 五、主循环（`main`）

### 5.1 推理流程

```
初始化参数 → 加载模型 → BPE 分词 → 预热推理 → 自回归循环
```

1. **参数解析**：通过 `gpt_params_parse` 解析命令行参数（模型路径、线程数、采样参数等）
2. **模型加载**：`gptj_model_load` 读取 ggml 二进制文件
3. **分词**：`gpt_tokenize` 使用 BPE（Byte Pair Encoding）算法将输入文本转为 token ID 序列
4. **预热**：先做一次 dummy 推理（`{0, 1, 2, 3}`）来测量 `mem_per_token`，用于后续缓冲区大小判断
5. **自回归生成循环**：
   - Prompt 阶段：将输入 token 以 `n_batch` 为单位批量送入推理
   - 生成阶段：每次推理一个 token，通过 top-k/top-p 采样得到下一个 token
   - 遇到 EOS token（ID=50256）或达到 `n_predict` 上限时停止

### 5.2 采样策略

使用 top-k + top-p 采样：
- **Top-k**（默认 k=40）：只从概率最高的 k 个候选中采样
- **Top-p**（默认 p=0.9）：在 top-k 候选中，只保留累积概率达到 p 的最小集合
- **Temperature**（默认 0.9）：控制分布的尖锐程度

### 5.3 批量处理

Prompt 处理阶段，每次最多处理 `n_batch=32` 个 token，以控制内存峰值。生成阶段每次只处理 1 个 token。

---

## 六、模型量化工具（quantize.cpp）

### 6.1 功能

将 f16/f32 格式的模型转换为量化格式，支持的量化类型包括：

| 类型 | 说明 |
|------|------|
| Q4_0 | 4-bit 量化，每块 32 个权重 |
| Q4_1 | 4-bit 量化，带缩放偏移 |
| Q5_0 | 5-bit 量化 |
| Q5_1 | 5-bit 量化，带偏移 |
| Q8_0 | 8-bit 量化 |
| Q2_K ~ Q6_K | K-quant 系列，不同精度级别 |

### 6.2 量化策略

```cpp
const std::vector<std::string> to_quant = { ".*weight" };
```

只有名称匹配 `.*weight` 的**二维张量**会被量化（即所有权重矩阵），一维张量（偏置、LayerNorm 参数等）保持 float32 不变。这是合理的——权重矩阵占模型体积的绝大部分，而偏置向量很小，量化偏置可能引入不必要的精度损失。

### 6.3 量化版本控制

文件头中的 `ftype` 字段编码了量化版本信息：
```cpp
const int32_t qntvr_src = hparams.ftype / GGML_QNT_VERSION_FACTOR;
const int32_t ftype_dst = GGML_QNT_VERSION * GGML_QNT_VERSION_FACTOR + ftype;
```

这确保了量化格式的向前兼容性。

---

## 七、模型转换脚本（convert-h5-to-ggml.py）

### 7.1 功能

将 HuggingFace 格式（PyTorch state_dict，通过 `GPTJForCausalLM` 加载）的 GPT-J 模型转换为 ggml 二进制格式。

### 7.2 转换流程

1. 加载 `vocab.json`（BPE 词表）、`added_tokens.json`（额外 token）、`config.json`（模型配置）
2. 用 `GPTJForCausalLM.from_pretrained` 加载完整模型
3. 写入文件头：magic + 超参数
4. 写入词表：通过 `bytes_to_unicode` 编码映射将 UTF-8 字节映射到 Unicode 字符
5. 遍历 `state_dict` 中所有张量：
   - 跳过 `attn.masked_bias` 和 `attn.bias`（推理不需要）
   - 二维权重矩阵：转换为 float16（默认）或保持 float32
   - 其他张量（偏置、LayerNorm 参数）：保持 float32

### 7.3 `bytes_to_unicode` 编码

这是 GPT-2/GPT-J 分词器使用的经典编码方式：将 UTF-8 字节映射到可打印的 Unicode 字符，避免在 BPE 过程中出现控制字符问题。所有 256 个字节值都有对应的 Unicode 映射。

---

## 八、关键特性总结

### 8.1 零运行时分配

模型加载后，推理阶段的内存使用模式：
- **静态部分**：模型权重 + KV cache 在 `gptj_model_load` 时一次性分配，生命周期持续到程序结束
- **动态部分**：每次 `gptj_eval` 创建的计算图使用静态缓冲区（`static void * buf`），通过 `ggml_init` / `ggml_free` 复用，不触发系统级 malloc/free

### 8.2 计算图驱动的推理

不手写任何矩阵运算代码，而是通过 ggml 的张量操作 API（`ggml_mul_mat`, `ggml_add`, `ggml_rope_inplace` 等）声明式构建计算图，由 ggml 运行时负责多线程并行执行和内存管理。

### 8.3 GPT-J 特有的并行化潜力

GPT-J 架构中 Self-Attention 和 FFN 共享相同的输入（LayerNorm 后的结果），二者在理论上可以并行计算后再相加。代码中用 `inpSA` 变量名明确保留了这一设计意图（`inpSA = cur` 在 LayerNorm 之后、Attention 计算之前保存）。

### 8.4 KV Cache 的视图操作

通过 `ggml_view_1d` / `ggml_view_2d` / `ggml_view_3d` 直接在预分配的大张量上创建"视图"来读写 KV cache，避免了拷贝开销。Key 以行优先方式存储，Value 以转置方式存储（`Vcur = transpose(V_proj_w @ cur)`），这是为了后续的 `ggml_mul_mat` 操作能够高效执行。

### 8.5 多精度权重管理

- 大型权重矩阵（wte, 投影矩阵, FFN 权重）：使用 `wtype`（float16 或量化格式）
- 偏置和 LayerNorm 参数：始终 float32
- KV cache：固定 float16

这种混合精度策略在节省内存的同时保持了关键计算的数值精度。
