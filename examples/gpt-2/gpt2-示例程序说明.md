# GPT-2 示例程序说明

## 一、概述

`examples/gpt-2/` 目录包含 6 个源文件，它们以 **渐进式** 的方式演示如何使用 GGML 张量库实现 GPT-2 模型的推理。这些示例程序从最基础的 `ggml_context` 直接使用，逐步引入图分配器、后端抽象、多后端调度器、批量解码等高级 API，完整展示了 GGML 库核心接口的设计意图和使用方法。

| 源文件 | 行数 | 核心主题 | 难度 |
|--------|------|----------|------|
| `main-ctx.cpp` | 841 | 旧版 ggml_context API 的直接使用 | 入门 |
| `main-alloc.cpp` | 881 | ggml-alloc 图分配器，分离构建与求值 | 基础 |
| `main-backend.cpp` | 947 | ggml_backend 后端抽象，支持 GPU 加速 | 中级 |
| `main-sched.cpp` | 1080 | ggml_backend_sched 多后端调度器 | 高级 |
| `main-batched.cpp` | 1211 | 批量解码与多序列并行生成 | 高级 |
| `quantize.cpp` | 185 | 模型权重量化工具 | 独立工具 |

---

## 二、GPT-2 模型架构（所有示例共用）

所有推理示例实现了相同的 GPT-2 transformer 架构（以 117M 参数模型为默认配置）：

### 超参数 (`gpt2_hparams`)
```
n_vocab = 50257    // 词表大小（Byte Pair Encoding）
n_ctx   = 1024     // 最大上下文长度
n_embd  = 768      // 嵌入维度
n_head  = 12       // 注意力头数
n_layer = 12       // transformer 层数
ftype   = 1        // 权重数据类型（0=FP32, 1=FP16, 2+=量化）
```

### 模型权重结构
```
gpt2_model
├── wte          [n_embd, n_vocab]   // token 嵌入矩阵
├── wpe          [n_embd, n_ctx]     // 位置嵌入矩阵
├── ln_f_g/b     [n_embd]            // 最终层归一化参数
├── lm_head      [n_embd, n_vocab]   // 语言模型输出头（部分模型与 wte 共享）
└── layers[0..11] (gpt2_layer)
    ├── ln_1_g/b        [n_embd]            // 注意力前层归一化
    ├── c_attn_attn_w/b [n_embd, 3*n_embd]  // Q/K/V 联合投影
    ├── c_attn_proj_w/b [n_embd, n_embd]     // 注意力输出投影
    ├── ln_2_g/b        [n_embd]            // MLP 前层归一化
    ├── c_mlp_fc_w/b    [n_embd, 4*n_embd]  // MLP 全连接层
    └── c_mlp_proj_w/b  [4*n_embd, n_embd]  // MLP 投影层
```

### 单层前向计算流程
```
输入 inpL [n_embd, N]
  │
  ├─ LayerNorm(ln_1) → c_attn(联合QKV投影) → Self-Attention → c_proj(输出投影)
  │    └─ 残差连接: cur = attention_output + inpL
  │
  ├─ LayerNorm(ln_2) → MLP(fc → GELU → proj)
  │    └─ 残差连接: output = mlp_output + cur
  │
  └─ 输出到下一层
```

### 自注意力机制细节
1. **联合投影**: `c_attn_attn_w` 形状为 `[n_embd, 3*n_embd]`，一次矩阵乘法同时计算 Q、K、V
2. **视图切分**: 通过 `ggml_view_2d` 将 `[3*n_embd, N]` 的输出切分为三个 `[n_embd, N]` 的 Q/K/V
3. **KV 缓存**: K、V 被存入 `kv_cache.k` / `kv_cache.v`，避免重复计算
4. **缩放点积注意力**:
   - Q reshape 为 `[n_embd/n_head, n_head, N]` 并 permute
   - K reshape 为 `[n_embd/n_head, n_head, n_kv]` 并 permute
   - `KQ = mul_mat(K, Q)` → scale → 加 mask → softmax
   - V 同样 reshape/permute 后与 KQ_soft_max 做矩阵乘法
5. **因果遮蔽**: 通过 KQ_mask 将"未来"位置设为 `-INFINITY`，确保自回归特性

### 全局前向流程
```
token_ids → wte(token_ids) + wpe(positions)
         → 12 层 transformer block
         → LayerNorm(ln_f)
         → lm_head (矩阵乘法)
         → logits [n_vocab, N]
         → 采样 (top-k, top-p, temperature)
```

---

## 三、各源文件详细分析

### 3.1 main-ctx.cpp — 旧版 ggml_context API

**定位**: 最基础的示例，展示 GGML 最初级的使用方式。

#### 核心特征
- **单一 `ggml_context`**: 所有权重张量、KV 缓存、计算图节点共用一个 context
- **`no_alloc = false`**: `ggml_init()` 时直接分配内存，张量的 `data` 指针立即可用
- **手动内存管理**: 使用 `malloc` 预分配内存 buffer，通过 `realloc` 动态调整大小
- **`_inplace` 操作**: 使用 `ggml_add_inplace()`、`ggml_mul_inplace()` 等原地操作（已弃用的模式）
- **直接求值**: `ggml_graph_compute_with_ctx(ctx, gf, n_threads)`

#### 关键代码模式
```cpp
// 1. 预计算内存大小，一次性分配
size_t ctx_size = 0;
ctx_size += ggml_row_size(wtype, n_vocab * n_embd);  // wte
ctx_size += ...;  // 所有权重 + KV 缓存

// 2. 初始化 context（no_alloc=false，直接分配内存）
struct ggml_init_params params = {
    /*.mem_size   =*/ ctx_size,
    /*.mem_buffer =*/ NULL,
    /*.no_alloc   =*/ false,
};
struct ggml_context * ctx = ggml_init(params);

// 3. 创建张量（data 指针已可用）
model.wte = ggml_new_tensor_2d(ctx, wtype, n_embd, n_vocab);

// 4. 直接读取权重数据到张量
fread(model.wte->data, sizeof(float), ggml_nbytes(model.wte), f);

// 5. 构建计算图并直接求值
ggml_graph_compute_with_ctx(ctx, gf, n_threads);
```

#### 局限性
- 不支持 GPU 加速
- 内存利用率低（一次性预分配所有空间）
- 原地操作在某些后端上不安全
- KV 缓存和权重混在同一内存空间

---

### 3.2 main-alloc.cpp — 图分配器（Graph Allocator）

**定位**: 引入 `ggml-alloc.h` API，分离图的构建与求值阶段。

#### 核心特征
- **`no_alloc = true`**: 权重 context 仅创建张量元数据，不分配数据空间
- **`ggml_gallocr`**: 图分配器，自动管理计算图的中间张量内存
- **`ggml_set_input()` / `ggml_set_output()`**: 标记输入/输出张量，供分配器识别
- **分离构建与求值**: `gpt2_graph()` 构建图 → `gpt2_eval()` 执行求值
- **`ggml_gallocr_reserve()`**: 预分配最坏情况下的计算缓冲区

#### 关键代码模式
```cpp
// 1. 权重 context 使用 no_alloc=true
struct ggml_init_params params = {
    /*.mem_size   =*/ ggml_tensor_overhead() * n_tensors,
    /*.mem_buffer =*/ NULL,
    /*.no_alloc   =*/ true,  // 仅创建张量描述，不分配数据空间
};

// 2. 标记输入/输出张量
struct ggml_tensor * inp_tokens = ggml_new_tensor_1d(ctx, GGML_TYPE_I32, N);
ggml_set_input(inp_tokens);   // 标记为输入
// ... 构建图 ...
ggml_set_output(result);      // 标记为输出

// 3. 使用 gallocr 预分配计算缓冲区
ggml_gallocr_t allocr = ggml_gallocr_new(GGML_BACKEND_CPU);
struct ggml_cgraph * gf = gpt2_graph(model, N);  // 构建最坏情况图
ggml_gallocr_reserve(allocr, gf);                // 预分配内存

// 4. 每次求值时重用预分配内存
struct ggml_cgraph * gf = gpt2_graph(model, n_tokens);
ggml_gallocr_alloc_graph(allocr, gf);            // 分配图节点
// 设置输入数据 ...
ggml_graph_compute_with_ctx(ctx, gf, n_threads); // 求值
```

#### 相比 main-ctx.cpp 的改进
- **内存效率**: 计算缓冲区按需分配，多个 token 复用同一块内存
- **安全性**: 不使用 `_inplace` 操作，避免数据竞争
- **可扩展性**: 图的构建与执行分离，便于后续引入后端抽象

---

### 3.3 main-backend.cpp — 后端抽象（Backend Abstraction）

**定位**: 引入 `ggml_backend_t` 抽象层，支持 CPU/CUDA/Metal 等多种硬件后端。

#### 核心特征
- **`ggml_backend_t`**: 统一的后端接口，屏蔽 CPU/GPU 差异
- **`ggml_backend_alloc_ctx_tensors()`**: 将 context 中的张量分配到后端缓冲区
- **`ggml_backend_tensor_set()`**: 跨设备数据传输（自动处理 host↔device）
- **`ggml_backend_graph_compute()`**: 通过后端执行计算图
- **`n_gpu_layers`**: 控制将多少层放到 GPU 上（本示例实际只使用单后端）

#### 关键代码模式
```cpp
// 1. 初始化后端
#ifdef GGML_USE_CUDA
if (n_gpu_layers > 0) {
    model.backend = ggml_backend_cuda_init(0);  // GPU 后端
}
#endif
if (!model.backend) {
    model.backend = ggml_backend_cpu_init();     // CPU 后端
}

// 2. 在后端缓冲区中分配权重
model.buffer_w = ggml_backend_alloc_ctx_tensors(model.ctx_w, model.backend);

// 3. 加载权重时区分 CPU/GPU
if (ggml_backend_is_cpu(model.backend)) {
    fin.read((char *)tensor->data, ggml_nbytes(tensor));       // 直接读取
} else {
    fin.read(read_buf.data(), ggml_nbytes(tensor));
    ggml_backend_tensor_set(tensor, read_buf.data(), 0, ...);  // 复制到设备
}

// 4. 通过后端执行计算
ggml_backend_graph_compute(model.backend, gf);

// 5. 从后端获取输出数据
ggml_backend_tensor_get(result, logits.data(), 0, nbytes);
```

#### 相比 main-alloc.cpp 的改进
- **硬件加速**: 支持 CUDA (NVIDIA GPU)、Metal (Apple GPU)、CPU 后端
- **跨设备透明**: `ggml_backend_tensor_set/get` 自动处理 host↔device 数据传输
- **后端统一接口**: 同一套代码无需修改即可在不同硬件上运行

---

### 3.4 main-sched.cpp — 多后端调度器（Backend Scheduler）

**定位**: 最复杂的推理示例，使用 `ggml_backend_sched_t` 实现多后端协同计算。

#### 核心特征
- **多后端链**: 同时初始化 GPU + BLAS + CPU 三个后端
- **层级别分割**: 根据 `n_gpu_layers` 参数决定哪些层放到 GPU，其余留在 CPU
- **`ggml_tallocr`**: 每个后端独立的内存分配器
- **`ggml_backend_sched_new()`**: 创建调度器，自动分析图依赖并分割到各后端
- **`GGML_BACKEND_SCHED_MEASURE`**: 预测量模式，用于确定各后端的内存需求

#### 关键代码模式
```cpp
// 1. 初始化多个后端
ggml_backend_t backend_gpu = ggml_backend_cuda_init(0);
ggml_backend_t backend_cpu = ggml_backend_cpu_init();
ggml_backend_t backend_blas = ggml_backend_blas_init();

// 2. 为每个后端创建分配器
ggml_tallocr_t tallocr_gpu = ggml_tallocr_new(backend_gpu);
ggml_tallocr_t tallocr_cpu = ggml_tallocr_new(backend_cpu);

// 3. 张量到后端的手动映射（关键！）
// 将权重分配到对应后端的缓冲区
for (int i = 0; i < n_layer; i++) {
    if (i < n_gpu_layers) {
        // GPU 层：将权重分配到 GPU 缓冲区
        ggml_tallocr_alloc(tallocr_gpu, layer.c_attn_attn_w);
    } else {
        // CPU 层：将权重分配到 CPU 缓冲区
        ggml_tallocr_alloc(tallocr_cpu, layer.c_attn_attn_w);
    }
}

// 4. KV 缓存放置策略
// 如果 GPU 层数 >= 总层数的一半，KV 缓存放 GPU；否则放 CPU
if (n_gpu_layers >= n_layer / 2) {
    ggml_tallocr_alloc(tallocr_gpu, model.kv_cache.k);
} else {
    ggml_tallocr_alloc(tallocr_cpu, model.kv_cache.k);
}

// 5. 创建调度器
ggml_backend_sched_t sched = ggml_backend_sched_new(
    backends.data(), n_backends,
    GGML_BACKEND_SCHED_MEASURE, ...);

// 6. 调度器自动分割图并执行
ggml_backend_sched_graph_compute(sched, gf);
// 调度器内部：
//   - 分析计算图的张量依赖关系
//   - 根据张量所在后端自动将图分割为多个子图
//   - 在各后端上依次执行子图
//   - 必要时自动插入跨设备数据拷贝
```

#### 相比 main-backend.cpp 的改进
- **混合计算**: GPU 处理前 N 层，CPU 处理剩余层，BLAS 处理大规模矩阵乘法
- **灵活部署**: 根据可用显存动态调整 GPU 层数
- **自动图分割**: 调度器自动处理跨设备的数据传输和子图调度

---

### 3.5 main-batched.cpp — 批量解码与多序列并行生成

**定位**: 演示如何同时处理多个 token 序列，实现高效的批量推理。

#### 核心特征
- **`gpt2_batch` 结构体**: 封装批量输入数据（token、位置、序列 ID、logits 标志）
- **多序列 KV 缓存**: 每个缓存单元 (`gpt2_kv_cell`) 记录位置和所属序列 ID
- **KV 缓存共享**: `gpt2_kv_cache_seq_cp()` 将 prompt 的 KV 缓存复制给所有并行序列
- **动态 KQ_mask**: 根据序列归属和位置关系动态构建注意力遮蔽
- **`gpt2_graph()` 的 `measure` 模式**: 构建最坏情况图用于内存预分配

#### 关键代码模式
```cpp
// 1. 批量输入结构
struct gpt2_batch {
    int32_t     n_tokens;  // 本批次 token 总数
    gpt_vocab::id * token;  // token ID 数组
    gpt2_pos    * pos;     // 位置数组
    gpt2_seq_id * seq_id;  // 序列 ID 数组
    int8_t      * logits;  // 是否为该 token 输出 logits
};

// 2. Prompt 阶段：所有 prompt token 一次性送入
batch.n_tokens = embd_inp.size();
for (int i = 0; i < batch.n_tokens; i++) {
    batch.token[i]  = embd_inp[i];
    batch.pos[i]    = i;
    batch.seq_id[i] = 0;        // 所有 prompt token 属于序列 0
    batch.logits[i] = false;    // 不需要中间 token 的 logits
}
batch.logits[batch.n_tokens - 1] = true;  // 仅最后一个 token 需要 logits

// 3. 将 prompt KV 缓存共享给所有并行序列
for (int i = 1; i < n_parallel; ++i) {
    gpt2_kv_cache_seq_cp(model.kv_cache, 0, i, 0, batch.n_tokens);
}

// 4. 生成阶段：每步将所有序列的新 token 组成一个批次
while (n_cur < n_len) {
    batch.n_tokens = 0;
    for (int i = 0; i < n_parallel; ++i) {
        if (i_batch[i] < 0) continue;  // 已完成的序列跳过

        // 采样下一个 token
        id = gpt_sample_top_k_top_p(vocab, logits_i, top_k, top_p, temp, rng);

        // 添加到当前批次
        batch.token [batch.n_tokens] = id;
        batch.pos   [batch.n_tokens] = n_cur;
        batch.seq_id[batch.n_tokens] = i;      // 不同序列的 token
        batch.logits[batch.n_tokens] = true;
        batch.n_tokens++;
    }
    gpt2_decode(model, allocr, batch, n_threads, logits);
}

// 5. 动态 KQ_mask 构建确保序列隔离
for (int j = 0; j < n_tokens; ++j) {
    for (int i = 0; i < n_kv; ++i) {
        if (!kv_cache.cells[i].has_seq_id(seq_id) || kv_cache.cells[i].pos > pos) {
            data_buf[j * n_kv + i] = -INFINITY;  // 不可见位置遮蔽
        }
    }
}
```

#### 相比其他示例的独特价值
- **推理效率**: 将多个序列的 token 合并到一个批次，一次前向计算同时服务多条序列
- **KV 缓存共享**: 多个序列共享 prompt 部分的 KV 缓存，避免重复计算
- **实际应用场景**: 类似于大语言模型服务中的 continuous batching 技术

---

### 3.6 quantize.cpp — 模型权重量化工具

**定位**: 独立的命令行工具，将 GGML 格式模型从高精度转换为量化格式。

#### 核心特征
- **流式处理**: 逐张量读取、量化、写入，内存占用低
- **选择性量化**: 通过正则表达式匹配需要量化的张量名
- **格式兼容**: 保持 GGML 二进制格式不变，仅修改权重数据类型

#### 量化的目标张量
```
model/wte                    // token 嵌入
model/lm_head                // 输出头
model/h.*/attn/c_attn/w      // 注意力联合投影权重
model/h.*/attn/c_proj/w      // 注意力输出投影权重
model/h.*/mlp/c_fc/w         // MLP 全连接权重
model/h.*/mlp/c_proj/w       // MLP 投影权重
```

#### GGML 模型文件格式
```
[magic: uint32]                    // 文件魔数
[n_vocab, n_ctx, n_embd, n_head, n_layer, ftype]  // 超参数
[n_vocab 个 (len, word) 对]        // 词表
[逐张量: n_dims, name_len, type, ne[], name, data[]]  // 权重数据
```

---

## 四、GGML API 演进路线（关键学习路径）

这 6 个示例程序展示了一条清晰的 API 演进路径：

### 阶段一：直接使用（main-ctx.cpp）
```
ggml_init(ctx_size, no_alloc=false)
  → 创建张量（data 立即可用）
  → 构建计算图
  → ggml_graph_compute_with_ctx(ctx, graph, n_threads)
```

**要点**: 理解 GGML 的核心抽象 — 张量（tensor）和计算图（cgraph）。

### 阶段二：图分配器（main-alloc.cpp）
```
ggml_init(meta_size, no_alloc=true)
  → 创建张量描述（无数据空间）
  → ggml_set_input() / ggml_set_output()
  → ggml_gallocr_new() + ggml_gallocr_reserve()
  → ggml_gallocr_alloc_graph() + ggml_graph_compute_with_ctx()
```

**要点**: 分离张量描述与数据存储；计算缓冲区自动管理和复用。

### 阶段三：后端抽象（main-backend.cpp）
```
ggml_backend_cpu_init() / ggml_backend_cuda_init()
  → ggml_backend_alloc_ctx_tensors()
  → ggml_backend_tensor_set() (host→device)
  → ggml_backend_graph_compute()
  → ggml_backend_tensor_get() (device→host)
```

**要点**: 后端接口统一了 CPU/GPU 的使用方式；跨设备数据传输透明化。

### 阶段四：多后端调度（main-sched.cpp）
```
初始化多个后端 (GPU + BLAS + CPU)
  → 手动将张量映射到各后端缓冲区
  → ggml_backend_sched_new()
  → ggml_backend_sched_graph_compute() (自动图分割)
```

**要点**: 多设备协同计算；层级别模型分割；自动跨设备数据传输。

### 阶段五：批量处理（main-batched.cpp）
```
基于后端抽象 + 图分配器
  → gpt2_batch 封装多序列输入
  → KV 缓存支持多序列（seq_id）
  → 动态 KQ_mask 确保序列隔离
  → 一次前向计算服务多个序列
```

**要点**: 高效推理的核心技术 — 批量处理和 KV 缓存共享。

---

## 五、关键技术细节总结

### 5.1 计算图构建模式

所有示例都遵循相同的图构建模式：

| 步骤 | 说明 |
|------|------|
| `ggml_init()` | 创建临时 context 用于图构建 |
| `ggml_new_tensor_*()` | 创建输入张量（标记 `ggml_set_input`） |
| `ggml_mul_mat / ggml_add / ggml_norm ...` | 构建计算节点 |
| `ggml_build_forward_expand(gf, output)` | 将输出节点加入图 |
| `ggml_free(ctx)` | 释放临时 context（张量描述由图持有） |

### 5.2 KV 缓存机制

KV 缓存是自回归推理的关键优化：

| 概念 | 说明 |
|------|------|
| 结构 | 两个 1D 张量 `k[n_layer * n_ctx * n_embd]` 和 `v[...]` |
| 写入 | 每层将新 token 的 K/V 通过 `ggml_cpy` 写入缓存对应位置 |
| 读取 | 通过 `ggml_view_1d` 以偏移量读取该层所有已缓存的 K/V |
| 批量扩展 | `gpt2_kv_cell` 追踪每个位置属于哪些序列，实现多序列共享 |

### 5.3 注意力遮蔽（KQ_mask）

遮蔽确保自回归特性——每个 token 只能看到自身及之前的 token：

```cpp
// main-ctx/alloc/backend 的简单遮蔽（单序列）
for (int i = 0; i < n_kv; i++) {
    data[j * n_kv + i] = (i > pos_j) ? -INFINITY : 0;
}

// main-batched 的多序列遮蔽
for (int i = 0; i < n_kv; i++) {
    if (!cells[i].has_seq_id(seq_id) || cells[i].pos > pos) {
        data[j * n_kv + i] = -INFINITY;
    }
}
```

### 5.4 Token 采样策略

所有推理示例使用相同的采样函数 `gpt_sample_top_k_top_p()`：

1. **Temperature**: `logits /= temperature`（温度越高越随机）
2. **Top-K**: 只保留概率最高的 K 个候选
3. **Top-P**: 从 Top-K 结果中按累积概率 P 截断
4. **随机采样**: 在过滤后的候选中按概率分布随机选取

### 5.5 权重共享

GPT-2 模型中 `wte`（token 嵌入）和 `lm_head`（语言模型头）共享权重：

```cpp
if (name == "model/wte" && !has_lm_head) {
    model.lm_head = tensor;  // 直接复用 wte 张量
}
```

这是一种参数高效化设计，将输入嵌入矩阵同时用作输出投影矩阵。

### 5.6 模型文件加载流程

所有推理示例的模型加载遵循相同流程：

```
打开文件 → 验证 magic → 读取 hparams → 读取 vocab
→ 预计算 buffer 大小 → 初始化 ggml_context
→ 创建张量描述 → 分配后端缓冲区
→ 逐张量读取数据（CPU 直接读，GPU 通过 staging buffer）
→ 验证维度和数据类型
```

---

## 六、各示例的依赖关系

```
quantize.cpp (独立工具，不依赖推理逻辑)

main-ctx.cpp        ← 最基础，无额外依赖
     ↓ 引入图分配器
main-alloc.cpp      ← 新增: ggml-alloc.h, ggml_set_input/output
     ↓ 引入后端抽象
main-backend.cpp    ← 新增: ggml-backend.h, ggml_backend_t
     ↓ 引入多后端调度
main-sched.cpp      ← 新增: ggml-backend.h, 多后端, ggml_tallocr
     ↓ 引入批量处理
main-batched.cpp    ← 新增: gpt2_batch, 多序列 KV 缓存
```

---

## 七、辅助文件

| 文件 | 说明 |
|------|------|
| `CMakeLists.txt` | 构建配置，定义所有可执行文件的编译规则 |
| `common.h` / `common-ggml.h` | 共享的工具函数（tokenizer、采样、参数解析） |
| `convert-h5-to-ggml.py` | 将 H5 格式的 GPT-2 权重转换为 GGML 格式 |
| `convert-ckpt-to-ggml.py` | 将 TensorFlow checkpoint 转换为 GGML 格式 |
| `convert-cerebras-to-ggml.py` | 将 Cerebras 模型转换为 GGML 格式 |
| `download-model.sh` | 下载原始 GPT-2 模型权重的脚本 |
| `download-ggml-model.sh` | 下载已转换为 GGML 格式的模型 |
| `README.md` | 使用说明文档 |
