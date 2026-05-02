# GGML 深度学习路线说明

## 项目定位

GGML 是一个面向机器学习推理和训练的**底层张量计算库**，核心设计目标：

- **零运行时内存分配**：所有内存在初始化时预分配
- **广泛量化支持**：从 1-bit 到 8-bit 的数十种量化格式
- **多硬件后端**：CPU/CUDA/Metal/Vulkan/OpenCL/SYCL 等 10+ 后端
- **无第三方依赖**：纯 C/C++ 实现
- **计算图模式**：声明式构建、统一执行

---

## 架构全景图

```
┌─────────────────────────────────────────────────────────────┐
│                      应用层 (examples/)                       │
│  GPT-2/J  SAM  YOLO  MNIST  Magika  Python绑定  性能测试     │
├─────────────────────────────────────────────────────────────┤
│                   高级 API 层 (include/)                      │
│  ggml-opt.h (训练)  gguf.h (模型格式)  ggml-cpp.h (C++封装)  │
├─────────────────────────────────────────────────────────────┤
│                  后端抽象层 (include/ggml-backend.h)           │
│  backend_t  buffer_t  sched_t  dev_t  event_t                │
├─────────────────────────────────────────────────────────────┤
│                    核心库 (src/ggml.c + ggml.h)               │
│  张量创建 │ 操作定义 │ 计算图构建 │ 图执行 │ 量化              │
├─────────────────────────────────────────────────────────────┤
│              内存管理 (src/ggml-alloc.c / ggml-alloc.h)       │
│  tallocr (张量级)  gallocr (图级自动分配)                     │
├──────────┬──────────┬──────────┬──────────┬─────────────────┤
│ CPU后端  │ CUDA后端 │ Metal后端│ Vulkan   │ 其他后端...      │
│ ggml-cpu │ ggml-cuda│ggml-metal│ggml-vulkan│ RPC/OpenCL/... │
│ AMX/SIMD │ TensorCore│ GPUCompute│ SPIR-V  │                │
└──────────┴──────────┴──────────┴──────────┴─────────────────┘
```

---

## 第一阶段：核心概念与张量基础

**目标**：理解 GGML 的核心数据模型和基本操作

### 1.1 张量（Tensor）—— 一切的基础

**关键文件**：`include/ggml.h` 中的 `struct ggml_tensor`

```
struct ggml_tensor {
    enum ggml_type type;              // 数据类型（F32/F16/各种量化）
    struct ggml_backend_buffer * buffer; // 后端缓冲区
    int64_t ne[4];                    // 4维形状（ne[0]最内层）
    size_t nb[4];                     // 4维字节步幅
    enum ggml_op op;                  // 产生该张量的操作
    int32_t op_params[16];            // 操作参数
    int32_t flags;                    // 标志位（input/output/param/loss）
    struct ggml_tensor * src[10];     // 源张量（操作数）
    void * data;                      // 数据指针
    char name[64];                    // 名称
};
```

**必须掌握的知识点**：

| 概念 | 说明 | 关键代码 |
|------|------|---------|
| 张量维度 | 最多 4D，`ne[i]` 是第 i 维大小，`nb[i]` 是第 i 维字节步幅 | `ne[0]*nb[0]` = 一个元素的字节数 |
| 数据布局 | 行主序，`nb[0] <= nb[1] <= nb[2] <= nb[3]` | 连续性判断：`nb[i] == nb[i-1]*ne[i-1]` |
| 张量类型 | 42 种数据类型，从 F32 到 1-bit 量化 | `enum ggml_type` |
| View 张量 | 零拷贝视图，共享底层内存 | `view_src` + `view_offs` |
| 标志位 | INPUT/OUTPUT/PARAM/LOSS 控制训练和推理行为 | `ggml_set_input()` 等 |

**实践练习**：

阅读 `examples/simple/simple-ctx.cpp`，理解：
- 如何创建 context
- 如何创建张量并填充数据
- 张量的 `ne`/`nb` 如何对应矩阵的行列

### 1.2 上下文（Context）—— 内存管理核心

**关键文件**：`include/ggml.h`

```c
struct ggml_init_params {
    size_t mem_size;      // 内存池大小
    void * mem_buffer;    // 可选的外部缓冲区
    bool no_alloc;        // 是否跳过张量数据分配
};
```

**核心机制**：
- Context 是一个**内存池**，所有张量的元数据和数据都分配在其中
- `no_alloc=false`：创建张量时立即分配数据内存（传统方式）
- `no_alloc=true`：仅分配元数据，数据由 `ggml_gallocr` 或 `ggml_backend_alloc_ctx_tensors` 后续分配（现代方式）
- `ggml_free()` 释放整个 context 的所有内存（无单独释放）

**两种内存管理模式对比**：

| 特性 | 传统模式 (simple-ctx) | 后端模式 (simple-backend) |
|------|----------------------|-------------------------|
| no_alloc | false | true |
| 数据分配时机 | ggml_init 时 | backend_alloc_ctx_tensors 时 |
| 数据位置 | context 内存池中 | 后端缓冲区（可能是 GPU） |
| 访问方式 | 直接指针 `tensor->data` | `ggml_backend_tensor_set/get` |

### 1.3 数据类型体系

**关键文件**：`include/ggml.h` 的 `enum ggml_type`、`src/ggml-common.h`

GGML 支持 **42 种数据类型**，按层次理解：

```
数据类型体系
├── 浮点类型
│   ├── GGML_TYPE_F32    (32-bit，标准精度)
│   ├── GGML_TYPE_F16    (16-bit，半精度)
│   ├── GGML_TYPE_BF16   (16-bit，Brain Float)
│   └── GGML_TYPE_F64    (64-bit，双精度)
│
├── 整数类型
│   ├── GGML_TYPE_I8/I16/I32/I64
│
├── 基础量化（block-based）
│   ├── Q4_0/Q4_1    (4-bit，block_size=32)
│   ├── Q5_0/Q5_1    (5-bit，block_size=32)
│   ├── Q8_0/Q8_1    (8-bit，block_size=32)
│   └── Q1_0         (1-bit，block_size=32)
│
├── K-量化（super-block）
│   ├── Q2_K ~ Q6_K  (2~6-bit，super-block 优化)
│   └── Q8_K         (8-bit，K系列中间格式)
│
├── I-量化（重要性矩阵）
│   ├── IQ1_S/IQ1_M  (1-bit)
│   ├── IQ2_XXS/XS/S (2-bit 变体)
│   ├── IQ3_XXS/S    (3-bit 变体)
│   └── IQ4_NL/XS    (4-bit 变体)
│
└── 新型量化
    ├── TQ1_0/TQ2_0  (三值量化)
    ├── MXFP4        (MX 格式 4-bit)
    └── NVFP4        (NVIDIA FP4)
```

**量化核心概念**（以 `block_q4_0` 为例）：

```c
// 每 32 个原始 F32 值量化为 1 个 block_q4_0
struct block_q4_0 {
    ggml_half d;      // 缩放因子 (delta)
    uint8_t qs[16];   // 16字节 = 32个4-bit值
};
// 原始: 32 × 4 bytes = 128 bytes
// 量化后: 2 + 16 = 18 bytes
// 压缩比: 128/18 ≈ 7.1x
```

**实践练习**：
- 阅读 `examples/python/example_test_all_quants.py`，理解不同量化类型的精度和压缩比
- 阅读 `tests/test-quantize-fns.cpp`，了解量化的测试方式

---

## 第二阶段：计算图与操作体系

**目标**：掌握 GGML 的计算图模型和丰富的操作集

### 2.1 计算图（Compute Graph）

**关键文件**：`include/ggml.h` 中 graph 相关函数

GGML 采用**延迟执行**模式：
1. 创建张量，定义操作 → 构建计算图
2. 一次性分配所有中间张量内存
3. 执行整个计算图

```
构建阶段:  a = new_tensor()  b = new_tensor()  c = mul_mat(a, b)
           ↓                  ↓                  ↓
           [节点]  ──src──→  [节点]  ──src──→  [节点] = c
                                            ↓
执行阶段:  ggml_build_forward_expand(graph, c) → 完整拓扑排序
           ggml_graph_compute(graph, plan)      → 逐节点执行
```

**关键函数链**：

```c
// 1. 创建图
struct ggml_cgraph * gf = ggml_new_graph(ctx);

// 2. 定义操作（只是声明，不执行计算）
struct ggml_tensor * result = ggml_mul_mat(ctx, weight, input);

// 3. 将结果展开为完整计算图
ggml_build_forward_expand(gf, result);

// 4. 执行计算
ggml_graph_compute_with_ctx(ctx, gf, n_threads);
```

**重要概念**：
- `ggml_build_forward_expand`：递归遍历所有源张量，构建完整的 DAG
- `ggml_cplan`：计算计划，包含线程数和工作缓冲区
- 节点顺序：拓扑排序保证依赖关系正确

### 2.2 操作分类体系

**关键文件**：`include/ggml.h` 的 `enum ggml_op`（186 种操作）

按功能分类理解：

#### A. 基础数学（约 20 种）
```
ADD, SUB, MUL, DIV       四则运算
SQR, SQRT, LOG, SIN, COS 数学函数
SUM, SUM_ROWS, MEAN       规约操作
SCALE, CLAMP              标量操作
```

#### B. 矩阵运算（3 种，但最核心）
```
MUL_MAT       矩阵乘法（GGML 中使用频率最高的操作）
MUL_MAT_ID    带索引的矩阵乘法（MoE 模型）
OUT_PROD      外积
```

**`ggml_mul_mat` 是整个库的灵魂**：
- `ggml_mul_mat(ctx, a, b)` 计算的是 `a @ b^T`（注意转置语义）
- 支持 F16/F32 输入、量化权重
- 后端会针对此操作做极致优化（Tensor Core、AMX、SIMD）

#### C. 形状操作（约 15 种）
```
RESHAPE, VIEW, PERMUTE, TRANSPOSE, CONT   形状变换
CONCAT, REPEAT, PAD, ROLL                 拼接/复制/填充
GET_ROWS, SET_ROWS                        行选择/设置
```

#### D. 归一化（5 种）
```
NORM         Layer Normalization
RMS_NORM     Root Mean Square Normalization（LLM 最常用）
GROUP_NORM   Group Normalization（CNN 用）
L2_NORM      L2 Normalization
```

#### E. 注意力机制（约 10 种）
```
SOFT_MAX              Softmax
ROPE                  Rotary Position Embedding（LLM 位置编码）
FLASH_ATTN_EXT        Flash Attention（优化实现）
DIAG_MASK_INF         因果注意力掩码
WIN_PART/WIN_UNPART   窗口分区（SAM 模型用）
GET_REL_POS/ADD_REL_POS 相对位置编码
```

#### F. 卷积与池化（约 12 种）
```
CONV_1D, CONV_2D, CONV_3D        卷积
CONV_TRANSPOSE_1D/2D             转置卷积
CONV_2D_DW                      深度可分离卷积
IM2COL                          卷积辅助操作
POOL_1D, POOL_2D                池化
```

#### G. 激活函数（约 20 种，通过 UNARY 实现）
```
RELU, GELU, SILU, TANH, SIGMOID   常用激活
HARDSWISH, HARDSIGMOID            硬近似
SWIGLU, GEGLU, REGLU              GLU 变体
```

#### H. 自定义操作
```c
// 注册自定义操作（SAM 中的 sin/cos 就是用这个实现的）
ggml_map_custom1(ctx, tensor, my_func, n_tasks, userdata);
```

### 2.3 计算图的内存管理

**关键文件**：`include/ggml-alloc.h`、`src/ggml-alloc.c`

`ggml_gallocr`（Graph Allocator）是理解 GGML 内存效率的关键：

```
计算图节点:  [A] → [B] → [C] → [D]
内存生命周期: t0    t0-t1   t1-t2   t2-t3

gallocr 分析:
- B 和 D 的生命周期不重叠 → 共享同一块内存
- 只需为最大同时活跃内存量分配空间
```

**关键流程**：
```c
ggml_gallocr_t allocr = ggml_gallocr_new(buffer_type);
ggml_gallocr_alloc_graph(allocr, graph);  // 分析图 → 重用内存 → 分配
```

**实践练习**：
- 阅读 `examples/magika/main.cpp` 中的 `magika_eval`，看 `ggml_gallocr` 如何为复杂 MLP 图分配内存
- 阅读 `examples/sam/sam.cpp` 中的 `sam_encode_image`，看大规模 ViT 图的内存分配

---

## 第三阶段：后端抽象与多硬件执行

**目标**：理解 GGML 如何在多种硬件上透明执行

### 3.1 后端抽象层架构

**关键文件**：`include/ggml-backend.h`、`src/ggml-backend.cpp`

```
ggml_backend_t           ── 后端实例（CPU/CUDA/Metal等）
ggml_backend_buffer_t    ── 内存缓冲区（后端管理的内存区域）
ggml_backend_buffer_type_t ── 缓冲区类型（控制分配策略）
ggml_backend_dev_t       ── 设备句柄（代表一个物理设备）
ggml_backend_sched_t     ── 调度器（将图分配到多个后端）
```

**后端接口（虚函数表设计）**：

```c
struct ggml_backend_i {
    const char * (*get_name)(ggml_backend_t backend);
    void (*free)(ggml_backend_t backend);
    ggml_backend_buffer_t (*alloc_buffer)(ggml_backend_t backend, size_t size);
    size_t (*get_alignment)(ggml_backend_t backend);
    void (*set_tensor_async)(ggml_backend_t backend, struct ggml_tensor * tensor, const void * data, size_t offset, size_t size);
    void (*get_tensor_async)(ggml_backend_t backend, const struct ggml_tensor * tensor, void * data, size_t offset, size_t size);
    bool (*graph_compute)(ggml_backend_t backend, struct ggml_cgraph * cgraph);
    // ... 更多函数
};
```

每个后端实现这套接口，核心工作包括：
- 内存分配和管理
- 张量数据的传输（host ↔ device）
- 计算图的执行（将 ggml_op 映射到硬件指令）

### 3.2 后端调度器（Scheduler）

**关键文件**：`src/ggml-backend.cpp` 中的 `ggml_backend_sched` 相关代码

调度器解决的核心问题：**如何将一个计算图分配到多个后端上执行？**

```
计算图: [CPU算子] → [GPU算子] → [CPU算子] → [GPU算子]
         ↓            ↓           ↓           ↓
调度器: 分配到CPU   分配到CUDA   分配到CPU   分配到CUDA
                 ↕              ↕
           自动插入数据传输节点
```

**使用模式**（来自 `examples/simple/simple-backend.cpp` 和 `examples/mnist/`）：

```c
// 1. 创建多个后端
ggml_backend_t gpu_backend = ggml_backend_init_best();
ggml_backend_t cpu_backend = ggml_backend_init_by_type(GGML_BACKEND_DEVICE_TYPE_CPU, NULL);

// 2. 创建调度器（按优先级排列后端）
ggml_backend_t backends[] = { gpu_backend, cpu_backend };
ggml_backend_sched_t sched = ggml_backend_sched_new(backends, NULL, 2, ...);

// 3. 分配并执行
ggml_backend_sched_reset(sched);
ggml_backend_sched_alloc_graph(sched, graph);
ggml_backend_sched_graph_compute(sched, graph);
```

### 3.3 CPU 后端内部结构

**关键文件**：`src/ggml-cpu/`

```
ggml-cpu/
├── ggml-cpu.cpp          主入口，注册后端
├── ops.cpp               操作实现分派
├── binary-ops.cpp        二元运算（add/sub/mul/div）
├── unary-ops.cpp         一元运算（relu/gelu/silu/...）
├── vec.h/vec.cpp         SIMD 向量化基础
├── simd-mappings.h       SIMD 操作映射
├── arch/
│   ├── arm/              ARM NEON 优化
│   └── x86/              x86 AVX/AVX2/AVX-512 优化
└── amx/                  Intel AMX 矩阵扩展
    ├── amx.cpp/h         AMX 初始化和检测
    ├── mmq.cpp/h         矩阵乘法（AMX 版本）
    └── common.h          AMX 公共定义
```

**SIMD 加速链**：
```
ggml_mul_mat → 检测 CPU 特性 → 选择最优实现
  ├── AVX-512 (Intel Sapphire Rapids)
  ├── AVX2 (大多数现代 x86)
  ├── NEON (ARM)
  └── 标量回退
```

### 3.4 GPU 后端（以 CUDA 为例）

**关键文件**：`src/ggml-cuda/`

```
ggml-cuda/
├── ggml-cuda.cu          主入口
├── fattn-*.cu            Flash Attention 多种实现
├── mmq-*.cu              量化矩阵乘法
├── mma.cuh               Matrix Multiply Accumulate
├── rope.cu               RoPE 实现
├── conv2d-impl.cuh       2D 卷积
├── template-instances/   模板实例化（代码生成）
└── vendors/              供应商特定优化
```

**实践练习**：
- 阅读 `examples/mnist/mnist-common.h` 的构造函数，看多后端调度器的完整使用
- 阅读 `examples/yolo/yolov3-tiny.cpp` 的 `create_backend`，看后端选择策略

---

## 第四阶段：模型格式与序列化

**目标**：掌握 GGUF 模型文件格式

### 4.1 GGUF 文件格式

**关键文件**：`include/gguf.h`、`src/gguf.cpp`

GGUF（GGML Universal File）是 GGML 的标准模型格式：

```
GGUF 文件结构:
┌─────────────────────┐
│ Magic: "GGUF"       │  4 bytes
│ Version: 3          │  4 bytes
│ Tensor count        │  8 bytes
│ Metadata KV count   │  8 bytes
├─────────────────────┤
│ Metadata KV pairs   │  可变长度
│  ├── "general.architecture" = "gpt2"
│  ├── "general.name" = "GPT-2"
│  ├── "gpt2.n_embd" = 768
│  └── ...
├─────────────────────┤
│ Tensor info array   │  每个张量一条记录
│  ├── name, dims, type, offset
│  └── ...
├─────────────────────┤
│ Padding (alignment) │
├─────────────────────┤
│ Tensor data         │  所有张量的连续数据
│  ├── weight_0 data
│  ├── weight_1 data
│  └── ...
└─────────────────────┘
```

**关键特性**：
- **mmap 兼容**：张量数据可以直接 mmap，无需读入内存
- **单文件部署**：模型架构、超参数、权重都在一个文件中
- **元数据系统**：KV 对支持 string/uint32/float32/array 等多种类型
- **对齐支持**：张量数据按 backend 要求对齐

### 4.2 模型加载标准流程

**来自 `examples/magika/main.cpp`**：

```c
// 1. 从文件初始化 GGUF 上下文
struct gguf_init_params params = { .no_alloc = true, .ctx = &ctx };
struct gguf_context * ctx_gguf = gguf_init_from_file(fname, params);

// 2. 分配后端内存
ggml_backend_buffer_t buf = ggml_backend_alloc_ctx_tensors(ctx, backend);

// 3. 逐个张量加载数据
for (int i = 0; i < n_tensors; i++) {
    const char * name = gguf_get_tensor_name(ctx_gguf, i);
    struct ggml_tensor * tensor = ggml_get_tensor(ctx, name);
    size_t offs = gguf_get_data_offset(ctx_gguf) + gguf_get_tensor_offset(ctx_gguf, i);
    // 从文件读取数据 → ggml_backend_tensor_set(tensor, data, 0, size)
}
```

**实践练习**：
- 阅读 `examples/magika/convert.py`，看 TensorFlow 模型如何转换为 GGUF
- 阅读 `examples/yolo/yolov3-tiny.cpp` 的 `load_model`，看完整的 GGUF 加载流程

---

## 第五阶段：训练与优化框架

**目标**：理解 GGML 的训练能力

### 5.1 ggml-opt 训练框架

**关键文件**：`include/ggml-opt.h`、`src/ggml-opt.cpp`

```
ggml-opt 组件:
├── ggml_opt_dataset_t    数据集管理（数据+标签，支持分片和打乱）
├── ggml_opt_context_t    训练上下文（前向/反向/优化器）
├── ggml_opt_result_t     结果收集（预测、损失、准确率）
│
├── 损失函数
│   ├── CROSS_ENTROPY     交叉熵（多分类）
│   ├── MSE               均方误差（回归）
│   └── MEAN/SUM          简单均值/求和
│
├── 优化器
│   ├── ADAMW             Adam with Weight Decay
│   └── SGD               随机梯度下降
│
└── 构建模式
    ├── FORWARD           仅前向（推理/评估）
    ├── FORWARD_BACKWARD  前向+反向（梯度计算）
    └── UPDATE            前向+反向+参数更新（完整训练步）
```

### 5.2 训练流程

**来自 `examples/mnist/`**：

```c
// 高级 API（一行训练）
ggml_opt_fit(sched, ctx_compute, inputs, outputs, dataset,
    GGML_OPT_LOSS_TYPE_CROSS_ENTROPY,
    GGML_OPT_OPTIMIZER_TYPE_ADAMW,
    ggml_opt_get_default_optimizer_params,
    nepoch, nbatch_logical, val_split, false);

// 低级 API（更多控制）
ggml_opt_params params = ggml_opt_default_params(sched, loss_type);
params.ctx_compute = ctx_compute;
params.inputs = images;
params.outputs = logits;
params.build_type = GGML_OPT_BUILD_TYPE_FORWARD;  // 仅前向
ggml_opt_context_t opt_ctx = ggml_opt_init(params);
ggml_opt_epoch(opt_ctx, dataset, nullptr, result, 0, nullptr, nullptr);
```

### 5.3 梯度累积

MNIST 示例中的关键设计：
- **逻辑批次**（logical batch）：梯度更新的数据量 = 1000
- **物理批次**（physical batch）：单次并行计算的数据量 = 500
- 累积 2 个物理批次的梯度后再更新参数

```
逻辑批次 1000 个样本:
  物理批次 1: 计算 500 个 → 累积梯度
  物理批次 2: 计算 500 个 → 累积梯度
  更新参数
```

**实践练习**：
- 完整阅读 `examples/mnist/mnist-common.cpp` 的 `mnist_model_train` 和 `mnist_model_eval`
- 阅读 `tests/test-opt.cpp` 了解训练的单元测试

---

## 第六阶段：实战项目精读

**目标**：通过完整项目深入理解各组件的协作方式

### 6.1 推荐精读顺序

| 序号 | 项目 | 难度 | 核心知识点 | 关键文件 |
|------|------|------|-----------|---------|
| 1 | **simple** | 入门 | context vs backend 两种 API | `simple-ctx.cpp`, `simple-backend.cpp` |
| 2 | **magika** | 简单 | GGUF 加载、MLP 构建、批量推理、gallocr | `main.cpp` |
| 3 | **mnist** | 中等 | CNN/FC、训练、评估、多后端调度、序列化 | `mnist-common.h/cpp` |
| 4 | **yolo** | 中等 | CNN推理、特征金字塔、NMS后处理 | `yolov3-tiny.cpp` |
| 5 | **python** | 中等 | Python绑定、量化透明处理 | `ggml/utils.py` |
| 6 | **sam** | 高级 | ViT编码器、交叉注意力、多阶段推理 | `sam.cpp` |

### 6.2 每个项目的学习重点

#### simple —— API 对比
```
理解目标：两种 API 的设计哲学差异
重点代码：
  - simple-ctx.cpp:  load_model() 中的 no_alloc=false
  - simple-backend.cpp: init_model() 中的 ggml_backend_sched_new
  - 两者 build_graph() 和 compute() 的对比
```

#### magika —— 最小完整推理流程
```
理解目标：从模型文件到推理结果的完整链路
重点代码：
  - magika_model_load(): GGUF 加载 + 后端内存分配
  - magika_graph(): MLP 计算图构建
  - magika_eval(): 输入编码 → 图执行 → 结果解析
  - 文件的 beg/mid/end 采样策略
  - one-hot 编码到 softmax 的完整数据流
```

#### mnist —— 训练全生命周期
```
理解目标：完整的训练/评估/保存/加载循环
重点代码：
  - mnist_model 构造函数: 多后端初始化和 fallback
  - mnist_model_build(): FC 和 CNN 两种架构的图构建
  - mnist_model_train(): ggml_opt_fit 高级训练 API
  - mnist_model_eval(): ggml_opt_epoch 评估流程
  - mnist_model_save(): 模型序列化到 GGUF
  - wasm_eval(): WebAssembly 推理支持
```

#### yolo —— 计算机视觉推理
```
理解目标：CNN 推理 + 后处理的完整流水线
重点代码：
  - load_model(): GGUF 加载到后端缓冲区
  - build_graph(): 13 层 CNN + FPN 结构
  - apply_conv2d(): Conv + BN + LeakyReLU 标准化
  - detect(): letterbox → 推理 → YOLO 解码 → NMS
  - draw_detections(): 可视化输出
```

#### sam —— 大模型推理
```
理解目标：多组件模型的分阶段推理
重点代码：
  - sam_hparams: 参数化模型配置（支持 ViT-B/L/H）
  - sam_encode_image(): ViT 编码器（全局/窗口注意力）
  - sam_encode_prompt(): 点/框提示编码
  - sam_decode_mask(): Two-Way Transformer 解码器
  - ggml_map_custom1: 注册自定义 sin/cos 操作
  - ggml_win_part/win_unpart: 窗口注意力
  - 多阶段执行：先编码图像（慢），再反复解码 prompt（快）
```

---

## 第七阶段：深入实现层

**目标**：理解核心库的内部实现机制

### 7.1 ggml.c 核心实现

**重点阅读**：

1. **张量创建链**：`ggml_new_tensor` → `ggml_new_tensor_impl` → 内存池分配
2. **操作注册链**：`ggml_mul_mat` → `ggml_mul_mat_impl` → 设置 src/op/params
3. **图构建链**：`ggml_build_forward_expand` → 递归遍历 src → 拓扑排序
4. **图执行链**：`ggml_graph_compute` → 按拓扑序遍历 → 分派到后端

### 7.2 量化实现

**关键文件**：`src/ggml-quants.c`、`src/ggml-common.h`

重点理解：
- `quantize_row_q4_0`: F32 → Q4_0 的量化算法
- `dequantize_row_q4_0`: Q4_0 → F32 的反量化算法
- `vec_dot_q4_0_q8_0`: Q4_0 与 Q8_0 的点积（量化域内直接计算）
- K-量化的 super-block 结构：为什么需要两级缩放

### 7.3 后端实现模式

以 CPU 后端为例，理解一个操作从声明到执行的完整链路：

```
ggml_mul_mat(ctx, a, b)          // 声明操作
  ↓
ggml_build_forward_expand(gf, c) // 加入计算图
  ↓
ggml_graph_compute(gf, plan)     // 执行
  ↓
ggml-backend.cpp: 为每个节点选择后端
  ↓
ggml-cpu/ops.cpp: ggml_cpu_mul_mat_op()
  ↓
检测 CPU 特性: AVX2? AVX-512? NEON?
  ↓
调用对应 SIMD 实现: ggml_vec_dot_f32_avx2()
```

---

## 第八阶段：测试与调试

**目标**：掌握验证和调试 GGML 代码的方法

### 8.1 测试文件导读

| 测试文件 | 测试内容 |
|---------|---------|
| `test-backend-ops.cpp` | 所有后端的所有操作正确性 |
| `test-quantize-fns.cpp` | 量化/反量化函数正确性 |
| `test-quantize-perf.cpp` | 量化操作性能基准 |
| `test-opt.cpp` | 训练优化器正确性 |
| `test-conv1d/2d.cpp` | 卷积操作正确性 |
| `test-cont.c` | 连续性检查 |
| `test-dup.c` | 张量复制 |

### 8.2 调试技巧

```c
// 打印计算图结构
ggml_graph_print(cgraph);

// 导出为 DOT 格式（可用 Graphviz 可视化）
ggml_graph_dump_dot(cgraph, NULL, "graph.dot");

// 打印张量数据
float * data = (float *)tensor->data;
for (int i = 0; i < 10; i++) printf("%f ", data[i]);

// Metal GPU Trace
ggml_backend_metal_capture_next_compute(backend);  // 捕获下一帧
```

---

## 关键知识点索引

### 核心概念速查表

| 概念 | 所在文件 | 一句话说明 |
|------|---------|-----------|
| `ggml_tensor` | `ggml.h` | 张量 = 形状 + 步幅 + 类型 + 操作 + 源张量 |
| `ggml_context` | `ggml.h` | 内存池，承载所有张量的元数据和数据 |
| `ggml_cgraph` | `ggml.h` | 计算图，DAG 结构的拓扑排序节点列表 |
| `ggml_gallocr` | `ggml-alloc.h` | 图分配器，分析生命周期自动重用内存 |
| `ggml_backend` | `ggml-backend.h` | 后端，抽象了不同硬件的计算和内存管理 |
| `ggml_backend_sched` | `ggml-backend.h` | 调度器，将计算图自动分配到多个后端 |
| `gguf_context` | `gguf.h` | GGUF 文件上下文，模型序列化格式 |
| `ggml_opt_context` | `ggml-opt.h` | 训练上下文，管理前向/反向/优化 |
| `ggml_op` | `ggml.h` | 186 种操作枚举，覆盖 ML 所有常见计算 |
| `ggml_type` | `ggml.h` | 42 种数据类型，从 F32 到 1-bit 量化 |

### API 演进路径

```
Level 0: ggml.h        纯 C API，张量 + 操作 + 图
Level 1: ggml-alloc.h  内存管理抽象
Level 2: ggml-backend.h 硬件后端抽象
Level 3: gguf.h         模型序列化
Level 4: ggml-opt.h     训练框架
Level 5: ggml-cpp.h     C++ 封装
```

---

## 学习时间建议

| 阶段 | 建议时间 | 核心产出 |
|------|---------|---------|
| 第一阶段 | 1-2 天 | 能用 context API 创建张量、执行简单运算 |
| 第二阶段 | 2-3 天 | 能构建完整计算图（MLP/CNN），理解操作分类 |
| 第三阶段 | 2-3 天 | 能使用 backend API 执行计算，理解多后端调度 |
| 第四阶段 | 1 天 | 能加载和保存 GGUF 模型文件 |
| 第五阶段 | 2 天 | 能使用 ggml-opt 训练简单模型 |
| 第六阶段 | 3-5 天 | 精读 2-3 个完整示例项目 |
| 第七阶段 | 3-5 天 | 阅读 ggml.c 核心实现，理解内部机制 |
| 第八阶段 | 1-2 天 | 能编写测试、调试计算图 |

**总计约 15-23 天**（根据背景和深度调整）

---

## 学习资源索引

| 资源 | 路径 | 用途 |
|------|------|------|
| 核心头文件 | `include/ggml.h` | API 参考手册 |
| 后端接口 | `include/ggml-backend.h` | 后端抽象参考 |
| 核心实现 | `src/ggml.c` | 内部实现参考 |
| 量化结构 | `src/ggml-common.h` | 量化 block 结构定义 |
| 入门示例 | `examples/simple/` | 最简使用范例 |
| 训练示例 | `examples/mnist/` | 完整训练流程 |
| 测试用例 | `tests/` | 操作正确性参考 |
| Python 绑定 | `examples/python/` | Python 互操作参考 |
