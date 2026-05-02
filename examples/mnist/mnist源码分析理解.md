# MNIST 源码分析理解

## 概述

MNIST 示例是 GGML 框架中最为完整的深度学习示例之一，展示了如何使用 GGML 进行**模型训练、评估和推理**。它基于经典的 MNIST 手写数字识别任务（28x28 像素图像，0-9 共 10 个类别），支持两种网络架构：全连接网络（FC）和卷积神经网络（CNN）。该示例是 GGML 中唯一展示完整训练流程的示例，涵盖了数据加载、模型初始化、训练、评估、保存的全生命周期。

## 文件结构

| 文件 | 功能 |
|------|------|
| `mnist-common.h/cpp` | 共享基础设施：模型定义、数据加载、训练/评估/保存逻辑 |
| `mnist-eval.cpp` | 评估入口：加载已训练模型，在测试集上评估准确率 |
| `mnist-train.cpp` | 训练入口：从随机权重开始训练并保存模型 |
| `CMakeLists.txt` | 构建配置（含 WebAssembly 支持） |

## 关键数据结构与常量

### 核心常量

```cpp
#define MNIST_NTRAIN 60000              // 训练集大小
#define MNIST_NTEST  10000              // 测试集大小
#define MNIST_NBATCH_LOGICAL  1000      // 逻辑批次大小（梯度累积）
#define MNIST_NBATCH_PHYSICAL  500      // 物理批次大小（并行计算）
#define MNIST_HW       28               // 图像高宽
#define MNIST_NINPUT   (28*28)          // 输入维度 784
#define MNIST_NCLASSES 10               // 分类数
#define MNIST_NHIDDEN  500              // FC 隐藏层大小
#define MNIST_CNN_NCB  8                // CNN 基础通道数
```

### `mnist_model` — 模型结构

该结构体统一承载两种架构，核心成员包括：

- **后端调度器** (`backend_sched`)：支持多后端优先级调度，主后端优先，其余作为 fallback
- **FC 架构权重**：`fc1_weight/bias` (784→500)、`fc2_weight/bias` (500→10)
- **CNN 架构权重**：
  - `conv1_kernel/bias`：3x3 卷积，1→8 通道
  - `conv2_kernel/bias`：3x3 卷积，8→16 通道
  - `dense_weight/bias`：全连接输出层
- **多上下文管理**：`ctx_gguf`（模型权重）、`ctx_static`（静态张量如输入图像）、`ctx_compute`（计算图构建）

## 核心流程分析

### 1. 后端初始化 (`mnist_model` 构造函数)

```cpp
mnist_model(const std::string & backend_name, const int nbatch_logical, const int nbatch_physical)
```

后端初始化策略：
1. **主后端**：根据用户指定的后端名称初始化（如 CPU、CUDA0）
2. **Fallback 后端**：遍历所有可用后端设备，全部加入调度器
3. **线程数设置**：CPU 后端自动设置线程数为 `min(ncores, (ncores+4)/2)`
4. **调度器创建**：`ggml_backend_sched_new`，按优先级排列后端，不支持的算子自动降级到 fallback

这展示了 GGML 的**多后端无缝协作**能力。

### 2. 两种网络架构

#### FC（全连接）架构

```
images [784, batch]
  → fc1: mul_mat(784×500) + bias → ReLU
  → fc2: mul_mat(500×10)  + bias
  → logits [10, batch]
```

#### CNN（卷积）架构

```
images [784, batch] → reshape [28, 28, 1, batch]
  → conv1: 3x3, 1→8 channels, padding=1 → ReLU    // [28, 28, 8, batch]
  → max_pool 2x2, stride 2                          // [14, 14, 8, batch]
  → conv2: 3x3, 8→16 channels, padding=1 → ReLU    // [14, 14, 16, batch]
  → max_pool 2x2, stride 2                          // [7, 7, 16, batch]
  → reshape + permute → dense_in [784, batch]
  → dense: mul_mat(784×10) + bias
  → logits [10, batch]
```

### 3. 数据加载

- **图像数据**：从 IDX 格式文件读取，自动归一化到 [0, 1] 范围
- **标签数据**：转换为 one-hot 编码（10 维向量）
- 使用 `ggml_opt_dataset_t` 统一管理数据集

### 4. 训练流程 (`mnist_model_train`)

```cpp
ggml_opt_fit(sched, ctx_compute, images, logits, dataset,
    GGML_OPT_LOSS_TYPE_CROSS_ENTROPY,    // 交叉熵损失
    GGML_OPT_OPTIMIZER_TYPE_ADAMW,        // AdamW 优化器
    ggml_opt_get_default_optimizer_params, // 默认超参数
    nepoch, nbatch_logical, val_split, false);
```

GGML 的 `ggml_opt` 模块提供了高级训练 API：
- **梯度累积**：通过 logical/physical batch 分离，支持超过 GPU 内存的逻辑批次
- **数据分片**：训练时使用 shard_size=10 减少数据打乱开销
- **验证集划分**：`val_split=0.05` 表示 5% 数据用于验证

### 5. 模型保存 (`mnist_model_save`)

保存流程：
1. 创建新的 ggml 上下文
2. 复制权重张量到新上下文
3. 写入 GGUF 文件，包含架构元数据（`general.architecture`）

### 6. 评估流程 (`mnist_model_eval`)

- 使用 `ggml_opt_init` 创建仅前向传播的优化上下文（`GGML_OPT_BUILD_TYPE_FORWARD`）
- 遍历整个测试集，收集预测结果
- 计算损失和准确率（含不确定度）

### 7. WebAssembly 支持

`mnist-common.cpp` 底部包含 `wasm_eval` 和 `wasm_random_digit` 函数，通过 Emscripten 编译为 WebAssembly，支持在浏览器中运行推理。`CMakeLists.txt` 中配置了 Web 版本的完整构建流程。

## 展示的 GGML 关键特性

1. **ggml-opt 训练框架**：GGML 内置的优化/训练子系统，支持前向传播、反向传播、AdamW 优化器
2. **`ggml_backend_sched` 多后端调度**：自动将计算图中的算子分配到最合适的后端
3. **梯度累积**：通过 logical/physical batch 分离实现
4. **GGUF 模型序列化**：完整的模型保存/加载流程
5. **卷积操作**：`ggml_conv_2d` + `ggml_pool_2d` 构建 CNN
6. **交叉熵损失**：多分类任务的标准化损失函数
7. **数据集管理**：`ggml_opt_dataset_t` 提供数据加载、打乱、分片等功能
8. **WebAssembly 部署**：展示 GGML 在浏览器端的推理能力
