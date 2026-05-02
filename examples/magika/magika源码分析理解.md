# Magika 源码分析理解

## 概述

Magika 是一个基于 GGML 的**文件类型识别**（file type identification）工具。它利用深度学习模型，通过读取文件的二进制内容来推断文件的实际类型（如 PDF、JPEG、Python、ZIP 等），支持 113 种文件类型标签。该项目展示了如何使用 GGML 框架加载 GGUF 格式模型、构建计算图并进行批量推理。

## 文件结构

| 文件 | 功能 |
|------|------|
| `main.cpp` | 核心源码：模型加载、计算图构建、推理执行 |
| `convert.py` | 模型转换工具：将 TensorFlow/Keras 模型转换为 GGUF 格式 |
| `CMakeLists.txt` | 构建配置 |

## 关键数据结构

### `magika_hparams` — 超参数配置

```cpp
struct magika_hparams {
    const int block_size = 4096;        // 文件读取总块大小
    const int beg_size = 512;           // 文件头部读取大小
    const int mid_size = 512;           // 文件中部读取大小
    const int end_size = 512;           // 文件尾部读取大小
    const int min_file_size_for_dl = 16;
    const int n_label = 113;            // 分类标签数
    const float f_norm_eps = 0.001f;    // LayerNorm 的 epsilon
    const int padding_token = 256;      // 填充 token 值（用于短文件）
};
```

这个设计说明模型并不读取整个文件，而是智能采样文件头部（beg）、中部（mid）和尾部（end）各 512 字节，总计 1536 字节。短于阈值的文件用 padding_token=256 填充。

### `magika_model` — 模型权重

模型是一个多层感知机（MLP）架构，包含：

- **dense** 层：全连接 + GELU 激活
- **layer_normalization**：第一个 LayerNorm
- **dense_1** + **dense_2**：两个全连接层 + GELU 激活，形成残差瓶颈结构
- **global_max_pooling1d**：1D 全局最大池化
- **layer_normalization_1**：第二个 LayerNorm
- **target_label**：最终输出层 + softmax

## 核心流程分析

### 1. 模型加载 (`magika_model_load`)

- 使用 `gguf_init_from_file` 从 GGUF 文件加载模型元数据和张量信息
- 调用 `ggml_backend_alloc_ctx_tensors` 为所有张量分配后端内存
- 逐个张量从文件中读取权重数据，通过 `ggml_backend_tensor_set` 写入后端缓冲区
- 使用 `checked_get_tensor` 安全获取张量（缺失时报错退出）

### 2. 计算图构建 (`magika_graph`)

模型的计算图清晰展示了数据流：

```
input [257, 1536, n_files]     // one-hot 编码的文件字节
  → dense (mul_mat + bias)     // [128, 1536, n_files]
  → GELU
  → reshape + transpose        // [384, 512, n_files]
  → layer_norm
  → dense_1 (mul_mat + bias)   // [256, 384, n_files]
  → GELU
  → dense_2 (mul_mat + bias)   // [256, 384, n_files]
  → GELU
  → global_max_pool_1d         // [1, 256, n_files]
  → reshape                    // [256, n_files]
  → layer_norm_1
  → target_label (mul_mat + bias) // [113, n_files]
  → softmax
  → target_label_probs
```

关键特性：
- **One-hot 编码**：每个字节值（0-255）加上 padding_token（256），共 257 维
- **批量推理**：支持 `n_files` 维度，一次前向传播处理多个文件
- **使用 `ggml_gallocr`**：通过图分配器自动管理中间张量的内存

### 3. 推理执行 (`magika_eval`)

文件内容的读取策略：

1. **头部（beg）**：从文件偏移 0 开始读 512 字节，不足部分尾部填充
2. **中部（mid）**：从文件中间位置读 512 字节，不足时两端填充
3. **尾部（end）**：从文件末尾读 512 字节，不足时头部填充

读取后将每个字节转换为 257 维 one-hot 向量，通过 `ggml_backend_tensor_set` 设置输入张量，最后调用 `ggml_backend_graph_compute` 执行计算。

结果输出：对每个文件打印 top-5 概率最高的文件类型标签及其置信度百分比。

### 4. 模型转换 (`convert.py`)

从 TensorFlow/Keras `.h5` 模型转换为 GGUF 格式：
- 遍历所有有权重层的 layer
- 对每个权重做转置后写入 GGUF（`weight_data.T`）
- 直接调用 `gguf` Python 库的 API

## 展示的 GGML 关键特性

1. **GGUF 模型格式**：标准化的模型文件格式，包含张量元数据和权重数据
2. **Backend 抽象**：通过 `ggml_backend_t` 实现后端无关的内存管理和计算
3. **计算图模式**：声明式构建计算图，由后端统一执行
4. **`ggml_gallocr` 自动内存分配**：根据计算图自动分析并分配所有中间张量的内存
5. **批量推理**：在计算图中嵌入 batch 维度，一次执行处理多个输入
6. **自定义操作组合**：灵活使用 `ggml_mul_mat`、`ggml_gelu`、`ggml_norm`、`ggml_pool_1d`、`ggml_soft_max` 等原语组合出复杂网络
