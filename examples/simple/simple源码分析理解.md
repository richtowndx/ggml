# Simple 源码分析理解

## 概述

simple 目录包含两个**最小化示例程序**，分别展示了 GGML 的两种使用方式：

1. **`simple-ctx.cpp`**：使用传统 `ggml_context` API 进行矩阵乘法——最简单的 GGML 入门示例
2. **`simple-backend.cpp`**：使用现代 `ggml_backend` API 进行矩阵乘法——展示后端抽象的使用方式

两者执行完全相同的计算（矩阵乘法 `A * B^T`），但使用不同的 API 层级。

## 文件结构

| 文件 | 功能 |
|------|------|
| `simple-ctx.cpp` | 传统 context API 示例 |
| `simple-backend.cpp` | 现代 backend API 示例 |
| `CMakeLists.txt` | 构建配置 |

## 示例 1：`simple-ctx.cpp` — 传统 Context API

### 执行流程

```
load_model()          → 创建 context，分配张量，填充数据
build_graph()         → 构建计算图：result = mul_mat(a, b)
compute()             → ggml_graph_compute_with_ctx() 执行计算
main()                → 读取并打印结果
```

### 关键代码分析

#### 模型结构
```cpp
struct simple_model {
    struct ggml_tensor * a;    // 矩阵 A (4x2)
    struct ggml_tensor * b;    // 矩阵 B (3x2)
    struct ggml_context * ctx; // GGML 上下文
};
```

#### 内存分配策略
```cpp
size_t ctx_size = 0;
ctx_size += rows_A * cols_A * ggml_type_size(GGML_TYPE_F32);  // 张量 a 的数据
ctx_size += rows_B * cols_B * ggml_type_size(GGML_TYPE_F32);  // 张量 b 的数据
ctx_size += 2 * ggml_tensor_overhead();                         // 张量元数据
ctx_size += ggml_graph_overhead();                              // 计算图
ctx_size += 1024;                                                // 开销余量

struct ggml_init_params params {
    .mem_size = ctx_size,
    .mem_buffer = NULL,
    .no_alloc = false,    // 关键：直接分配内存
};
```

- `no_alloc = false`：在 `ggml_init` 时直接分配所有张量的数据内存
- 内存布局紧凑：张量元数据和数据都在同一个 context 内存池中

#### 计算图构建
```cpp
struct ggml_cgraph * build_graph(const simple_model& model) {
    struct ggml_cgraph * gf = ggml_new_graph(model.ctx);
    struct ggml_tensor * result = ggml_mul_mat(model.ctx, model.a, model.b);
    ggml_build_forward_expand(gf, result);
    return gf;
}
```

极简的计算图：只有一个 `mul_mat` 节点。`ggml_build_forward_expand` 将结果节点展开为完整的计算图。

#### 执行与结果获取
```cpp
ggml_graph_compute_with_ctx(model.ctx, gf, n_threads);
return ggml_graph_node(gf, -1);  // 获取最后一个节点（即结果）
```

结果直接通过 `result->data` 指针访问（因为 `no_alloc=false`，数据在 context 内存中）。

## 示例 2：`simple-backend.cpp` — 现代 Backend API

### 执行流程

```
init_model()          → 加载所有后端，创建调度器
build_graph()         → 构建计算图（no_alloc=true，稍后由调度器分配）
compute()             → 调度器分配内存 → 设置数据 → 计算图执行
main()                → 通过 ggml_backend_tensor_get 获取结果
```

### 关键代码分析

#### 后端初始化
```cpp
void init_model(simple_model & model) {
    ggml_backend_load_all();                                         // 加载所有可用后端
    model.backend = ggml_backend_init_best();                        // 最佳后端（通常是 GPU）
    model.cpu_backend = ggml_backend_init_by_type(GGML_BACKEND_DEVICE_TYPE_CPU, nullptr);

    ggml_backend_t backends[2] = { model.backend, model.cpu_backend };
    model.sched = ggml_backend_sched_new(backends, nullptr, 2, ...); // 创建调度器
}
```

- `ggml_backend_load_all()`：动态加载所有可用的后端库（CUDA、Metal、CPU 等）
- `ggml_backend_init_best()`：自动选择最佳后端（优先 GPU）
- 调度器：将主后端和 CPU 后端结合，自动处理算子在不同后端间的分配

#### 计算图构建（no_alloc 模式）
```cpp
struct ggml_init_params params0 = {
    .mem_size = buf_size,
    .mem_buffer = model.buf.data(),
    .no_alloc = true,    // 关键：不立即分配，由调度器后续分配
};
```

- `no_alloc = true`：仅创建张量元数据，不分配数据内存
- 使用预分配的 `std::vector<uint8_t>` 作为 context 内存

#### 数据传输
```cpp
ggml_backend_tensor_set(model.a, matrix_A, 0, ggml_nbytes(model.a));  // CPU → 后端
ggml_backend_tensor_set(model.b, matrix_B, 0, ggml_nbytes(model.b));  // CPU → 后端
```

数据需要显式从主机内存传输到后端内存（如 GPU 显存）。

#### 结果获取
```cpp
std::vector<float> out_data(ggml_nelements(result));
ggml_backend_tensor_get(result, out_data.data(), 0, ggml_nbytes(result));
```

通过 `ggml_backend_tensor_get` 将数据从后端内存拷贝回主机内存。

## 两种 API 的对比

| 特性 | simple-ctx | simple-backend |
|------|-----------|----------------|
| API 层级 | 低级 context API | 高级 backend API |
| 后端支持 | 仅 CPU | CPU + GPU + 任意后端 |
| 内存分配 | `no_alloc=false`，init 时直接分配 | `no_alloc=true`，调度器按需分配 |
| 数据访问 | 直接指针 `tensor->data` | 需通过 `set/get` 函数 |
| 多后端 | 不支持 | 支持调度器自动分配 |
| 代码复杂度 | 极简 | 稍复杂但更灵活 |
| 适用场景 | 快速原型、简单推理 | 生产部署、GPU 加速 |

## 展示的 GGML 关键特性

1. **双 API 层级**：context API（简单直接）vs backend API（灵活强大）
2. **`ggml_mul_mat`**：GGML 中最核心的操作——矩阵乘法
3. **计算图模式**：声明式构建 → 统一执行
4. **`ggml_backend_sched`**：多后端调度器，自动选择最优执行路径
5. **`ggml_backend_load_all`**：运行时动态发现和加载所有可用后端
6. **数据传输 API**：`ggml_backend_tensor_set/get` 主机与设备间的数据传输
