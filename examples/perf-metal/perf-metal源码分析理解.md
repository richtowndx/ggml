# perf-metal 源码分析理解

## 概述

perf-metal 是一个专门用于**Apple Metal GPU 后端性能测试和调试**的工具。它构建一个可配置大小的虚拟计算图，在 Metal 后端上反复执行，并支持 Xcode GPU Trace 捕获，用于分析 Metal 后端的计算性能。该工具是 GGML 中唯一针对 Metal 后端的专用性能测试工具。

## 文件结构

| 文件 | 功能 |
|------|------|
| `perf-metal.cpp` | 完整的性能测试工具 |
| `CMakeLists.txt` | 构建配置（仅链接 ggml 核心库） |

## 核心流程分析

### 1. 初始化

```cpp
int n_op = 1024;    // 计算图中的操作数（默认 1024）
int n_iter = 128;    // 重复执行次数（默认 128）
```

程序接受两个命令行参数控制测试规模：
- `n_op`：计算图中 `mul_mat` + `scale` 操作对的数量
- `n_iter`：性能测量的迭代次数

### 2. Metal 后端初始化

```cpp
ggml_backend_t backend = ggml_backend_metal_init();
```

直接使用 Metal 专用初始化函数创建后端。如果初始化失败（如在非 macOS 平台），程序直接退出。

### 3. 测试张量创建

```cpp
const int ne00 = 8, ne01 = 8, ne11 = 8;  // 小尺寸 8x8 矩阵
struct ggml_tensor * t0 = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, 8, 8);
struct ggml_tensor * t1 = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, 8, 8);
```

使用很小的张量（8x8），目的是让 GPU 计算本身极快，从而测量**调度和提交开销**而非计算时间。

### 4. 计算图构建

构建的虚拟图结构如下：

```
x = mul_mat(t0, t1)
x = scale(x, 1.0)
x = mul_mat(x, t1)      // 链式 matmul
x = scale(x, 1.0)
... 重复 n_op - 1 次 ...
x = scale(x, 42.0)      // 最终乘以 42，用于验证计算正确性
```

- 每个迭代包含 `mul_mat` + `scale` 两个操作
- 所有 `scale` 系数为 1.0（不改变数值），仅最后一个 scale 乘以 42
- 最终结果应该是 `t0 * t1^T` 的各元素乘以 42

这个设计的巧妙之处：scale(x, 1.0) 是一个 no-op 操作，用于增加图节点数而不改变计算结果，从而测试 Metal 后端处理大量节点的调度能力。

### 5. 性能测量

```cpp
// 1. 预热（warm-up）
ggml_backend_graph_compute(backend, gf);

// 2. 计时测量
const int64_t t_start = ggml_time_us();
for (int iter = 0; iter < n_iter; iter++) {
    ggml_backend_graph_compute(backend, gf);
}
const int64_t t_end = ggml_time_us();
```

标准的预热+计时模式，输出每次迭代的平均毫秒数。

### 6. GPU Trace 捕获

```cpp
ggml_backend_metal_capture_next_compute(backend);
ggml_backend_graph_compute(backend, gf);
```

这是该工具最重要的功能。连续调用三次 `ggml_backend_metal_capture_next_compute`，触发 Metal GPU Frame Capture。开发者可以在 Xcode Instruments 中打开生成的 `.gputrace` 文件，分析：
- 每个 kernel 的执行时间
- 内存访问模式
- GPU 利用率
- 串行化瓶颈

使用方法（源码注释中给出）：
```bash
rm -rf /tmp/perf-metal.gputrace
make -j perf-metal && METAL_CAPTURE_ENABLED=1 ./bin/perf-metal
open /tmp/perf-metal.gputrace
```

### 7. 结果验证

```cpp
struct ggml_tensor * res = ggml_graph_node(gf, -1);
// ... 读取并打印结果矩阵 ...
```

打印最终计算结果，确认数值正确性（应全为 42.0 的倍数），验证计算图在优化和执行过程中没有引入错误。

## 展示的 GGML 关键特性

1. **Metal 后端专用 API**：`ggml_backend_metal_init()`、`ggml_backend_metal_capture_next_compute()`
2. **自定义大小计算图**：`ggml_new_graph_custom(ctx, 4*n_op, false)` 创建非默认大小的计算图
3. **`ggml_gallocr` 内存管理**：自动为计算图的中间张量分配 Metal 缓冲区
4. **GPU Frame Capture**：Metal 特有的性能分析能力，与 Xcode 工具链集成
5. **大规模图测试**：通过参数化节点数测试后端调度器的极限
