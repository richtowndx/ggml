# YOLO 源码分析理解

## 概述

YOLO（You Only Look Once）示例实现了 **YOLOv3-tiny** 目标检测模型的推理。YOLOv3-tiny 是 YOLO 系列的轻量版本，用于在图像中实时检测和定位多个目标，输出目标的边界框（bounding box）和类别标签。该示例展示了如何使用 GGML 构建卷积神经网络推理流程，以及后处理（NMS、边界框解码等）的实现。

## 文件结构

| 文件 | 功能 |
|------|------|
| `yolov3-tiny.cpp` | 主程序：模型加载、计算图构建、检测与后处理 |
| `yolo-image.h/cpp` | 图像处理工具：加载、缩放、letterbox、绘制检测框和标签 |
| `convert-yolov3-tiny.py` | Darknet 模型到 GGUF 格式转换 |
| `CMakeLists.txt` | 构建配置 |

## 模型架构

YOLOv3-tiny 由 13 个卷积层组成，结构如下：

```
输入图像 [416, 416, 3]
  → conv0 (3x3, 3→16) + BN + LeakyReLU
  → maxpool 2x2
  → conv1 (3x3, 16→32) + BN + LeakyReLU
  → maxpool 2x2
  → conv2 (3x3, 32→64) + BN + LeakyReLU
  → maxpool 2x2
  → conv3 (3x3, 64→128) + BN + LeakyReLU
  → maxpool 2x2
  → conv4 (3x3, 128→256) + BN + LeakyReLU   → 保存为 layer_8
  → maxpool 2x2
  → conv5 (3x3, 256→512) + BN + LeakyReLU
  → maxpool 2x2 (stride=1, size=1)
  → conv6 (3x3, 512→1024) + BN + LeakyReLU
  → conv7 (1x1, 1024→256) + BN + LeakyReLU   → 保存为 layer_13
  → conv8 (3x3, 256→512) + BN + LeakyReLU
  → conv9 (1x1, 512→255) linear              → 输出 1 (layer_15)
  → conv10 (1x1, 256→128) linear             → 从 layer_13 分支
  → upscale 2x
  → concat with layer_8 (沿通道维度)
  → conv11 (3x3, 384→256) + BN + LeakyReLU
  → conv12 (1x1, 256→255) linear             → 输出 2 (layer_22)
```

关键特征：
- **特征金字塔**：类似 FPN 结构，从不同尺度（13x13 和 26x26）输出检测结果
- **残差连接**：`layer_13` 分支后 upscale 并与 `layer_8` concat
- **255 维输出**：3 个 anchor × (5 + 80 类) = 255

## 核心数据结构

### `conv2d_layer` — 卷积层

```cpp
struct conv2d_layer {
    struct ggml_tensor * weights;
    struct ggml_tensor * biases;
    struct ggml_tensor * scales;          // BatchNorm 缩放
    struct ggml_tensor * rolling_mean;    // BatchNorm 均值
    struct ggml_tensor * rolling_variance;// BatchNorm 方差
    int padding = 1;
    bool batch_normalize = true;
    bool activate = true;                 // LeakyReLU vs linear
};
```

### `yolo_layer` — YOLO 检测层

```cpp
struct yolo_layer {
    int classes = 80;                     // COCO 数据集 80 类
    std::vector<int> mask;                // anchor 索引掩码
    std::vector<float> anchors;           // 6 个 anchor 尺寸
    std::vector<float> predictions;       // 检测层输出
};
```

### `detection` — 检测结果

```cpp
struct detection {
    box bbox;                 // {x, y, w, h} 中心坐标 + 宽高
    std::vector<float> prob;  // 每个类别的概率
    float objectness;         // 目标置信度
};
```

## 核心流程分析

### 1. 模型加载 (`load_model`)

使用 GGUF 格式加载：
1. `gguf_init_from_file` 读取模型
2. 在 model context 中复制张量定义（`ggml_dup_tensor`）
3. `ggml_backend_alloc_ctx_tensors` 分配后端内存
4. 逐个将张量数据从临时上下文拷贝到后端缓冲区
5. 按命名规则绑定 13 个卷积层的权重

### 2. 计算图构建 (`build_graph`)

按 YOLOv3-tiny 的层序列构建计算图，每个卷积层的处理：

```cpp
static ggml_tensor * apply_conv2d(ggml_context * ctx, ggml_tensor * input, const conv2d_layer & layer) {
    result = ggml_conv_2d(ctx, layer.weights, input, 1, 1, padding, padding, 1, 1);
    if (batch_normalize) {
        result = (result - rolling_mean) / sqrt(rolling_variance);
        result = result * scales;
    }
    result = result + biases;
    if (activate) result = ggml_leaky_relu(ctx, result, 0.1f, true);
    return result;
}
```

特征融合：
```cpp
result = ggml_upscale(ctx_cgraph, result, 2, GGML_SCALE_MODE_NEAREST);  // 上采样
result = ggml_concat(ctx_cgraph, result, layer_8, 2);                    // 通道拼接
```

### 3. 检测后处理 (`detect`)

#### Letterbox 预处理
```cpp
yolo_image sized = letterbox_image(img, model.width, model.height);
```
保持宽高比缩放到 416x416，空白区域填充 0.5。

#### YOLO 解码 (`apply_yolo`)
- 对 xy 坐标和 objectness/class_prob 应用 sigmoid 激活
- 解码边界框坐标：`x = (col + sigmoid(tx)) / grid_w`

#### 边界框转换 (`get_yolo_box` + `correct_yolo_box`)
```
x = (col + sigmoid(tx)) / grid_w           // 相对坐标
y = (row + sigmoid(ty)) / grid_h
w = exp(tw) * anchor_w / net_w             // 基于先验尺寸
h = exp(th) * anchor_h / net_h
```
然后修正 letterbox 的缩放偏移。

#### NMS（非极大值抑制） (`do_nms_sort`)
```cpp
for each class:
    按概率降序排列所有检测
    for each detection:
        抑制 IoU > 0.45 的后续检测
```

### 4. 结果可视化

```
draw_detections():
  → 为每个有效检测选择颜色（按类别哈希）
  → 绘制边界框 (draw_box_width)
  → 生成标签图像 (get_label，使用预加载的字母图片)
  → 叠加标签到原图 (draw_label)
  → 保存为 JPEG
```

### 5. 后端选择 (`create_backend`)

```cpp
// 优先级：用户指定 > GPU > CPU
if (!params.device.empty()) backend = ggml_backend_dev_init(dev, nullptr);
if (!backend) backend = ggml_backend_init_by_type(GGML_BACKEND_DEVICE_TYPE_GPU, nullptr);
if (!backend) backend = ggml_backend_init_by_type(GGML_BACKEND_DEVICE_TYPE_CPU, nullptr);
```

支持通过 `-d` 参数指定后端设备，并列出所有可用设备及其显存信息。

## 展示的 GGML 关键特性

1. **卷积神经网络构建**：`ggml_conv_2d`、`ggml_pool_2d`、`ggml_leaky_relu` 等 CV 操作
2. **BatchNorm 融合推理**：将 BN 参数直接融合到卷积计算中（减均值、除方差、乘缩放）
3. **特征金字塔**：`ggml_upscale` + `ggml_concat` 实现 FPN 式多尺度特征融合
4. **多输出计算图**：一个图中同时输出两个 YOLO 检测层的结果
5. **后端自动选择**：GPU 优先、CPU 兜底的后端选择策略
6. **GGUF 模型加载**：标准的模型权重加载流程
7. **`ggml_gallocr`**：自动为计算图中的输入和中间张量分配内存
8. **图像 I/O**：通过 stb_image 实现图像加载和保存
