#pragma once  // 防止重复包含

#include "common.h"  // 引入 common.h 头文件

#ifdef __cplusplus  // 如果定义了 __cplusplus 则编译
extern "C" {  // C 链接声明
#endif  // 条件编译结束

void ggml_compute_forward_add_non_quantized(const struct ggml_compute_params * params, struct ggml_tensor * dst);  // ggml_compute_forward_add_non_quantized
void ggml_compute_forward_sub(const struct ggml_compute_params * params, struct ggml_tensor * dst);  // ggml_compute_forward_sub
void ggml_compute_forward_mul(const struct ggml_compute_params * params, struct ggml_tensor * dst);  // ggml_compute_forward_mul
void ggml_compute_forward_div(const struct ggml_compute_params * params, struct ggml_tensor * dst);  // ggml_compute_forward_div

#ifdef __cplusplus  // 如果定义了 __cplusplus 则编译
}
#endif  // 条件编译结束
