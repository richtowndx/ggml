#pragma once  // 防止重复包含

#include "common.h"  // 引入 common.h 头文件

#ifdef __cplusplus  // 如果定义了 __cplusplus 则编译
extern "C" {  // C 链接声明
#endif  // 条件编译结束

void ggml_compute_forward_abs(const struct ggml_compute_params * params, struct ggml_tensor * dst);  // ggml_compute_forward_abs
void ggml_compute_forward_sgn(const struct ggml_compute_params * params, struct ggml_tensor * dst);  // ggml_compute_forward_sgn
void ggml_compute_forward_neg(const struct ggml_compute_params * params, struct ggml_tensor * dst);  // ggml_compute_forward_neg
void ggml_compute_forward_step(const struct ggml_compute_params * params, struct ggml_tensor * dst);  // ggml_compute_forward_step
void ggml_compute_forward_tanh(const struct ggml_compute_params * params, struct ggml_tensor * dst);  // ggml_compute_forward_tanh
void ggml_compute_forward_elu(const struct ggml_compute_params * params, struct ggml_tensor * dst);  // ggml_compute_forward_elu
void ggml_compute_forward_relu(const struct ggml_compute_params * params, struct ggml_tensor * dst);  // ggml_compute_forward_relu
void ggml_compute_forward_sigmoid(const struct ggml_compute_params * params, struct ggml_tensor * dst);  // ggml_compute_forward_sigmoid
void ggml_compute_forward_hardsigmoid(const struct ggml_compute_params * params, struct ggml_tensor * dst);  // ggml_compute_forward_hardsigmoid
void ggml_compute_forward_exp(const struct ggml_compute_params * params, struct ggml_tensor * dst);  // ggml_compute_forward_exp
void ggml_compute_forward_hardswish(const struct ggml_compute_params * params, struct ggml_tensor * dst);  // ggml_compute_forward_hardswish
void ggml_compute_forward_sqr(const struct ggml_compute_params * params, struct ggml_tensor * dst);  // ggml_compute_forward_sqr
void ggml_compute_forward_sqrt(const struct ggml_compute_params * params, struct ggml_tensor * dst);  // ggml_compute_forward_sqrt
void ggml_compute_forward_sin(const struct ggml_compute_params * params, struct ggml_tensor * dst);  // ggml_compute_forward_sin
void ggml_compute_forward_cos(const struct ggml_compute_params * params, struct ggml_tensor * dst);  // ggml_compute_forward_cos
void ggml_compute_forward_log(const struct ggml_compute_params * params, struct ggml_tensor * dst);  // ggml_compute_forward_log
void ggml_compute_forward_expm1(const struct ggml_compute_params * params, struct ggml_tensor * dst);  // ggml_compute_forward_expm1
void ggml_compute_forward_softplus(const struct ggml_compute_params * params, struct ggml_tensor * dst);  // ggml_compute_forward_softplus
void ggml_compute_forward_floor(const struct ggml_compute_params * params, struct ggml_tensor * dst);  // ggml_compute_forward_floor
void ggml_compute_forward_ceil(const struct ggml_compute_params * params, struct ggml_tensor * dst);  // ggml_compute_forward_ceil
void ggml_compute_forward_round(const struct ggml_compute_params * params, struct ggml_tensor * dst);  // ggml_compute_forward_round
void ggml_compute_forward_trunc(const struct ggml_compute_params * params, struct ggml_tensor * dst);  // ggml_compute_forward_trunc
void ggml_compute_forward_xielu(const struct ggml_compute_params * params, struct ggml_tensor * dst);  // ggml_compute_forward_xielu

#ifdef __cplusplus  // 如果定义了 __cplusplus 则编译
}
#endif  // 条件编译结束
