#ifndef GGML_SYCL_ELEMENTWISE_HPP  // 如果未定义 GGML_SYCL_ELEMENTWISE_HPP 则编译
#define GGML_SYCL_ELEMENTWISE_HPP  // 宏定义 GGML_SYCL_ELEMENTWISE_HPP

#include "common.hpp"  // 引入 common.hpp 头文件
#include "ggml.h"  // 引入 ggml.h 头文件
#include <limits> // For std::numeric_limits  // 引入 limits 头文件

#define SYCL_GLU_BLOCK_SIZE 256  // 宏定义 SYCL_GLU_BLOCK_SIZE

template <typename T>  // 模板
T neg_infinity() {
    return -std::numeric_limits<T>::infinity();
}

template<typename T_Dst, typename T_Src = T_Dst>  // 模板
struct typed_data {  // 结构体定义
    const T_Src * src;
    T_Dst * dst;
};

template<typename T_Dst, typename T_Src = T_Dst>  // 模板
typed_data<T_Dst, T_Src> cast_data(ggml_tensor * dst) {
    return {  // 返回
        /* .src = */ static_cast<const T_Src *>(dst->src[0]->data),
        /* .dst = */ static_cast<T_Dst *>(dst->data)
    };
}

const float GELU_QUICK_COEF = -1.702f;


void ggml_sycl_sqrt(ggml_backend_sycl_context & ctx, ggml_tensor * dst);  // ggml_sycl_sqrt

void ggml_sycl_sin(ggml_backend_sycl_context & ctx, ggml_tensor * dst);  // ggml_sycl_sin

void ggml_sycl_cos(ggml_backend_sycl_context & ctx, ggml_tensor * dst);  // ggml_sycl_cos

void ggml_sycl_acc(ggml_backend_sycl_context & ctx, ggml_tensor * dst);  // ggml_sycl_acc

void ggml_sycl_gelu(ggml_backend_sycl_context & ctx, ggml_tensor * dst);  // ggml_sycl_gelu

void ggml_sycl_silu(ggml_backend_sycl_context & ctx, ggml_tensor * dst);  // ggml_sycl_silu

void ggml_sycl_gelu_quick(ggml_backend_sycl_context & ctx, ggml_tensor * dst);  // ggml_sycl_gelu_quick

void ggml_sycl_swiglu_oai(ggml_backend_sycl_context & ctx, ggml_tensor * dst);  // ggml_sycl_swiglu_oai

void ggml_sycl_gelu_erf(ggml_backend_sycl_context & ctx, ggml_tensor * dst);  // ggml_sycl_gelu_erf

void ggml_sycl_tanh(ggml_backend_sycl_context & ctx, ggml_tensor * dst);  // ggml_sycl_tanh

void ggml_sycl_relu(ggml_backend_sycl_context & ctx, ggml_tensor * dst);  // ggml_sycl_relu

void ggml_sycl_sigmoid(ggml_backend_sycl_context & ctx, ggml_tensor * dst);  // ggml_sycl_sigmoid

void ggml_sycl_hardsigmoid(ggml_backend_sycl_context & ctx, ggml_tensor * dst);  // ggml_sycl_hardsigmoid

void ggml_sycl_hardswish(ggml_backend_sycl_context & ctx, ggml_tensor * dst);  // ggml_sycl_hardswish

void ggml_sycl_exp(ggml_backend_sycl_context & ctx, ggml_tensor * dst);  // ggml_sycl_exp

void ggml_sycl_log(ggml_backend_sycl_context & ctx, ggml_tensor * dst);  // ggml_sycl_log

void ggml_sycl_softplus(ggml_backend_sycl_context & ctx, ggml_tensor * dst);  // ggml_sycl_softplus

void ggml_sycl_neg(ggml_backend_sycl_context & ctx, ggml_tensor * dst);  // ggml_sycl_neg

void ggml_sycl_step(ggml_backend_sycl_context & ctx, ggml_tensor * dst);  // ggml_sycl_step

void ggml_sycl_leaky_relu(ggml_backend_sycl_context & ctx, ggml_tensor * dst);  // ggml_sycl_leaky_relu

void ggml_sycl_sqr(ggml_backend_sycl_context & ctx, ggml_tensor * dst);  // ggml_sycl_sqr

void ggml_sycl_clamp(ggml_backend_sycl_context & ctx, ggml_tensor * dst);  // ggml_sycl_clamp

void ggml_sycl_sgn(ggml_backend_sycl_context & ctx, ggml_tensor * dst);  // ggml_sycl_sgn

void ggml_sycl_abs(ggml_backend_sycl_context & ctx, ggml_tensor * dst);  // ggml_sycl_abs

void ggml_sycl_elu(ggml_backend_sycl_context & ctx, ggml_tensor * dst);  // ggml_sycl_elu

void ggml_sycl_geglu(ggml_backend_sycl_context & ctx, ggml_tensor * dst);  // ggml_sycl_geglu
void ggml_sycl_reglu(ggml_backend_sycl_context & ctx, ggml_tensor * dst);  // ggml_sycl_reglu
void ggml_sycl_swiglu(ggml_backend_sycl_context & ctx, ggml_tensor * dst);  // ggml_sycl_swiglu
void ggml_sycl_geglu_erf(ggml_backend_sycl_context & ctx, ggml_tensor * dst);  // ggml_sycl_geglu_erf
void ggml_sycl_geglu_quick(ggml_backend_sycl_context & ctx, ggml_tensor * dst);  // ggml_sycl_geglu_quick
void ggml_sycl_floor(ggml_backend_sycl_context & ctx, ggml_tensor * dst);  // ggml_sycl_floor
void ggml_sycl_ceil(ggml_backend_sycl_context & ctx, ggml_tensor * dst);  // ggml_sycl_ceil
void ggml_sycl_round(ggml_backend_sycl_context & ctx, ggml_tensor * dst);  // ggml_sycl_round
void ggml_sycl_trunc(ggml_backend_sycl_context & ctx, ggml_tensor * dst);  // ggml_sycl_trunc

void ggml_sycl_arange(ggml_backend_sycl_context & ctx, ggml_tensor * dst);  // ggml_sycl_arange

#endif // GGML_SYCL_ELEMENTWISE_HPP  // 条件编译结束
