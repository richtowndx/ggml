#ifndef GGML_SYCL_GLA_HPP  // 如果未定义 GGML_SYCL_GLA_HPP 则编译
#define GGML_SYCL_GLA_HPP  // 宏定义 GGML_SYCL_GLA_HPP

#include "common.hpp"  // 引入 common.hpp 头文件

void ggml_sycl_op_gated_linear_attn(ggml_backend_sycl_context & ctx, ggml_tensor * dst);  // ggml_sycl_op_gated_linear_attn

#endif  // GGML_SYCL_GLA_HPP  // 条件编译结束
