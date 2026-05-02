#ifndef GGML_SYCL_BINBCAST_HPP  // 如果未定义 GGML_SYCL_BINBCAST_HPP 则编译
#define GGML_SYCL_BINBCAST_HPP  // 宏定义 GGML_SYCL_BINBCAST_HPP
#include "common.hpp"  // 引入 common.hpp 头文件


static __dpct_inline__ float op_repeat(const float a, const float b) {
    return b;  // 返回
    GGML_UNUSED(a);
}

static __dpct_inline__ float op_add(const float a, const float b) {
    return a + b;  // 返回
}

static __dpct_inline__ float op_sub(const float a, const float b) {
    return a - b;  // 返回
}

static __dpct_inline__ float op_mul(const float a, const float b) {
    return a * b;  // 返回
}

static __dpct_inline__ float op_div(const float a, const float b) {
    return a / b;  // 返回
}

void ggml_sycl_add(ggml_backend_sycl_context & ctx, ggml_tensor * dst);  // ggml_sycl_add

void ggml_sycl_sub(ggml_backend_sycl_context & ctx, ggml_tensor * dst);  // ggml_sycl_sub

void ggml_sycl_mul(ggml_backend_sycl_context & ctx, ggml_tensor * dst);  // ggml_sycl_mul

void ggml_sycl_div(ggml_backend_sycl_context & ctx, ggml_tensor * dst);  // ggml_sycl_div

void ggml_sycl_repeat(ggml_backend_sycl_context & ctx, ggml_tensor * dst);  // ggml_sycl_repeat


#endif //GGML_SYCL_BINBCAST_HPP  // 条件编译结束

