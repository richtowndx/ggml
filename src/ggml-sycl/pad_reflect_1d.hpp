#ifndef GGML_SYCL_PAD_REFLECT_1D_HPP  // 如果未定义 GGML_SYCL_PAD_REFLECT_1D_HPP 则编译
#define GGML_SYCL_PAD_REFLECT_1D_HPP  // 宏定义 GGML_SYCL_PAD_REFLECT_1D_HPP

#include "common.hpp"  // 引入 common.hpp 头文件

#define SYCL_PAD_REFLECT_1D_BLOCK_SIZE 256  // 宏定义 SYCL_PAD_REFLECT_1D_BLOCK_SIZE

void ggml_sycl_op_pad_reflect_1d(ggml_backend_sycl_context& ctx, ggml_tensor* dst);  // ggml_sycl_op_pad_reflect_1d

#endif // GGML_SYCL_PAD_REFLECT_1D_HPP  // 条件编译结束
