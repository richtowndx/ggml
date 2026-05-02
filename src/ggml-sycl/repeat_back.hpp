#ifndef GGML_SYCL_REPEAT_BACK_HPP  // 如果未定义 GGML_SYCL_REPEAT_BACK_HPP 则编译
#define GGML_SYCL_REPEAT_BACK_HPP  // 宏定义 GGML_SYCL_REPEAT_BACK_HPP

#include "common.hpp"  // 引入 common.hpp 头文件

void ggml_sycl_op_repeat_back(ggml_backend_sycl_context & ctx, ggml_tensor * dst);  // ggml_sycl_op_repeat_back

#endif  // GGML_SYCL_REPEAT_BACK_HPP  // 条件编译结束
