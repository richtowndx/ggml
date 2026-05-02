#ifndef GGML_SYCL_COUNT_EQUAL_HPP  // 如果未定义 GGML_SYCL_COUNT_EQUAL_HPP 则编译
#define GGML_SYCL_COUNT_EQUAL_HPP  // 宏定义 GGML_SYCL_COUNT_EQUAL_HPP
#include "common.hpp"  // 引入 common.hpp 头文件

#define SYCL_COUNT_EQUAL_CHUNK_SIZE 128  // 宏定义 SYCL_COUNT_EQUAL_CHUNK_SIZE

void ggml_sycl_count_equal(ggml_backend_sycl_context & ctx, ggml_tensor * dst);  // ggml_sycl_count_equal

#endif //GGML_SYCL_COUNT_EQUAL_HPP  // 条件编译结束
