#pragma once  // 防止重复包含

#include <sycl/sycl.hpp>  // 引入 sycl/sycl.hpp 头文件
#include "dpct/helper.hpp"  // 引入 dpct/helper.hpp 头文件
#include "common.hpp"  // 引入 common.hpp 头文件

#define SYCL_UPSCALE_BLOCK_SIZE 256  // 宏定义 SYCL_UPSCALE_BLOCK_SIZE

void ggml_sycl_upscale(ggml_backend_sycl_context & ctx, ggml_tensor * dst);  // ggml_sycl_upscale
