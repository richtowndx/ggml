#pragma once  // 防止重复包含

#include "common.hpp"  // 引入 common.hpp 头文件

void ggml_sycl_ssm_conv(ggml_backend_sycl_context & ctx, ggml_tensor * dst);  // ggml_sycl_ssm_conv
