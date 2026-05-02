#pragma once  // 防止重复包含
#include "backend.hpp"  // 引入 backend.hpp 头文件
#include "ggml.h"  // 引入 ggml.h 头文件

void ggml_sycl_op_set(ggml_backend_sycl_context & ctx, ggml_tensor * dst);  // ggml_sycl_op_set
