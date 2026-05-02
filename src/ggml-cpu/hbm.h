#pragma once  // 防止重复包含

#include "ggml-backend.h"  // 引入 ggml-backend.h 头文件
#include "ggml.h"  // 引入 ggml.h 头文件

// GGML CPU internal header

ggml_backend_buffer_type_t ggml_backend_cpu_hbm_buffer_type(void);  // ggml_backend_cpu_hbm_buffer_type
