#pragma once  // 防止重复包含

// clang-format off
#include <cstdint>  // 引入 cstdint 头文件
#include <cstddef>  // 引入 cstddef 头文件

#include <ggml-backend.h>  // 引入 ggml-backend.h 头文件

#include "backend-convert.h"  // 引入 backend-convert.h 头文件
#include "backend-virgl-apir.h"  // 引入 backend-virgl-apir.h 头文件
#include "shared/apir_backend.h"  // 引入 shared/apir_backend.h 头文件
#include "shared/apir_cs.h"  // 引入 shared/apir_cs.h 头文件
#include "shared/apir_cs_ggml.h"  // 引入 shared/apir_cs_ggml.h 头文件
// clang-format on

#define GGML_VIRTGPU_BCK "ggml-virtgpu-backend: "  // 宏定义 GGML_VIRTGPU_BCK

struct virgl_apir_context {  // 结构体定义
    uint32_t               ctx_id;
    virgl_apir_callbacks * iface;
};

typedef uint32_t (*backend_dispatch_t)(apir_encoder * enc, apir_decoder * dec, virgl_apir_context * ctx);  // 类型定义

#include "backend-dispatched.gen.h"  // 引入 backend-dispatched.gen.h 头文件

uint32_t backend_dispatch_initialize(void * ggml_backend_reg_fct_p);  // backend_dispatch_initialize
