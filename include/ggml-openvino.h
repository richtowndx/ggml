#pragma once  // 防止重复包含

#include "ggml-backend.h"  // 引入 ggml-backend.h 头文件

#include <cstring>  // 引入 cstring 头文件

#ifdef __cplusplus  // 如果定义了 __cplusplus 则编译
extern "C" {  // C 链接声明
#endif  // 条件编译结束

#define GGML_OPENVINO_NAME "OPENVINO"  // 宏定义 GGML_OPENVINO_NAME

// backend API
GGML_BACKEND_API ggml_backend_t ggml_backend_openvino_init(int device);

GGML_BACKEND_API bool ggml_backend_is_openvino(ggml_backend_t backend);

GGML_BACKEND_API bool ggml_backend_buffer_is_openvino(ggml_backend_buffer_t buffer);

GGML_BACKEND_API bool ggml_backend_buft_is_openvino(ggml_backend_buffer_type_t buft);

GGML_BACKEND_API bool ggml_backend_buft_is_openvino_host(ggml_backend_buffer_type_t buft);

GGML_BACKEND_API size_t ggml_backend_openvino_buffer_get_ctx_id(ggml_backend_buffer_t buffer);

// device buffer
GGML_BACKEND_API ggml_backend_buffer_type_t ggml_backend_openvino_buffer_type(int device);

GGML_BACKEND_API ggml_backend_buffer_type_t ggml_backend_openvino_host_buffer_type(int device);

GGML_BACKEND_API int ggml_backend_openvino_get_device_count(void);

GGML_BACKEND_API ggml_backend_reg_t ggml_backend_openvino_reg(void);

#ifdef __cplusplus  // 如果定义了 __cplusplus 则编译
}
#endif  // 条件编译结束
