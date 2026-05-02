#pragma once  // 防止重复包含

#include "ggml.h"  // 引入 ggml.h 头文件
#include "ggml-backend.h"  // 引入 ggml-backend.h 头文件

#ifdef  __cplusplus  // 如果定义了 __cplusplus 则编译
extern "C" {  // C 链接声明
#endif  // 条件编译结束

GGML_BACKEND_API ggml_backend_reg_t ggml_backend_virtgpu_reg();

#ifdef  __cplusplus  // 如果定义了 __cplusplus 则编译
}
#endif  // 条件编译结束
