#pragma once  // 防止重复包含

#include "ggml.h"  // 引入 ggml.h 头文件
#include "ggml-backend.h"  // 引入 ggml-backend.h 头文件


#ifdef  __cplusplus  // 如果定义了 __cplusplus 则编译
extern "C" {  // C 链接声明
#endif  // 条件编译结束

// backend API
GGML_BACKEND_API ggml_backend_t ggml_backend_blas_init(void);

GGML_BACKEND_API bool ggml_backend_is_blas(ggml_backend_t backend);

// number of threads used for conversion to float
// for openblas and blis, this will also set the number of threads used for blas operations
GGML_BACKEND_API void ggml_backend_blas_set_n_threads(ggml_backend_t backend_blas, int n_threads);

GGML_BACKEND_API ggml_backend_reg_t ggml_backend_blas_reg(void);


#ifdef  __cplusplus  // 如果定义了 __cplusplus 则编译
}
#endif  // 条件编译结束
