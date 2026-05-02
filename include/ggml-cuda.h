#pragma once  // 防止重复包含

#include "ggml.h"  // 引入 ggml.h 头文件
#include "ggml-backend.h"  // 引入 ggml-backend.h 头文件

#ifdef  __cplusplus  // 如果定义了 __cplusplus 则编译
extern "C" {  // C 链接声明
#endif  // 条件编译结束

#ifdef GGML_USE_HIP  // 如果定义了 GGML_USE_HIP 则编译
#define GGML_CUDA_NAME "ROCm"  // 宏定义 GGML_CUDA_NAME
#define GGML_CUBLAS_NAME "hipBLAS"  // 宏定义 GGML_CUBLAS_NAME
#elif defined(GGML_USE_MUSA)  // 否则如果
#define GGML_CUDA_NAME "MUSA"  // 宏定义 GGML_CUDA_NAME
#define GGML_CUBLAS_NAME "muBLAS"  // 宏定义 GGML_CUBLAS_NAME
#else  // 否则
#define GGML_CUDA_NAME "CUDA"  // 宏定义 GGML_CUDA_NAME
#define GGML_CUBLAS_NAME "cuBLAS"  // 宏定义 GGML_CUBLAS_NAME
#endif  // 条件编译结束
#define GGML_CUDA_MAX_DEVICES       16  // 宏定义 GGML_CUDA_MAX_DEVICES

// backend API
GGML_BACKEND_API ggml_backend_t ggml_backend_cuda_init(int device);

GGML_BACKEND_API bool ggml_backend_is_cuda(ggml_backend_t backend);

// device buffer
GGML_BACKEND_API ggml_backend_buffer_type_t ggml_backend_cuda_buffer_type(int device);

// conduct allreduce operation between devices
GGML_BACKEND_API bool ggml_backend_cuda_allreduce_tensor(ggml_backend_t * backends, struct ggml_tensor ** tensors, size_t n_backends);

// split tensor buffer that splits matrices by rows across multiple devices
GGML_BACKEND_API ggml_backend_buffer_type_t ggml_backend_cuda_split_buffer_type(int main_device, const float * tensor_split);

// pinned host buffer for use with the CPU backend for faster copies between CPU and GPU
GGML_BACKEND_API ggml_backend_buffer_type_t ggml_backend_cuda_host_buffer_type(void);

GGML_BACKEND_API int  ggml_backend_cuda_get_device_count(void);
GGML_BACKEND_API void ggml_backend_cuda_get_device_description(int device, char * description, size_t description_size);
GGML_BACKEND_API void ggml_backend_cuda_get_device_memory(int device, size_t * free, size_t * total);

GGML_BACKEND_API bool ggml_backend_cuda_register_host_buffer(void * buffer, size_t size);
GGML_BACKEND_API void ggml_backend_cuda_unregister_host_buffer(void * buffer);

GGML_BACKEND_API ggml_backend_reg_t ggml_backend_cuda_reg(void);

#ifdef  __cplusplus  // 如果定义了 __cplusplus 则编译
}
#endif  // 条件编译结束
