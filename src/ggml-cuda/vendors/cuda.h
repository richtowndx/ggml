#pragma once  // 防止重复包含

#include <cuda_runtime.h>  // 引入 cuda_runtime.h 头文件
#include <cuda.h>  // 引入 cuda.h 头文件
#include <cublas_v2.h>  // 引入 cublas_v2.h 头文件
#include <cuda_bf16.h>  // 引入 cuda_bf16.h 头文件
#include <cuda_fp16.h>  // 引入 cuda_fp16.h 头文件

#ifdef GGML_USE_NCCL  // 如果定义了 GGML_USE_NCCL 则编译
#include <nccl.h>  // 引入 nccl.h 头文件
#endif // GGML_USE_NCCL  // 条件编译结束

#if CUDART_VERSION >= 11080  // 条件编译
#include <cuda_fp8.h>  // 引入 cuda_fp8.h 头文件
#define FP8_AVAILABLE  // 宏定义 FP8_AVAILABLE
#endif // CUDART_VERSION >= 11080  // 条件编译结束

#if CUDART_VERSION >= 12080  // 条件编译
#include <cuda_fp4.h>  // 引入 cuda_fp4.h 头文件
#endif // CUDART_VERSION >= 12080  // 条件编译结束

#if CUDART_VERSION < 11020  // 条件编译
#define CU_DEVICE_ATTRIBUTE_VIRTUAL_MEMORY_MANAGEMENT_SUPPORTED CU_DEVICE_ATTRIBUTE_VIRTUAL_ADDRESS_MANAGEMENT_SUPPORTED  // 宏定义 CU_DEVICE_ATTRIBUTE_VIRTUAL_MEMORY_MANAGEMENT_SUPPORTED
#define CUBLAS_TF32_TENSOR_OP_MATH CUBLAS_TENSOR_OP_MATH  // 宏定义 CUBLAS_TF32_TENSOR_OP_MATH
#define CUBLAS_COMPUTE_16F CUDA_R_16F  // 宏定义 CUBLAS_COMPUTE_16F
#define CUBLAS_COMPUTE_32F CUDA_R_32F  // 宏定义 CUBLAS_COMPUTE_32F
#define cublasComputeType_t cudaDataType_t  // 宏定义 cublasComputeType_t
#endif // CUDART_VERSION < 11020  // 条件编译结束
