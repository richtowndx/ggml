//
//  MIT license
//  Copyright (C) 2024 Intel Corporation
//  SPDX-License-Identifier: MIT
//

#pragma once  // 防止重复包含

#include "ggml.h"  // 引入 ggml.h 头文件
#include "ggml-backend.h"  // 引入 ggml-backend.h 头文件

#define GGML_SYCL_NAME "SYCL"  // 宏定义 GGML_SYCL_NAME
#define GGML_SYCL_MAX_DEVICES 48  // 宏定义 GGML_SYCL_MAX_DEVICES

#ifdef  __cplusplus  // 如果定义了 __cplusplus 则编译
extern "C" {  // C 链接声明
#endif  // 条件编译结束

// backend API
GGML_BACKEND_API ggml_backend_t ggml_backend_sycl_init(int device);

GGML_BACKEND_API bool ggml_backend_is_sycl(ggml_backend_t backend);

// devide buffer
GGML_BACKEND_API ggml_backend_buffer_type_t ggml_backend_sycl_buffer_type(int device);

// split tensor buffer that splits matrices by rows across multiple devices
GGML_BACKEND_API ggml_backend_buffer_type_t ggml_backend_sycl_split_buffer_type(const float * tensor_split);

// pinned host buffer for use with the CPU backend for faster copies between CPU and GPU
GGML_BACKEND_API ggml_backend_buffer_type_t ggml_backend_sycl_host_buffer_type(void);

GGML_BACKEND_API void ggml_backend_sycl_print_sycl_devices(void);
GGML_BACKEND_API void ggml_backend_sycl_get_gpu_list(int *id_list, int max_len);
GGML_BACKEND_API void ggml_backend_sycl_get_device_description(int device,
                                                       char *description,
                                                       size_t description_size);
GGML_BACKEND_API int  ggml_backend_sycl_get_device_count();
GGML_BACKEND_API void ggml_backend_sycl_get_device_memory(int device, size_t *free, size_t *total);

// SYCL doesn't support registering host memory, keep here for reference
// GGML_BACKEND_API bool ggml_backend_sycl_register_host_buffer(void * buffer, size_t size);
// GGML_BACKEND_API void ggml_backend_sycl_unregister_host_buffer(void * buffer);

GGML_BACKEND_API ggml_backend_reg_t ggml_backend_sycl_reg(void);

#ifdef  __cplusplus  // 如果定义了 __cplusplus 则编译
}
#endif  // 条件编译结束
