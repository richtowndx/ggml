#pragma once  // 防止重复包含
#include "common.h"  // 引入 common.h 头文件

size_t ggml_backend_amx_desired_wsize(const struct ggml_tensor * dst);  // ggml_backend_amx_desired_wsize

size_t ggml_backend_amx_get_alloc_size(const struct ggml_tensor * tensor);  // ggml_backend_amx_get_alloc_size

void ggml_backend_amx_convert_weight(struct ggml_tensor * tensor, const void * data, size_t offset, size_t size);  // ggml_backend_amx_convert_weight

void ggml_backend_amx_mul_mat(const struct ggml_compute_params * params, struct ggml_tensor * dst);  // ggml_backend_amx_mul_mat
