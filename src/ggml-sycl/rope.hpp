//
// MIT license
// Copyright (C) 2024 Intel Corporation
// SPDX-License-Identifier: MIT
//

//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//

#ifndef GGML_SYCL_ROPE_HPP  // 如果未定义 GGML_SYCL_ROPE_HPP 则编译
#define GGML_SYCL_ROPE_HPP  // 宏定义 GGML_SYCL_ROPE_HPP

#include "common.hpp"  // 引入 common.hpp 头文件

#define SYCL_ROPE_BLOCK_SIZE 256  // 宏定义 SYCL_ROPE_BLOCK_SIZE

void ggml_sycl_rope(ggml_backend_sycl_context & ctx, ggml_tensor *dst);  // ggml_sycl_rope

void ggml_sycl_rope_back(ggml_backend_sycl_context & ctx, ggml_tensor * dst);  // ggml_sycl_rope_back

void ggml_sycl_rope_fused(ggml_backend_sycl_context & ctx, ggml_tensor * dst, ggml_tensor * set_rows);  // ggml_sycl_rope_fused

#endif // GGML_SYCL_ROPE_HPP  // 条件编译结束
