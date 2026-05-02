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

#ifndef GGML_SYCL_NORM_HPP  // 如果未定义 GGML_SYCL_NORM_HPP 则编译
#define GGML_SYCL_NORM_HPP  // 宏定义 GGML_SYCL_NORM_HPP

#include "common.hpp"  // 引入 common.hpp 头文件

void ggml_sycl_op_norm(ggml_backend_sycl_context& ctx, ggml_tensor* dst);  // ggml_sycl_op_norm

void ggml_sycl_op_rms_norm(ggml_backend_sycl_context& ctx, ggml_tensor* dst);  // ggml_sycl_op_rms_norm

void ggml_sycl_op_rms_norm_back(ggml_backend_sycl_context& ctx, ggml_tensor* dst);  // ggml_sycl_op_rms_norm_back

void ggml_sycl_op_group_norm(ggml_backend_sycl_context& ctx, ggml_tensor* dst);  // ggml_sycl_op_group_norm

void ggml_sycl_op_l2_norm(ggml_backend_sycl_context& ctx, ggml_tensor* dst);  // ggml_sycl_op_l2_norm

#endif // GGML_SYCL_NORM_HPP  // 条件编译结束
