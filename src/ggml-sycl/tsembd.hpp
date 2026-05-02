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

#ifndef GGML_SYCL_TSEMBD_HPP  // 如果未定义 GGML_SYCL_TSEMBD_HPP 则编译
#define GGML_SYCL_TSEMBD_HPP  // 宏定义 GGML_SYCL_TSEMBD_HPP

#include "common.hpp"  // 引入 common.hpp 头文件

void ggml_sycl_op_timestep_embedding(ggml_backend_sycl_context & ctx, ggml_tensor * dst);  // ggml_sycl_op_timestep_embedding

#endif // GGML_SYCL_TSEMBD_HPP  // 条件编译结束
