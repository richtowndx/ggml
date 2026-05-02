//
// MIT license
// Copyright (C) 2025 Intel Corporation
// SPDX-License-Identifier: MIT
//

//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//

#ifndef GGML_SYCL_CONVERT_HPP  // 如果未定义 GGML_SYCL_CONVERT_HPP 则编译
#define GGML_SYCL_CONVERT_HPP  // 宏定义 GGML_SYCL_CONVERT_HPP

#include "common.hpp"  // 引入 common.hpp 头文件

template <typename T>  // 模板
using to_t_sycl_t = void (*)(const void * __restrict__ x, T * __restrict__ y, int64_t k, dpct::queue_ptr stream);
typedef to_t_sycl_t<float>      to_fp32_sycl_t;  // 类型定义
typedef to_t_sycl_t<sycl::half> to_fp16_sycl_t;  // 类型定义

to_fp16_sycl_t ggml_get_to_fp16_sycl(ggml_type type, ggml_tensor * dst);  // ggml_get_to_fp16_sycl
to_fp32_sycl_t ggml_get_to_fp32_sycl(ggml_type type, ggml_tensor * dst);  // ggml_get_to_fp32_sycl

#ifdef GGML_SYCL_HAS_BF16  // 如果定义了 GGML_SYCL_HAS_BF16 则编译
typedef to_t_sycl_t<sycl::ext::oneapi::bfloat16> to_bf16_sycl_t;  // 类型定义
to_bf16_sycl_t ggml_get_to_bf16_sycl(ggml_type type, ggml_tensor * dst);  // ggml_get_to_bf16_sycl
#endif  // 条件编译结束

// Nc = Non-contiguous
template <typename T>  // 模板
using to_t_nc_sycl_t = void (*)(const void * x, T * y, int64_t ne00, int64_t ne01, int64_t ne02, int64_t ne03,
                                   int64_t s01, int64_t s02, int64_t s03, dpct::queue_ptr queue);

typedef to_t_nc_sycl_t<sycl::half> to_fp16_nc_sycl_t;  // 类型定义
to_fp16_nc_sycl_t ggml_get_to_fp16_nc_sycl(ggml_type type);  // ggml_get_to_fp16_nc_sycl

template<typename dst_t, typename src_t>  // 模板
 inline dst_t ggml_sycl_cast(src_t x) {
    if constexpr (std::is_same_v<dst_t, src_t>) {
        return x;  // 返回
#ifdef GGML_SYCL_HAS_BF16  // 如果定义了 GGML_SYCL_HAS_BF16 则编译
    } else if constexpr (std::is_same_v<dst_t, sycl::ext::oneapi::bfloat16>) {
        return sycl::ext::oneapi::bfloat16(float(x));
    } else if constexpr (std::is_same_v<src_t, sycl::ext::oneapi::bfloat16>) {
        return static_cast<float>(x);
#endif  // 条件编译结束
    } else if constexpr (std::is_same_v<src_t, sycl::float2> && std::is_same_v<dst_t, sycl::half2>) {
        return x.template convert<sycl::half, sycl::rounding_mode::rte>();
#ifdef GGML_SYCL_HAS_BF16  // 如果定义了 GGML_SYCL_HAS_BF16 则编译
    } else if constexpr (std::is_same_v<src_t, sycl::float2> &&
                         std::is_same_v<dst_t, sycl::vec<sycl::ext::oneapi::bfloat16, 2>>) {
        return {x.x, x.y};  // 返回
#endif  // 条件编译结束
    } else if constexpr(std::is_same_v<dst_t, int32_t>) {
        return int32_t(x);  // int32_t
    } else {
        return float(x);  // float
    }
}


#endif  // GGML_SYCL_CONVERT_HPP  // 条件编译结束
