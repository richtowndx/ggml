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

#ifndef GGML_SYCL_BACKEND_HPP  // 如果未定义 GGML_SYCL_BACKEND_HPP 则编译
#define GGML_SYCL_BACKEND_HPP  // 宏定义 GGML_SYCL_BACKEND_HPP

#include "binbcast.hpp"  // 引入 binbcast.hpp 头文件
#include "common.hpp"  // 引入 common.hpp 头文件
#include "concat.hpp"  // 引入 concat.hpp 头文件
#include "conv.hpp"  // 引入 conv.hpp 头文件
#include "convert.hpp"  // 引入 convert.hpp 头文件
#include "count-equal.hpp"  // 引入 count-equal.hpp 头文件
#include "cpy.hpp"  // 引入 cpy.hpp 头文件
#include "dequantize.hpp"  // 引入 dequantize.hpp 头文件
#include "dmmv.hpp"  // 引入 dmmv.hpp 头文件
#include "element_wise.hpp"  // 引入 element_wise.hpp 头文件
#include "fattn.hpp"  // 引入 fattn.hpp 头文件
#include "gated_delta_net.hpp"  // 引入 gated_delta_net.hpp 头文件
#include "gla.hpp"  // 引入 gla.hpp 头文件
#include "im2col.hpp"  // 引入 im2col.hpp 头文件
#include "mmq.hpp"  // 引入 mmq.hpp 头文件
#include "mmvq.hpp"  // 引入 mmvq.hpp 头文件
#include "norm.hpp"  // 引入 norm.hpp 头文件
#include "outprod.hpp"  // 引入 outprod.hpp 头文件
#include "pad.hpp"  // 引入 pad.hpp 头文件
#include "pad_reflect_1d.hpp"  // 引入 pad_reflect_1d.hpp 头文件
#include "quantize.hpp"  // 引入 quantize.hpp 头文件
#include "quants.hpp"  // 引入 quants.hpp 头文件
#include "roll.hpp"  // 引入 roll.hpp 头文件
#include "rope.hpp"  // 引入 rope.hpp 头文件
#include "set_rows.hpp"  // 引入 set_rows.hpp 头文件
#include "ssm_conv.hpp"  // 引入 ssm_conv.hpp 头文件
#include "softmax.hpp"  // 引入 softmax.hpp 头文件
#include "tsembd.hpp"  // 引入 tsembd.hpp 头文件
#include "upscale.hpp"  // 引入 upscale.hpp 头文件
#include "wkv.hpp"  // 引入 wkv.hpp 头文件


#endif  // GGML_SYCL_BACKEND_HPP  // 条件编译结束
