// SPDX-FileCopyrightText: Copyright 2025 Arm Limited and/or its affiliates <open-source-office@arm.com>
// SPDX-License-Identifier: MIT
//

#pragma once  // 防止重复包含

#include "ggml-alloc.h"  // 引入 ggml-alloc.h 头文件

#ifdef  __cplusplus  // 如果定义了 __cplusplus 则编译
extern "C" {  // C 链接声明
#endif  // 条件编译结束

ggml_backend_buffer_type_t ggml_backend_cpu_kleidiai_buffer_type(void);  // ggml_backend_cpu_kleidiai_buffer_type

#ifdef  __cplusplus  // 如果定义了 __cplusplus 则编译
}
#endif  // 条件编译结束
