#ifndef GGML_ZDNN_MMF_HPP  // 如果未定义 GGML_ZDNN_MMF_HPP 则编译
#define GGML_ZDNN_MMF_HPP  // 宏定义 GGML_ZDNN_MMF_HPP

#include "common.hpp"  // 引入 common.hpp 头文件

void ggml_zdnn_mul_mat_f(
    const ggml_backend_zdnn_context * ctx,
    const               ggml_tensor * src0,
    const               ggml_tensor * src1,
                        ggml_tensor * dst);

#endif  // GGML_ZDNN_MMF_HPP  // 条件编译结束
