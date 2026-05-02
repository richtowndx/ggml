#ifndef GGML_ZDNN_UTILITIES_HPP  // 如果未定义 GGML_ZDNN_UTILITIES_HPP 则编译
#define GGML_ZDNN_UTILITIES_HPP  // 宏定义 GGML_ZDNN_UTILITIES_HPP

#include "common.hpp"  // 引入 common.hpp 头文件

zdnn_data_types ggml_zdnn_type_mapping(ggml_type type);  // ggml_zdnn_type_mapping

void ggml_zdnn_create_tensor(zdnn_tensor_desc & pre_tfm_desc,
                             zdnn_tensor_desc & tfm_desc,
                             zdnn_ztensor     & ztensor,
                      const ggml_tensor       * src,
                      const int64_t           * ne,
                      const zdnn_data_layouts   layout);

void ggml_zdnn_load_tensor(zdnn_ztensor & ztensor, void * buffer);  // ggml_zdnn_load_tensor

void ggml_zdnn_init_tensor(ggml_backend_zdnn_buffer * buffer, const ggml_tensor * tensor);  // ggml_zdnn_init_tensor

#endif  // GGML_ZDNN_UTILITIES_HPP  // 条件编译结束
