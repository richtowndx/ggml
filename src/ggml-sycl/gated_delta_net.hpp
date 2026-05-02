#pragma once  // 防止重复包含

#include <sycl/sycl.hpp>  // 引入 sycl/sycl.hpp 头文件
#include "dpct/helper.hpp"  // 引入 dpct/helper.hpp 头文件
#include "common.hpp"  // 引入 common.hpp 头文件
#include "ggml.h"  // 引入 ggml.h 头文件

void ggml_sycl_gated_delta_net(ggml_backend_sycl_context & ctx, ggml_tensor * dst);  // ggml_sycl_gated_delta_net
