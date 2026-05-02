#pragma once  // 防止重复包含
#include "ggml-backend-impl.h"  // 引入 ggml-backend-impl.h 头文件
#include "ggml-cpu-impl.h"  // 引入 ggml-cpu-impl.h 头文件
#include "ggml.h"  // 引入 ggml.h 头文件

#ifdef __cplusplus  // 如果定义了 __cplusplus 则编译
#    include <vector>  // 引入 vector 头文件
extern "C" {  // C 链接声明
#endif  // 条件编译结束

// return true if op part of extra "accelerator"
bool ggml_cpu_extra_compute_forward(struct ggml_compute_params * params, struct ggml_tensor * op);  // ggml_cpu_extra_compute_forward
bool ggml_cpu_extra_work_size(int n_threads, const struct ggml_tensor * op, size_t * size);  // ggml_cpu_extra_work_size

#ifdef __cplusplus  // 如果定义了 __cplusplus 则编译
}

namespace ggml::cpu {  // 命名空间
// register in tensor->extra
class tensor_traits {  // 类定义
  public:
    virtual ~tensor_traits();
    virtual bool work_size(int n_threads, const struct ggml_tensor * op, size_t & size)        = 0;
    virtual bool compute_forward(struct ggml_compute_params * params, struct ggml_tensor * op) = 0;
};

class extra_buffer_type {  // 类定义
  public:
    virtual ~extra_buffer_type();
    virtual bool            supports_op(ggml_backend_dev_t dev, const struct ggml_tensor * op) = 0;
    virtual tensor_traits * get_tensor_traits(const struct ggml_tensor * op)                   = 0;
};
}  // namespace ggml::cpu

// implemented in ggml-cpu.cpp.
std::vector<ggml_backend_buffer_type_t> & ggml_backend_cpu_get_extra_buffer_types();

#endif  // 条件编译结束
