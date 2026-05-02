#include "backend/shared/apir_backend.h"  // 引入 backend/shared/apir_backend.h 头文件
#include "ggml-alloc.h"  // 引入 ggml-alloc.h 头文件
#include "ggml-impl.h"  // 引入 ggml-impl.h 头文件
#include "ggml.h"  // 引入 ggml.h 头文件
#include "virtgpu-shm.h"  // 引入 virtgpu-shm.h 头文件
#include "virtgpu-utils.h"  // 引入 virtgpu-utils.h 头文件

struct apir_buffer_context_t {  // 结构体定义
    apir_buffer_host_handle_t host_handle;

    struct virtgpu_shmem           shmem;
    apir_buffer_type_host_handle_t buft_host_handle;
};

#include "virtgpu-forward.gen.h"  // 引入 virtgpu-forward.gen.h 头文件
