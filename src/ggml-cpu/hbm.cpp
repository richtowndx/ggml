#ifdef GGML_USE_CPU_HBM  // 如果定义了 GGML_USE_CPU_HBM 则编译

#include "ggml-backend.h"  // 引入 ggml-backend.h 头文件
#include "ggml-backend-impl.h"  // 引入 ggml-backend-impl.h 头文件
#include "ggml-cpu.h"  // 引入 ggml-cpu.h 头文件
#include "ggml-impl.h"  // 引入 ggml-impl.h 头文件

#include "hbm.h"  // 引入 hbm.h 头文件

// buffer type HBM

#include <hbwmalloc.h>  // 引入 hbwmalloc.h 头文件

static const char * ggml_backend_cpu_hbm_buffer_type_get_name(ggml_backend_buffer_type_t buft) {
    return "CPU_HBM";  // 返回

    GGML_UNUSED(buft);
}

static void ggml_backend_cpu_hbm_buffer_free_buffer(ggml_backend_buffer_t buffer) {
    hbw_free(buffer->context);
}

static ggml_backend_buffer_t ggml_backend_cpu_hbm_buffer_type_alloc_buffer(ggml_backend_buffer_type_t buft,
                                                                           size_t                     size) {
    void * ptr;
    int    result = hbw_posix_memalign(&ptr, ggml_backend_cpu_buffer_type_get_alignment(buft), size);
    if (result != 0) {
        GGML_LOG_ERROR("failed to allocate HBM buffer of size %zu\n", size);
        return NULL;  // 返回
    }

    ggml_backend_buffer_t buffer = ggml_backend_cpu_buffer_from_ptr(ptr, size);
    buffer->buft                 = buft;
    buffer->iface.free_buffer    = ggml_backend_cpu_hbm_buffer_free_buffer;

    return buffer;  // 返回
}

ggml_backend_buffer_type_t ggml_backend_cpu_hbm_buffer_type(void) {
    static struct ggml_backend_buffer_type ggml_backend_cpu_buffer_type_hbm = {
        /* .iface    = */ {
                           /* .get_name         = */ ggml_backend_cpu_hbm_buffer_type_get_name,
                           /* .alloc_buffer     = */ ggml_backend_cpu_hbm_buffer_type_alloc_buffer,
                           /* .get_alignment    = */ ggml_backend_cpu_buffer_type_get_alignment,
                           /* .get_max_size     = */ nullptr,  // defaults to SIZE_MAX
                           /* .get_alloc_size   = */ nullptr,  // defaults to ggml_nbytes
                           /* .is_host          = */ ggml_backend_cpu_buffer_type_is_host,
                           },
        /* .context  = */ nullptr,
    };

    return &ggml_backend_cpu_buffer_type_hbm;  // 返回
}
#endif  // 条件编译结束
