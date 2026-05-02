#pragma once  // 防止重复包含

#include "ggml-backend-impl.h"  // 引入 ggml-backend-impl.h 头文件
#include "ggml-backend.h"  // 引入 ggml-backend.h 头文件
#include "ggml-impl.h"  // 引入 ggml-impl.h 头文件
#include "virtgpu.h"  // 引入 virtgpu.h 头文件

#include <memory>  // 引入 memory 头文件
#include <string>  // 引入 string 头文件

#define GGML_VIRTGPU_NAME "ggml-virtgpu"  // 宏定义 GGML_VIRTGPU_NAME
#define GGML_VIRTGPU      "ggml-virtgpu: "  // 宏定义 GGML_VIRTGPU

// USE_ALWAYS_TRUE_SUPPORTS_OP: 1 is fast, 0 avoid micro-benchmark crashes

#define USE_ALWAYS_TRUE_SUPPORTS_OP 1  // 宏定义 USE_ALWAYS_TRUE_SUPPORTS_OP
#define USE_METAL_GUEST_SUPPORTS_OP 0  // 宏定义 USE_METAL_GUEST_SUPPORTS_OP

#define DEV_TO_GPU(name) ((ggml_backend_remoting_device_context *) (name)->context)->gpu  // 宏定义 DEV_TO_GPU

#define BUFFER_TO_GGML_CONTEXT(name) ((ggml_backend_remoting_buffer_context *) (name)->context)  // 宏定义 BUFFER_TO_GGML_CONTEXT

#define BUFFER_TO_APIR_CONTEXT(name) &((ggml_backend_remoting_buffer_context *) (name)->context)->apir_context  // 宏定义 BUFFER_TO_APIR_CONTEXT

#define BUFFER_TO_HOST_HANDLE(name) ((ggml_backend_remoting_buffer_context *) (name)->context)->apir_context.host_handle  // 宏定义 BUFFER_TO_HOST_HANDLE

#define GET_DEVICE_CONTEXT() (ggml_backend_remoting_device_context *) ggml_backend_remoting_get_device(0)->context  // 宏定义 GET_DEVICE_CONTEXT

#define BUFT_TO_GPU(name) ((ggml_backend_remoting_device_context *) (name)->device->context)->gpu  // 宏定义 BUFT_TO_GPU

struct ggml_backend_remoting_device_context {  // 结构体定义
    size_t      device;
    std::string name;
    std::string description;

    std::vector<std::tuple<void *, size_t, virtgpu_shmem *>> shared_memory;

    virtgpu * gpu;
};

struct ggml_backend_remoting_buffer_context {  // 结构体定义
    apir_buffer_context_t apir_context;

    virtgpu * gpu;

    void * base;

    bool is_from_ptr;
};

extern const ggml_backend_buffer_type_i ggml_backend_remoting_buffer_type_interface;
extern const ggml_backend_device_i      ggml_backend_remoting_device_interface;
extern const ggml_backend_buffer_i      ggml_backend_remoting_buffer_interface;
extern const ggml_backend_buffer_type_i ggml_backend_remoting_buffer_from_ptr_type_interface;
extern const ggml_backend_buffer_i      ggml_backend_remoting_buffer_from_ptr_interface;

ggml_backend_dev_t         ggml_backend_remoting_get_device(size_t device);  // ggml_backend_remoting_get_device
ggml_backend_t             ggml_backend_remoting_device_init(ggml_backend_dev_t dev, const char * params);  // ggml_backend_remoting_device_init
ggml_backend_buffer_type_t ggml_backend_remoting_device_get_buffer_type(ggml_backend_dev_t dev);  // ggml_backend_remoting_device_get_buffer_type

static inline apir_buffer_type_host_handle_t ggml_buffer_type_to_apir_handle(ggml_backend_buffer_type_t buft) {
    // in the backend, the buffer handle is the buffer pointer
    return (apir_buffer_type_host_handle_t) buft->context;
}

static inline apir_buffer_host_handle_t ggml_buffer_to_apir_handle(ggml_backend_buffer_t buffer) {
    if (!buffer->context) {
        GGML_ABORT(GGML_VIRTGPU "%s: no context available :/", __func__);
    }
    return BUFFER_TO_HOST_HANDLE(buffer);  // BUFFER_TO_HOST_HANDLE
}
