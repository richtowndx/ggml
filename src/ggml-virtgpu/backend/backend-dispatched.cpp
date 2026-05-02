#include "backend-dispatched.h"  // 引入 backend-dispatched.h 头文件

#include "backend-virgl-apir.h"  // 引入 backend-virgl-apir.h 头文件
#include "ggml-backend-impl.h"  // 引入 ggml-backend-impl.h 头文件
#include "ggml-backend.h"  // 引入 ggml-backend.h 头文件
#include "ggml-impl.h"  // 引入 ggml-impl.h 头文件

#include <cstdint>  // 引入 cstdint 头文件

ggml_backend_reg_t reg = NULL;
ggml_backend_dev_t dev = NULL;
ggml_backend_t     bck = NULL;

uint64_t timer_start = 0;
uint64_t timer_total = 0;
uint64_t timer_count = 0;

uint32_t backend_dispatch_initialize(void * ggml_backend_reg_fct_p) {
    if (reg != NULL) {
        GGML_LOG_WARN(GGML_VIRTGPU_BCK "%s: already initialized\n", __func__);
        return APIR_BACKEND_INITIALIZE_ALREADY_INITED;  // 返回
    }
    ggml_backend_reg_t (*ggml_backend_reg_fct)(void) = (ggml_backend_reg_t (*)()) ggml_backend_reg_fct_p;

    reg = ggml_backend_reg_fct();
    if (reg == NULL) {
        GGML_LOG_ERROR(GGML_VIRTGPU_BCK "%s: backend registration failed\n", __func__);
        return APIR_BACKEND_INITIALIZE_BACKEND_REG_FAILED;  // 返回
    }

    size_t device_count = reg->iface.get_device_count(reg);
    if (!device_count) {
        GGML_LOG_ERROR(GGML_VIRTGPU_BCK "%s: no device found\n", __func__);
        return APIR_BACKEND_INITIALIZE_NO_DEVICE;  // 返回
    }

    dev = reg->iface.get_device(reg, 0);

    if (!dev) {
        GGML_LOG_ERROR(GGML_VIRTGPU_BCK "%s: failed to get device\n", __func__);
        return APIR_BACKEND_INITIALIZE_NO_DEVICE;  // 返回
    }

    bck = dev->iface.init_backend(dev, NULL);
    if (!bck) {
        GGML_LOG_ERROR(GGML_VIRTGPU_BCK "%s: backend initialization failed\n", __func__);
        return APIR_BACKEND_INITIALIZE_BACKEND_INIT_FAILED;  // 返回
    }

    return APIR_BACKEND_INITIALIZE_SUCCESS;  // 返回
}
