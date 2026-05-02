#pragma once  // 防止重复包含

// clang-format off
#include "virtgpu-utils.h"  // 引入 virtgpu-utils.h 头文件
#include "virtgpu-shm.h"  // 引入 virtgpu-shm.h 头文件
#include "virtgpu-apir.h"  // 引入 virtgpu-apir.h 头文件

#include "backend/shared/api_remoting.h"  // 引入 backend/shared/api_remoting.h 头文件
#include "backend/shared/apir_cs.h"  // 引入 backend/shared/apir_cs.h 头文件

#include <fcntl.h>  // 引入 fcntl.h 头文件
#include <stdbool.h>  // 引入 stdbool.h 头文件
#include <stdio.h>  // 引入 stdio.h 头文件
#include <sys/stat.h>  // 引入 sys/stat.h 头文件
#include <sys/sysmacros.h>  // 引入 sys/sysmacros.h 头文件
#include <threads.h>  // 引入 threads.h 头文件
#include <xf86drm.h>  // 引入 xf86drm.h 头文件

#include <cstring>  // 引入 cstring 头文件

#include "ggml-remoting.h"  // 引入 ggml-remoting.h 头文件

#define VIRGL_RENDERER_UNSTABLE_APIS 1  // 宏定义 VIRGL_RENDERER_UNSTABLE_APIS
#include "apir_hw.h"  // 引入 apir_hw.h 头文件
#include <drm/virtgpu_drm.h>  // 引入 drm/virtgpu_drm.h 头文件
#include "venus_hw.h"  // 引入 venus_hw.h 头文件
// clang-format on

#ifndef VIRTGPU_DRM_CAPSET_APIR  // 如果未定义 VIRTGPU_DRM_CAPSET_APIR 则编译
// Will be defined include/drm/virtgpu_drm.h when
// https://gitlab.freedesktop.org/virgl/virglrenderer/-/merge_requests/1590/diffs
// is merged
#    define VIRTGPU_DRM_CAPSET_APIR 10  // 宏定义 VIRTGPU_DRM_CAPSET_APIR
#endif  // 条件编译结束

// Mesa/Virlgrenderer Venus internal. Only necessary during the
// Venus->APIR transition in Virglrenderer
#define VENUS_COMMAND_TYPE_LENGTH 331  // 宏定义 VENUS_COMMAND_TYPE_LENGTH

#ifndef VIRTGPU_DRM_CAPSET_VENUS  // only available with Linux >= v6.16  // 如果未定义 VIRTGPU_DRM_CAPSET_VENUS 则编译
#    define VIRTGPU_DRM_CAPSET_VENUS 4  // 宏定义 VIRTGPU_DRM_CAPSET_VENUS
#endif  // 条件编译结束

typedef uint32_t virgl_renderer_capset;  // 类型定义

/* from src/virtio/vulkan/vn_renderer_virtgpu.c */
#define VIRTGPU_PCI_VENDOR_ID       0x1af4  // 宏定义 VIRTGPU_PCI_VENDOR_ID
#define VIRTGPU_PCI_DEVICE_ID       0x1050  // 宏定义 VIRTGPU_PCI_DEVICE_ID
#define VIRTGPU_BLOB_MEM_GUEST_VRAM 0x0004  // 宏定义 VIRTGPU_BLOB_MEM_GUEST_VRAM
#define VIRTGPU_PARAM_GUEST_VRAM    9  // 宏定义 VIRTGPU_PARAM_GUEST_VRAM

#define SHMEM_DATA_SIZE  0x1830000  // 24MiB  // 宏定义 SHMEM_DATA_SIZE
#define SHMEM_REPLY_SIZE 0x4000  // 宏定义 SHMEM_REPLY_SIZE

#define ARRAY_SIZE(x) (sizeof(x) / sizeof((x)[0]))  // 宏定义 ARRAY_SIZE

enum virt_gpu_result_t {  // 枚举定义
    APIR_SUCCESS                     = 0,
    APIR_ERROR_INITIALIZATION_FAILED = -1,
};

#define PRINTFLIKE(f, a) __attribute__((format(__printf__, f, a)))  // 宏定义 PRINTFLIKE

struct virtgpu {  // 结构体定义
    bool use_apir_capset;

    int fd;

    struct {
        virgl_renderer_capset      id;
        uint32_t                   version;
        virgl_renderer_capset_apir data;
    } capset;

    util_sparse_array shmem_array;

    /* APIR communication pages */
    virtgpu_shmem reply_shmem;
    virtgpu_shmem data_shmem;

    /* Mutex to protect shared data_shmem buffer from concurrent access */
    mtx_t data_shmem_mutex;

    /* Cached device information to prevent memory leaks and race conditions */
    struct {
        char *   description;
        char *   name;
        int32_t  device_count;
        uint32_t type;
        size_t   memory_free;
        size_t   memory_total;
    } cached_device_info;

    /* Cached buffer type information to prevent memory leaks and race conditions */
    struct {
        apir_buffer_type_host_handle_t host_handle;
        char *                         name;
        size_t                         alignment;
        size_t                         max_size;
    } cached_buffer_type;
};

static inline int virtgpu_ioctl(virtgpu * gpu, unsigned long request, void * args) {
    return drmIoctl(gpu->fd, request, args);  // drmIoctl
}

virtgpu * create_virtgpu();  // create_virtgpu

apir_encoder * remote_call_prepare(virtgpu * gpu, ApirCommandType apir_cmd_type, int32_t cmd_flags);  // remote_call_prepare

uint32_t remote_call(virtgpu *       gpu,
                     apir_encoder *  enc,
                     apir_decoder ** dec,
                     float           max_wait_ms,
                     long long *     call_duration_ns);

void remote_call_finish(virtgpu * gpu, apir_encoder * enc, apir_decoder * dec);  // remote_call_finish
