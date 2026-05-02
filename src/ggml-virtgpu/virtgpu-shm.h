#pragma once  // 防止重复包含

#include "virtgpu-utils.h"  // 引入 virtgpu-utils.h 头文件

#include <sys/mman.h>  // 引入 sys/mman.h 头文件

#include <atomic>  // 引入 atomic 头文件
#include <cassert>  // 引入 cassert 头文件
#include <cstddef>  // 引入 cstddef 头文件
#include <cstdint>  // 引入 cstdint 头文件

struct virtgpu;

struct virtgpu_shmem {  // 结构体定义
    uint32_t res_id;
    size_t   mmap_size;
    void *   mmap_ptr;

    uint32_t gem_handle;
};

int  virtgpu_shmem_create(virtgpu * gpu, size_t size, virtgpu_shmem * shmem);  // virtgpu_shmem_create
void virtgpu_shmem_destroy(virtgpu * gpu, virtgpu_shmem * shmem);  // virtgpu_shmem_destroy
