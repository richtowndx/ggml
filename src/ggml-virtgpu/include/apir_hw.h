#pragma once  // 防止重复包含

#include <stdint.h>  // 引入 stdint.h 头文件

struct virgl_renderer_capset_apir {  // 结构体定义
    uint32_t apir_version;
    uint32_t supports_blob_resources;
    uint32_t reserved[4];  // For future expansion
};
