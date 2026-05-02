#ifndef GGML_ZDNN_COMMON_HPP  // 如果未定义 GGML_ZDNN_COMMON_HPP 则编译
#define GGML_ZDNN_COMMON_HPP  // 宏定义 GGML_ZDNN_COMMON_HPP

#include "ggml.h"  // 引入 ggml.h 头文件
#include "ggml-impl.h"  // 引入 ggml-impl.h 头文件

#include "zdnn.h"  // 引入 zdnn.h 头文件

#include <vector>  // 引入 vector 头文件
#include <memory>  // 引入 memory 头文件

#define GGML_ZDNN_NAME    "zDNN"  // 宏定义 GGML_ZDNN_NAME
#define GGML_ZDNN_VERSION ZDNN_VERNUM  // 宏定义 GGML_ZDNN_VERSION

#define ZDNN_CHECK(stmt)                \
    do {                                \
        zdnn_status status = (stmt);    \
        GGML_ASSERT(status == ZDNN_OK); \
    } while (0);

struct ggml_backend_zdnn_device_context {  // 结构体定义
    int zdnn_device;
    int zdnn_device_ref_count;

    bool has_parmblkformat_0;
    bool has_parmblkformat_1;  // checks for z17

    size_t max_size;

    char name[128];
};

struct ggml_backend_zdnn_context {  // 结构体定义
    int device;
    ggml_cgraph * gf;
};

struct ggml_backend_zdnn_buffer {  // 结构体定义
    void * data;
    ggml_backend_zdnn_buffer * extra;  // for bias, etc.
    size_t size;

    zdnn_tensor_desc pre_tfm_desc;
    zdnn_tensor_desc tfm_desc;
    zdnn_ztensor     ztensor;

    char name[GGML_MAX_NAME];
};

struct ggml_backend_zdnn_buffer_context {  // 结构体定义
    void * all_data;
    size_t all_size;
    bool owned;

    int n_buffers;
    std::vector<std::unique_ptr<ggml_backend_zdnn_buffer>> buffers;
};

#endif  // GGML_ZDNN_COMMON_HPP  // 条件编译结束
