#pragma once  // 防止重复包含

#include "ggml-backend.h"  // 引入 ggml-backend.h 头文件

#ifdef  __cplusplus  // 如果定义了 __cplusplus 则编译
extern "C" {  // C 链接声明
#endif  // 条件编译结束

#define RPC_PROTO_MAJOR_VERSION    4  // 宏定义 RPC_PROTO_MAJOR_VERSION
#define RPC_PROTO_MINOR_VERSION    0  // 宏定义 RPC_PROTO_MINOR_VERSION
#define RPC_PROTO_PATCH_VERSION    0  // 宏定义 RPC_PROTO_PATCH_VERSION

#ifdef  __cplusplus  // 如果定义了 __cplusplus 则编译
static_assert(GGML_OP_COUNT == 96, "GGML_OP_COUNT has changed - update RPC_PROTO_PATCH_VERSION");
#endif  // 条件编译结束

#define GGML_RPC_MAX_SERVERS       16  // 宏定义 GGML_RPC_MAX_SERVERS

// backend API
GGML_BACKEND_API ggml_backend_t ggml_backend_rpc_init(const char * endpoint, uint32_t device);
GGML_BACKEND_API bool ggml_backend_is_rpc(ggml_backend_t backend);

GGML_BACKEND_API ggml_backend_buffer_type_t ggml_backend_rpc_buffer_type(const char * endpoint, uint32_t device);

GGML_BACKEND_API void ggml_backend_rpc_get_device_memory(const char * endpoint, uint32_t device, size_t * free, size_t * total);

GGML_BACKEND_API void ggml_backend_rpc_start_server(const char * endpoint, const char * cache_dir,
                                                    size_t n_threads, size_t n_devices, ggml_backend_dev_t * devices);

GGML_BACKEND_API ggml_backend_reg_t ggml_backend_rpc_reg(void);
GGML_BACKEND_API ggml_backend_reg_t ggml_backend_rpc_add_server(const char * endpoint);

#ifdef  __cplusplus  // 如果定义了 __cplusplus 则编译
}
#endif  // 条件编译结束
