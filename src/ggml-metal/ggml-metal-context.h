#pragma once  // 防止重复包含

#include "ggml-metal-device.h"  // 引入 ggml-metal-device.h 头文件

#ifdef __cplusplus  // 如果定义了 __cplusplus 则编译
extern "C" {  // C 链接声明
#endif  // 条件编译结束

//
// backend context
//

typedef struct ggml_metal * ggml_metal_t;  // 类型定义

ggml_metal_t ggml_metal_init(ggml_metal_device_t dev);  // ggml_metal_init
void ggml_metal_free(ggml_metal_t ctx);  // ggml_metal_free

const char * ggml_metal_get_name(ggml_metal_t ctx);  // ggml_metal_get_name

void ggml_metal_synchronize(ggml_metal_t ctx);  // ggml_metal_synchronize

void ggml_metal_set_tensor_async(ggml_metal_t ctx, struct ggml_tensor * tensor, const void * data, size_t offset, size_t size);  // ggml_metal_set_tensor_async
void ggml_metal_get_tensor_async(ggml_metal_t ctx, const struct ggml_tensor * tensor, void * data, size_t offset, size_t size);  // ggml_metal_get_tensor_async
bool ggml_metal_cpy_tensor_async(ggml_metal_t ctx_src, ggml_metal_t ctx_dst, const struct ggml_tensor * src, struct ggml_tensor * dst);  // ggml_metal_cpy_tensor_async

enum ggml_status ggml_metal_graph_compute (ggml_metal_t ctx, struct ggml_cgraph * gf);  // ggml_metal_graph_compute
void             ggml_metal_graph_optimize(ggml_metal_t ctx, struct ggml_cgraph * gf);  // ggml_metal_graph_optimize

void ggml_metal_event_record(ggml_metal_t ctx, ggml_metal_event_t ev);  // ggml_metal_event_record
void ggml_metal_event_wait  (ggml_metal_t ctx, ggml_metal_event_t ev);  // ggml_metal_event_wait

ggml_metal_event_t ggml_metal_get_ev_cpy(ggml_metal_t ctx);  // ggml_metal_get_ev_cpy

void ggml_metal_set_n_cb            (ggml_metal_t ctx, int n_cb);  // ggml_metal_set_n_cb
void ggml_metal_set_abort_callback  (ggml_metal_t ctx, ggml_abort_callback abort_callback, void * user_data);  // ggml_metal_set_abort_callback
bool ggml_metal_supports_family     (ggml_metal_t ctx, int family);  // ggml_metal_supports_family
void ggml_metal_capture_next_compute(ggml_metal_t ctx);  // ggml_metal_capture_next_compute

#ifdef __cplusplus  // 如果定义了 __cplusplus 则编译
}
#endif  // 条件编译结束
