#pragma once  // 防止重复包含

#ifndef __cplusplus  // 如果未定义 __cplusplus 则编译
#error "This header is for C++ only"
#endif  // 条件编译结束

#include "ggml.h"  // 引入 ggml.h 头文件
#include "ggml-alloc.h"  // 引入 ggml-alloc.h 头文件
#include "ggml-backend.h"  // 引入 ggml-backend.h 头文件
#include "gguf.h"  // 引入 gguf.h 头文件
#include <memory>  // 引入 memory 头文件

// Smart pointers for ggml types

// ggml

struct ggml_context_deleter { void operator()(ggml_context * ctx) { ggml_free(ctx); } };  // 结构体定义
struct gguf_context_deleter { void operator()(gguf_context * ctx) { gguf_free(ctx); } };  // 结构体定义

typedef std::unique_ptr<ggml_context, ggml_context_deleter> ggml_context_ptr;  // 类型定义
typedef std::unique_ptr<gguf_context, gguf_context_deleter> gguf_context_ptr;  // 类型定义

// ggml-alloc

struct ggml_gallocr_deleter { void operator()(ggml_gallocr_t galloc) { ggml_gallocr_free(galloc); } };  // 结构体定义

typedef std::unique_ptr<ggml_gallocr, ggml_gallocr_deleter> ggml_gallocr_ptr;  // 类型定义

// ggml-backend

struct ggml_backend_deleter        { void operator()(ggml_backend_t backend)       { ggml_backend_free(backend); } };  // 结构体定义
struct ggml_backend_buffer_deleter { void operator()(ggml_backend_buffer_t buffer) { ggml_backend_buffer_free(buffer); } };  // 结构体定义
struct ggml_backend_event_deleter  { void operator()(ggml_backend_event_t event)   { ggml_backend_event_free(event); } };  // 结构体定义
struct ggml_backend_sched_deleter  { void operator()(ggml_backend_sched_t sched)   { ggml_backend_sched_free(sched); } };  // 结构体定义

typedef std::unique_ptr<ggml_backend,        ggml_backend_deleter>        ggml_backend_ptr;  // 类型定义
typedef std::unique_ptr<ggml_backend_buffer, ggml_backend_buffer_deleter> ggml_backend_buffer_ptr;  // 类型定义
typedef std::unique_ptr<ggml_backend_event,  ggml_backend_event_deleter>  ggml_backend_event_ptr;  // 类型定义
typedef std::unique_ptr<ggml_backend_sched,  ggml_backend_sched_deleter>  ggml_backend_sched_ptr;  // 类型定义
