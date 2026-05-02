// helper functions for ggml-metal that are too difficult to implement in Objective-C

#pragma once  // 防止重复包含

#include <stdbool.h>  // 引入 stdbool.h 头文件

#ifdef __cplusplus  // 如果定义了 __cplusplus 则编译
extern "C" {  // C 链接声明
#endif  // 条件编译结束

struct ggml_tensor;
struct ggml_cgraph;

enum ggml_mem_range_type {  // 枚举定义
    MEM_RANGE_TYPE_SRC = 0,
    MEM_RANGE_TYPE_DST = 1,
};

// a helper object that can be used for reordering operations to improve concurrency
//
// the fundamental idea is that a set of tasks (either ggml ops, or something else) can run concurrently if they
//   don't write to a memory that is being read by another task or written to by another task in the set
//
// with this structure, we can add tasks to the set, setting memory constraints. we can also check if a new task
//   can be added to the set without violating the constraints (i.e. if it can be executed concurrently with the
//   tasks already in the set)
//
typedef struct ggml_mem_ranges * ggml_mem_ranges_t;  // 类型定义

ggml_mem_ranges_t ggml_mem_ranges_init(int debug);  // ggml_mem_ranges_init
void ggml_mem_ranges_free(ggml_mem_ranges_t mrs);  // ggml_mem_ranges_free

// remove all ranges from the set
void ggml_mem_ranges_reset(ggml_mem_ranges_t mrs);  // ggml_mem_ranges_reset

// add src or dst ranges to track
bool ggml_mem_ranges_add(ggml_mem_ranges_t mrs, const struct ggml_tensor * tensor);  // ggml_mem_ranges_add

// return false if:
// - new src range overlaps with any existing dst range
// - new dst range overlaps with any existing range (src or dst)
bool ggml_mem_ranges_check(ggml_mem_ranges_t mrs, const struct ggml_tensor * tensor);  // ggml_mem_ranges_check

// reorder the nodes in the graph to improve concurrency, while respecting fusion
//
// note: this implementation is generic and not specific to metal
//       if it proves to work well, we can start using it for other backends in the future
void ggml_graph_optimize(struct ggml_cgraph * gf);  // ggml_graph_optimize

#ifdef __cplusplus  // 如果定义了 __cplusplus 则编译
}
#endif  // 条件编译结束
