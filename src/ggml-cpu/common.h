#pragma once  // 防止重复包含

#include "ggml.h"  // 引入 ggml.h 头文件
#include "traits.h"  // 引入 traits.h 头文件
#include "ggml-cpu-impl.h"  // 引入 ggml-cpu-impl.h 头文件
#include "ggml-impl.h"  // 引入 ggml-impl.h 头文件
#include "simd-mappings.h"  // 引入 simd-mappings.h 头文件

#define GGML_FA_TILE_Q  64  // 宏定义 GGML_FA_TILE_Q
#define GGML_FA_TILE_KV 64  // 宏定义 GGML_FA_TILE_KV

#ifdef __cplusplus  // 如果定义了 __cplusplus 则编译

#include <utility>  // 引入 utility 头文件

// convenience functions/macros for use in template calls
// note: these won't be required after the 'traits' lookup table is used.
static inline ggml_fp16_t f32_to_f16(float x) {
    return GGML_CPU_FP32_TO_FP16(x);  // GGML_CPU_FP32_TO_FP16
}

static inline float f16_to_f32(ggml_fp16_t x) {
    return GGML_CPU_FP16_TO_FP32(x);  // GGML_CPU_FP16_TO_FP32
}

static inline ggml_bf16_t f32_to_bf16(float x) {
    return GGML_FP32_TO_BF16(x);  // GGML_FP32_TO_BF16
}

static inline float bf16_to_f32(ggml_bf16_t x) {
    return GGML_BF16_TO_FP32(x);  // GGML_BF16_TO_FP32
}

static inline float i32_to_f32(int32_t x) {
    return x;  // 返回
}

static inline int32_t f32_to_i32(float x) {
    return x;  // 返回
}

static inline float f32_to_f32(float x) {
    return x;  // 返回
}

// TODO - merge this into the traits table, after using row-based conversions
template <class T>  // 模板
struct type_conversion_table;

template <>  // 模板
struct type_conversion_table<ggml_fp16_t> {  // 结构体定义
    static constexpr float (*to_f32)(ggml_fp16_t) = f16_to_f32;  // float
    static constexpr ggml_fp16_t (*from_f32)(float) = f32_to_f16;  // ggml_fp16_t
};

template <>  // 模板
struct type_conversion_table<float> {  // 结构体定义
    static constexpr float (*to_f32)(float) = f32_to_f32;  // float
    static constexpr float (*from_f32)(float) = f32_to_f32;  // float
};

template <>  // 模板
struct type_conversion_table<ggml_bf16_t> {  // 结构体定义
    static constexpr float (*to_f32)(ggml_bf16_t) = bf16_to_f32;  // float
    static constexpr ggml_bf16_t (*from_f32)(float) = f32_to_bf16;  // ggml_bf16_t
};

template <>  // 模板
struct type_conversion_table<int32_t> {  // 结构体定义
    static constexpr float (*to_f32)(int32_t) = i32_to_f32;  // float
    static constexpr int32_t (*from_f32)(float) = f32_to_i32;  // int32_t
};

static std::pair<int64_t, int64_t> get_thread_range(const struct ggml_compute_params * params, const struct ggml_tensor * src0) {
    const int64_t ith = params->ith;
    const int64_t nth = params->nth;

    const int64_t nr  = ggml_nrows(src0);

    // rows per thread
    const int64_t dr = (nr + nth - 1)/nth;

    // row range for this thread
    const int64_t ir0 = dr*ith;
    const int64_t ir1 = MIN(ir0 + dr, nr);

    return {ir0, ir1};  // 返回
}

struct ggml_fa_tile_config {  // 结构体定义
    static constexpr size_t Q  = GGML_FA_TILE_Q;
    static constexpr size_t KV = GGML_FA_TILE_KV;
};

#endif  // 条件编译结束
