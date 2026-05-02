#pragma once  // 防止重复包含

// GGML internal header

#include "ggml.h"  // 引入 ggml.h 头文件
#include "gguf.h"  // 引入 gguf.h 头文件

#include <assert.h>  // 引入 assert.h 头文件
#include <math.h>  // 引入 math.h 头文件
#include <stdlib.h> // load `stdlib.h` before other headers to work around MinGW bug: https://sourceforge.net/p/mingw-w64/bugs/192/  // 引入 stdlib.h 头文件
#include <stdbool.h>  // 引入 stdbool.h 头文件
#include <stdint.h>  // 引入 stdint.h 头文件
#include <string.h>  // 引入 string.h 头文件

#ifdef __ARM_FEATURE_SVE  // 如果定义了 __ARM_FEATURE_SVE 则编译
#include <arm_sve.h>  // 引入 arm_sve.h 头文件
#endif // __ARM_FEATURE_SVE  // 条件编译结束

#if defined(__ARM_NEON) && !defined(__CUDACC__) && !defined(__MUSACC__)  // 条件编译
// if YCM cannot find <arm_neon.h>, make a symbolic link to it, for example:
//
//   $ ln -sfn /Library/Developer/CommandLineTools/usr/lib/clang/13.1.6/include/arm_neon.h ./src/
//
#include <arm_neon.h>  // 引入 arm_neon.h 头文件
#endif  // 条件编译结束

#ifdef __cplusplus  // 如果定义了 __cplusplus 则编译
extern "C" {  // C 链接声明
#endif  // 条件编译结束

void ggml_print_backtrace(void);  // ggml_print_backtrace

uint64_t ggml_graph_next_uid(void);  // ggml_graph_next_uid

#ifndef MIN  // 如果未定义 MIN 则编译
#    define MIN(a, b) ((a) < (b) ? (a) : (b))  // 宏定义 MIN
#endif  // 条件编译结束

#ifndef MAX  // 如果未定义 MAX 则编译
#    define MAX(a, b) ((a) > (b) ? (a) : (b))  // 宏定义 MAX
#endif  // 条件编译结束

// required for mmap as gguf only guarantees 32-byte alignment
#define TENSOR_ALIGNMENT 32  // 宏定义 TENSOR_ALIGNMENT

// static_assert should be a #define, but if it's not,
// fall back to the _Static_assert C11 keyword.
// if C99 - static_assert is noop
// ref: https://stackoverflow.com/a/53923785/4039976
#ifndef __cplusplus  // 如果未定义 __cplusplus 则编译
    #ifndef static_assert  // 如果未定义 static_assert 则编译
        #if defined(__STDC_VERSION__) && (__STDC_VERSION__ >= 201100L)  // 条件编译
            #define static_assert(cond, msg) _Static_assert(cond, msg)  // 宏定义 static_assert
        #else  // 否则
            #define static_assert(cond, msg) struct global_scope_noop_trick  // 宏定义 static_assert
        #endif  // 条件编译结束
    #endif  // 条件编译结束
#endif  // 条件编译结束

static inline int ggml_up32(int n) {
    return (n + 31) & ~31;
}

//static inline int ggml_up64(int n) {
//    return (n + 63) & ~63;
//}

static inline int ggml_up(int n, int m) {
    // assert m is a power of 2
    GGML_ASSERT((m & (m - 1)) == 0);
    return (n + m - 1) & ~(m - 1);
}

// TODO: move to ggml.h? (won't be able to inline)
static bool ggml_are_same_layout(const struct ggml_tensor * a, const struct ggml_tensor * b) {
    if (a->type != b->type) {
        return false;  // 返回
    }
    for (int i = 0; i < GGML_MAX_DIMS; i++) {
        if (a->ne[i] != b->ne[i]) {
            return false;  // 返回
        }
        if (a->nb[i] != b->nb[i]) {
            return false;  // 返回
        }
    }
    return true;  // 返回
}

static bool ggml_op_is_empty(enum ggml_op op) {
    switch (op) {
        case GGML_OP_NONE:
        case GGML_OP_RESHAPE:
        case GGML_OP_TRANSPOSE:
        case GGML_OP_VIEW:
        case GGML_OP_PERMUTE:
            return true;  // 返回
        default:
            return false;  // 返回
    }
}

static inline bool ggml_impl_is_view(const struct ggml_tensor * t) {
    return t->view_src != NULL;  // 返回
}

static inline float ggml_compute_softplus_f32(float input) {
    return (input > 20.0f) ? input : logf(1 + expf(input));
}
//
// logging
//

GGML_ATTRIBUTE_FORMAT(2, 3)
GGML_API void ggml_log_internal        (enum ggml_log_level level, const char * format, ...);
GGML_API void ggml_log_callback_default(enum ggml_log_level level, const char * text, void * user_data);

#define GGML_LOG(...)       ggml_log_internal(GGML_LOG_LEVEL_NONE , __VA_ARGS__)  // 宏定义 GGML_LOG
#define GGML_LOG_INFO(...)  ggml_log_internal(GGML_LOG_LEVEL_INFO , __VA_ARGS__)  // 宏定义 GGML_LOG_INFO
#define GGML_LOG_WARN(...)  ggml_log_internal(GGML_LOG_LEVEL_WARN , __VA_ARGS__)  // 宏定义 GGML_LOG_WARN
#define GGML_LOG_ERROR(...) ggml_log_internal(GGML_LOG_LEVEL_ERROR, __VA_ARGS__)  // 宏定义 GGML_LOG_ERROR
#define GGML_LOG_DEBUG(...) ggml_log_internal(GGML_LOG_LEVEL_DEBUG, __VA_ARGS__)  // 宏定义 GGML_LOG_DEBUG
#define GGML_LOG_CONT(...)  ggml_log_internal(GGML_LOG_LEVEL_CONT , __VA_ARGS__)  // 宏定义 GGML_LOG_CONT

#define GGML_DEBUG 0  // 宏定义 GGML_DEBUG

#if (GGML_DEBUG >= 1)  // 条件编译
#define GGML_PRINT_DEBUG(...) GGML_LOG_DEBUG(__VA_ARGS__)  // 宏定义 GGML_PRINT_DEBUG
#else  // 否则
#define GGML_PRINT_DEBUG(...)  // 宏定义 GGML_PRINT_DEBUG
#endif  // 条件编译结束

#if (GGML_DEBUG >= 5)  // 条件编译
#define GGML_PRINT_DEBUG_5(...) GGML_LOG_DEBUG(__VA_ARGS__)  // 宏定义 GGML_PRINT_DEBUG_5
#else  // 否则
#define GGML_PRINT_DEBUG_5(...)  // 宏定义 GGML_PRINT_DEBUG_5
#endif  // 条件编译结束

#if (GGML_DEBUG >= 10)  // 条件编译
#define GGML_PRINT_DEBUG_10(...) GGML_LOG_DEBUG(__VA_ARGS__)  // 宏定义 GGML_PRINT_DEBUG_10
#else  // 否则
#define GGML_PRINT_DEBUG_10(...)  // 宏定义 GGML_PRINT_DEBUG_10
#endif  // 条件编译结束

// tensor params

static void ggml_set_op_params(struct ggml_tensor * tensor, const void * params, size_t params_size) {
    GGML_ASSERT(tensor != NULL); // silence -Warray-bounds warnings
    assert(params_size <= GGML_MAX_OP_PARAMS);
    memcpy(tensor->op_params, params, params_size);
}

static int32_t ggml_get_op_params_i32(const struct ggml_tensor * tensor, uint32_t i) {
    assert(i < GGML_MAX_OP_PARAMS / sizeof(int32_t));
    return ((const int32_t *)(tensor->op_params))[i];
}

static float ggml_get_op_params_f32(const struct ggml_tensor * tensor, uint32_t i) {
    assert(i < GGML_MAX_OP_PARAMS / sizeof(float));
    return ((const float *)(tensor->op_params))[i];
}

static void ggml_set_op_params_i32(struct ggml_tensor * tensor, uint32_t i, int32_t value) {
    assert(i < GGML_MAX_OP_PARAMS / sizeof(int32_t));
    ((int32_t *)(tensor->op_params))[i] = value;
}

static void ggml_set_op_params_f32(struct ggml_tensor * tensor, uint32_t i, float value) {
    assert(i < GGML_MAX_OP_PARAMS / sizeof(float));
    ((float *)(tensor->op_params))[i] = value;
}

struct ggml_map_custom1_op_params {  // 结构体定义
    ggml_custom1_op_t  fun;
    int                n_tasks;
    void             * userdata;
};

struct ggml_map_custom2_op_params {  // 结构体定义
    ggml_custom2_op_t   fun;
    int                 n_tasks;
    void              * userdata;
};

struct ggml_map_custom3_op_params {  // 结构体定义
    ggml_custom3_op_t fun;
    int               n_tasks;
    void            * userdata;
};

struct ggml_custom_op_params {  // 结构体定义
    ggml_custom_op_t fun;
    int              n_tasks;
    void           * userdata;
};

// bitset

typedef uint32_t ggml_bitset_t;  // 类型定义

static_assert(sizeof(ggml_bitset_t) == 4, "bitset_t constants must be updated");
#define BITSET_SHR 5 // log2(sizeof(ggml_bitset_t)*8)  // 宏定义 BITSET_SHR
#define BITSET_MASK (sizeof(ggml_bitset_t)*8 - 1)  // 宏定义 BITSET_MASK

static size_t ggml_bitset_size(size_t n) {
    return (n + BITSET_MASK) >> BITSET_SHR;
}

static inline bool ggml_bitset_get(const ggml_bitset_t * bitset, size_t i) {
    return !!(bitset[i >> BITSET_SHR] & (1u << (i & BITSET_MASK)));
}

static inline void ggml_bitset_set(ggml_bitset_t * bitset, size_t i) {
    bitset[i >> BITSET_SHR] |= (1u << (i & BITSET_MASK));
}

static inline void ggml_bitset_clear(ggml_bitset_t * bitset, size_t i) {
    bitset[i >> BITSET_SHR] &= ~(1u << (i & BITSET_MASK));
}

// hash set

#define GGML_HASHSET_FULL ((size_t)-1)  // 宏定义 GGML_HASHSET_FULL
#define GGML_HASHSET_ALREADY_EXISTS ((size_t)-2)  // 宏定义 GGML_HASHSET_ALREADY_EXISTS

struct ggml_hash_set {  // 结构体定义
    size_t size;
    ggml_bitset_t * used;       // whether or not the keys are in use i.e. set
    struct ggml_tensor ** keys; // actual tensors in the set, keys[i] is only defined if ggml_bitset_get(used, i)
};

struct ggml_hash_set ggml_hash_set_new(size_t size);  // ggml_hash_set_new
void                 ggml_hash_set_free(struct ggml_hash_set * hash_set);  // ggml_hash_set_free

// returns the minimum size for a hash set that can hold min_sz elements
size_t ggml_hash_size(size_t min_sz);  // ggml_hash_size

// remove all elements from the hash set
void ggml_hash_set_reset(struct ggml_hash_set * hash_set);  // ggml_hash_set_reset

// returns true if key is in the hash set
static bool ggml_hash_contains(const struct ggml_hash_set * hash_set, struct ggml_tensor * key);  // ggml_hash_contains

// returns GGML_HASHSET_FULL if table is full, otherwise the current index of the key or where it should be inserted
static size_t ggml_hash_find(const struct ggml_hash_set * hash_set, const struct ggml_tensor * key);  // ggml_hash_find

// returns GGML_HASHSET_ALREADY_EXISTS if key already exists, index otherwise, asserts if table is full
static size_t ggml_hash_insert(struct ggml_hash_set * hash_set, struct ggml_tensor * key);  // ggml_hash_insert

// return index, asserts if table is full
static size_t ggml_hash_find_or_insert(struct ggml_hash_set * hash_set, struct ggml_tensor * key);  // ggml_hash_find_or_insert

// hash function for ggml_tensor
static inline size_t ggml_hash(const struct ggml_tensor * p) {
    // the last 4 bits are always zero due to alignment
    return (size_t)(uintptr_t)p >> 4;
}

static size_t ggml_hash_find(const struct ggml_hash_set * hash_set, const struct ggml_tensor * key) {
    size_t h = ggml_hash(key) % hash_set->size;

    // linear probing
    size_t i = h;
    while (ggml_bitset_get(hash_set->used, i) && hash_set->keys[i] != key) {
        i = (i + 1) % hash_set->size;
        if (i == h) {
            // visited all hash table entries -> not found
            return GGML_HASHSET_FULL;  // 返回
        }
    }
    return i;  // 返回
}

static bool ggml_hash_contains(const struct ggml_hash_set * hash_set, struct ggml_tensor * key) {
    size_t i = ggml_hash_find(hash_set, key);
    return i != GGML_HASHSET_FULL && ggml_bitset_get(hash_set->used, i);
}

static size_t ggml_hash_insert(struct ggml_hash_set * hash_set, struct ggml_tensor * key) {
    size_t h = ggml_hash(key) % hash_set->size;

    // linear probing
    size_t i = h;
    do {
        if (!ggml_bitset_get(hash_set->used, i)) {
            ggml_bitset_set(hash_set->used, i);
            hash_set->keys[i] = key;
            return i;  // 返回
        }
        if (hash_set->keys[i] == key) {
            return GGML_HASHSET_ALREADY_EXISTS;  // 返回
        }
        i = (i + 1) % hash_set->size;
    } while (i != h);

    // visited all hash table entries -> not found
    GGML_ABORT("fatal error");
}

static size_t ggml_hash_find_or_insert(struct ggml_hash_set * hash_set, struct ggml_tensor * key) {
    size_t h = ggml_hash(key) % hash_set->size;

    // linear probing
    size_t i = h;
    do {
        if (!ggml_bitset_get(hash_set->used, i)) {
            ggml_bitset_set(hash_set->used, i);
            hash_set->keys[i] = key;
            return i;  // 返回
        }
        if (hash_set->keys[i] == key) {
            return i;  // 返回
        }
        i = (i + 1) % hash_set->size;
    } while (i != h);

    // visited all hash table entries -> not found
    GGML_ABORT("fatal error");
}

// computation graph

enum ggml_cgraph_eval_order {  // 枚举定义
    GGML_CGRAPH_EVAL_ORDER_LEFT_TO_RIGHT = 0,
    GGML_CGRAPH_EVAL_ORDER_RIGHT_TO_LEFT,
    GGML_CGRAPH_EVAL_ORDER_COUNT
};

struct ggml_cgraph {  // 结构体定义
    int size;    // maximum number of nodes/leafs/grads/grad_accs
    int n_nodes; // number of nodes currently in use
    int n_leafs; // number of leafs currently in use

    struct ggml_tensor ** nodes;     // tensors with data that can change if the graph is evaluated
    struct ggml_tensor ** grads;     // the outputs of these tensors are the gradients of the nodes
    struct ggml_tensor ** grad_accs; // accumulators for node gradients
    struct ggml_tensor ** leafs;     // tensors with constant data
    int32_t             * use_counts;// number of uses of each tensor, indexed by hash table slot

    struct ggml_hash_set visited_hash_set;

    enum ggml_cgraph_eval_order order;

    // an optional identifier that can be utilized to recognize same graphs if two non-zero values match
    // a value of 0 means it is not set and should be ignored
    uint64_t uid;
};

// returns a slice of cgraph with nodes [i0, i1)
// the slice does not have leafs or gradients
// if you need the gradients, get them from the original graph
struct ggml_cgraph ggml_graph_view(struct ggml_cgraph * cgraph, int i0, int i1);  // ggml_graph_view

// ggml-alloc.c: true if the operation can reuse memory from its sources
GGML_API bool ggml_op_can_inplace(enum ggml_op op);


// Memory allocation

GGML_API void * ggml_aligned_malloc(size_t size);
GGML_API void ggml_aligned_free(void * ptr, size_t size);

// FP16 <-> FP32
// ref: https://github.com/Maratyszcza/FP16

static inline float fp32_from_bits(uint32_t w) {
    union {
        uint32_t as_bits;
        float as_value;
    } fp32;
    fp32.as_bits = w;
    return fp32.as_value;  // 返回
}

static inline uint32_t fp32_to_bits(float f) {
    union {
        float as_value;
        uint32_t as_bits;
    } fp32;
    fp32.as_value = f;
    return fp32.as_bits;  // 返回
}

static inline float ggml_compute_fp16_to_fp32(ggml_fp16_t h) {
    const uint32_t w = (uint32_t) h << 16;
    const uint32_t sign = w & UINT32_C(0x80000000);
    const uint32_t two_w = w + w;

    const uint32_t exp_offset = UINT32_C(0xE0) << 23;
#if (defined(__STDC_VERSION__) && (__STDC_VERSION__ >= 199901L) || defined(__GNUC__) && !defined(__STRICT_ANSI__)) && (!defined(__cplusplus) || __cplusplus >= 201703L)  // 条件编译
    const float exp_scale = 0x1.0p-112f;
#else  // 否则
    const float exp_scale = fp32_from_bits(UINT32_C(0x7800000));
#endif  // 条件编译结束
    const float normalized_value = fp32_from_bits((two_w >> 4) + exp_offset) * exp_scale;

    const uint32_t magic_mask = UINT32_C(126) << 23;
    const float magic_bias = 0.5f;
    const float denormalized_value = fp32_from_bits((two_w >> 17) | magic_mask) - magic_bias;

    const uint32_t denormalized_cutoff = UINT32_C(1) << 27;
    const uint32_t result = sign |
        (two_w < denormalized_cutoff ? fp32_to_bits(denormalized_value) : fp32_to_bits(normalized_value));
    return fp32_from_bits(result);  // fp32_from_bits
}

static inline ggml_fp16_t ggml_compute_fp32_to_fp16(float f) {
#if (defined(__STDC_VERSION__) && (__STDC_VERSION__ >= 199901L) || defined(__GNUC__) && !defined(__STRICT_ANSI__)) && (!defined(__cplusplus) || __cplusplus >= 201703L)  // 条件编译
    const float scale_to_inf = 0x1.0p+112f;
    const float scale_to_zero = 0x1.0p-110f;
#else  // 否则
    const float scale_to_inf = fp32_from_bits(UINT32_C(0x77800000));
    const float scale_to_zero = fp32_from_bits(UINT32_C(0x08800000));
#endif  // 条件编译结束
    float base = (fabsf(f) * scale_to_inf) * scale_to_zero;

    const uint32_t w = fp32_to_bits(f);
    const uint32_t shl1_w = w + w;
    const uint32_t sign = w & UINT32_C(0x80000000);
    uint32_t bias = shl1_w & UINT32_C(0xFF000000);
    if (bias < UINT32_C(0x71000000)) {
        bias = UINT32_C(0x71000000);
    }

    base = fp32_from_bits((bias >> 1) + UINT32_C(0x07800000)) + base;
    const uint32_t bits = fp32_to_bits(base);
    const uint32_t exp_bits = (bits >> 13) & UINT32_C(0x00007C00);
    const uint32_t mantissa_bits = bits & UINT32_C(0x00000FFF);
    const uint32_t nonsign = exp_bits + mantissa_bits;
    return (sign >> 16) | (shl1_w > UINT32_C(0xFF000000) ? UINT16_C(0x7E00) : nonsign);
}

#define GGML_COMPUTE_FP16_TO_FP32(x) ggml_compute_fp16_to_fp32(x)  // 宏定义 GGML_COMPUTE_FP16_TO_FP32
#define GGML_COMPUTE_FP32_TO_FP16(x) ggml_compute_fp32_to_fp16(x)  // 宏定义 GGML_COMPUTE_FP32_TO_FP16

#define GGML_FP16_TO_FP32(x) GGML_COMPUTE_FP16_TO_FP32(x)  // 宏定义 GGML_FP16_TO_FP32
#define GGML_FP32_TO_FP16(x) GGML_COMPUTE_FP32_TO_FP16(x)  // 宏定义 GGML_FP32_TO_FP16

static inline float ggml_e8m0_to_fp32(uint8_t x) {
    uint32_t bits;  // Stores the raw bit representation of the float

    // Handle special case for minimum exponent (denormalized float)
    if (x == 0) {
        // Bit pattern for 2^(-127):
        // - Sign bit: 0 (positive)
        // - Exponent: 0 (denormalized number)
        // - Mantissa: 0x400000 (0.5 in fractional form)
        // Value = 0.5 * 2^(-126) = 2^(-127)
        bits = 0x00400000;
    }
    // note: disabled as we don't need to handle NaNs
    //// Handle special case for NaN (all bits set)
    //else if (x == 0xFF) {
    //    // Standard quiet NaN pattern:
    //    // - Sign bit: 0
    //    // - Exponent: all 1s (0xFF)
    //    // - Mantissa: 0x400000 (quiet NaN flag)
    //    bits = 0x7FC00000;
    //}
    // Normalized values (most common case)
    else {
        // Construct normalized float by shifting exponent into position:
        // - Exponent field: 8 bits (positions 30-23)
        // - Mantissa: 0 (implicit leading 1)
        // Value = 2^(x - 127)
        bits = (uint32_t) x << 23;
    }

    float result;  // Final float value
                   // Safely reinterpret bit pattern as float without type-punning issues
    memcpy(&result, &bits, sizeof(float));
    return result;  // 返回
}

// Equal to ggml_e8m0_to_fp32/2
// Useful with MXFP4 quantization since the E0M2 values are doubled
static inline float ggml_e8m0_to_fp32_half(uint8_t x) {
    uint32_t bits;

    // For x < 2: use precomputed denormal patterns
    if (x < 2) {
        // 0x00200000 = 2^(-128), 0x00400000 = 2^(-127)
        bits = 0x00200000 << x;
    }
    // For x >= 2: normalized exponent adjustment
    else {
        // 0.5 * 2^(x-127) = 2^(x-128) = normalized with exponent (x-1)
        bits = (uint32_t)(x - 1) << 23;
    }
    // Note: NaNs are not handled here

    float result;
    memcpy(&result, &bits, sizeof(float));
    return result;  // 返回
}

#define GGML_E8M0_TO_FP32(x) ggml_e8m0_to_fp32(x)  // 宏定义 GGML_E8M0_TO_FP32
#define GGML_E8M0_TO_FP32_HALF(x) ggml_e8m0_to_fp32_half(x)  // 宏定义 GGML_E8M0_TO_FP32_HALF

// UE4M3: unsigned, 4 exp bits (bias=7), 3 mantissa bits
// Returns value * 0.5 to match kvalues_mxfp4 convention (kvalues = 2 * E2M1_float)
static inline float ggml_ue4m3_to_fp32(uint8_t x) {
    if (x == 0 || x == 0x7F) {
        return 0.0f;  // 返回
    }
    int   exp = (x >> 3) & 0xF;
    int   man = x & 0x7;
    float raw;
    if (exp == 0) {
        raw = ldexpf((float) man, -9);
    } else {
        raw = ldexpf(1.0f + (float) man / 8.0f, exp - 7);
    }
    return raw * 0.5f;  // 返回
}

static inline uint8_t ggml_fp32_to_ue4m3(float x) {
    if (!(x > 0.0f)) {
        return 0;  // 返回
    }
    if (x > 448.0f) {
        x = 448.0f;
    }
    uint32_t bits;
    memcpy(&bits, &x, 4);
    int fp32_exp  = ((bits >> 23) & 0xFF) - 127;
    int fp32_man  = (bits >> 20) & 0x7;
    int ue4m3_exp = fp32_exp + 7;
    if (ue4m3_exp <= 0) {
        // subnormal: value = man * 2^-9, man = round(x * 2^9)
        int man = (int) (x * 512.0f + 0.5f);
        if (man > 7) {
            man = 7;
        }
        if (man < 1) {
            return 0;  // 返回
        }
        return (uint8_t) man;
    }
    if (ue4m3_exp >= 15) {
        return 0x7E;  // 返回
    }
    int round_bit = (bits >> 19) & 1;
    int ue4m3_man = fp32_man + round_bit;
    if (ue4m3_man > 7) {
        ue4m3_man = 0;
        ue4m3_exp++;
        if (ue4m3_exp >= 15) {
            return 0x7E;  // 返回
        }
    }
    return (uint8_t) ((ue4m3_exp << 3) | ue4m3_man);
}

/**
 * Converts brain16 to float32.
 *
 * The bfloat16 floating point format has the following structure:
 *
 *       ┌sign
 *       │
 *       │   ┌exponent
 *       │   │
 *       │   │      ┌mantissa
 *       │   │      │
 *       │┌──┴───┐┌─┴───┐
 *     0b0000000000000000 brain16
 *
 * Since bf16 has the same number of exponent bits as a 32bit float,
 * encoding and decoding numbers becomes relatively straightforward.
 *
 *       ┌sign
 *       │
 *       │   ┌exponent
 *       │   │
 *       │   │      ┌mantissa
 *       │   │      │
 *       │┌──┴───┐┌─┴───────────────────┐
 *     0b00000000000000000000000000000000 IEEE binary32
 *
 * For comparison, the standard fp16 format has fewer exponent bits.
 *
 *       ┌sign
 *       │
 *       │  ┌exponent
 *       │  │
 *       │  │    ┌mantissa
 *       │  │    │
 *       │┌─┴─┐┌─┴──────┐
 *     0b0000000000000000 IEEE binary16
 *
 * @see IEEE 754-2008
 */
static inline float ggml_compute_bf16_to_fp32(ggml_bf16_t h) {
    union {
        float f;
        uint32_t i;
    } u;
    u.i = (uint32_t)h.bits << 16;
    return u.f;  // 返回
}

/**
 * Converts float32 to brain16.
 *
 * This is binary identical with Google Brain float conversion.
 * Floats shall round to nearest even, and NANs shall be quiet.
 * Subnormals aren't flushed to zero, except perhaps when used.
 * This code should vectorize nicely if using modern compilers.
 */
static inline ggml_bf16_t ggml_compute_fp32_to_bf16(float s) {
    ggml_bf16_t h;
    union {
        float f;
        uint32_t i;
    } u;
    u.f = s;
    if ((u.i & 0x7fffffff) > 0x7f800000) { /* nan */
        h.bits = (u.i >> 16) | 64; /* force to quiet */
        return h;  // 返回
    }
    h.bits = (u.i + (0x7fff + ((u.i >> 16) & 1))) >> 16;
    return h;  // 返回
}

#define GGML_FP32_TO_BF16(x) ggml_compute_fp32_to_bf16(x)  // 宏定义 GGML_FP32_TO_BF16
#define GGML_BF16_TO_FP32(x) ggml_compute_bf16_to_fp32(x)  // 宏定义 GGML_BF16_TO_FP32

static inline int32_t ggml_node_get_use_count(const struct ggml_cgraph * cgraph, int node_idx) {
    const struct ggml_tensor * node = cgraph->nodes[node_idx];

    size_t hash_pos = ggml_hash_find(&cgraph->visited_hash_set, node);
    if (!ggml_bitset_get(cgraph->visited_hash_set.used, hash_pos)) {
        return 0;  // 返回
    }
    return cgraph->use_counts[hash_pos];  // 返回
}

// return true if the node's results are only used by N other nodes
// and can be fused into their calculations.
static inline bool ggml_node_has_n_uses(const struct ggml_cgraph * cgraph, int node_idx, int32_t n_uses) {
    const struct ggml_tensor * node = cgraph->nodes[node_idx];

    // check the use count against how many we're replacing
    if (ggml_node_get_use_count(cgraph, node_idx) != n_uses) {
        return false;  // 返回
    }

    // if node is a view, some other node might be using the intermediate result
    // via the view source.
    if (node->view_src) {
        return false;  // 返回
    }

    // If the user requested output for the node, can't fuse
    if (node->flags & GGML_TENSOR_FLAG_OUTPUT) {
        return false;  // 返回
    }

    return true;  // 返回
}

// Returns true if nodes with indices { node_idxs } are the sequence of ggml_ops in ops[]
// and are fusable. Nodes are considered fusable according to this function if:
// - all nodes except the last have only one use and are not views/outputs (see ggml_node_has_N_uses).
// - all nodes except the last are a src of the following node.
// - all nodes are the same shape.
// TODO: Consider allowing GGML_OP_NONE nodes in between
static inline bool ggml_can_fuse_ext(const struct ggml_cgraph * cgraph, const int * node_idxs, const enum ggml_op * ops, int num_ops) {
    for (int i = 0; i < num_ops; ++i) {
        if (node_idxs[i] >= cgraph->n_nodes) {
            return false;  // 返回
        }

        struct ggml_tensor * node = cgraph->nodes[node_idxs[i]];
        if (node->op != ops[i]) {
            return false;  // 返回
        }
        if ((node->flags & GGML_TENSOR_FLAG_COMPUTE) == 0) {
            return false;  // 返回
        }
        if (i < num_ops - 1 && !ggml_node_has_n_uses(cgraph, node_idxs[i], 1)) {
            return false;  // 返回
        }
        if (i > 0) {
            struct ggml_tensor * prev = cgraph->nodes[node_idxs[i - 1]];
            if (node->src[0] != prev && node->src[1] != prev) {
                return false;  // 返回
            }
            if (!ggml_are_same_shape(node, prev)) {
                return false;  // 返回
            }
        }
    }
    return true;  // 返回
}

// same as above, for sequential indices starting at node_idx
static inline bool ggml_can_fuse(const struct ggml_cgraph * cgraph, int node_idx, const enum ggml_op * ops, int num_ops) {
    assert(num_ops < 32);

    if (node_idx + num_ops > cgraph->n_nodes) {
        return false;  // 返回
    }

    int idxs[32];
    for (int i = 0; i < num_ops; ++i) {
        idxs[i] = node_idx + i;
    }

    return ggml_can_fuse_ext(cgraph, idxs, ops, num_ops);  // ggml_can_fuse_ext
}

GGML_API bool ggml_can_fuse_subgraph_ext(const struct ggml_cgraph * cgraph,
                                         const int *                node_idxs,
                                         int                        count,
                                         const enum ggml_op *       ops,
                                         const int *                outputs,
                                         int                        num_outputs);

// Returns true if the subgraph formed by {node_idxs} can be fused
// checks whethers all nodes which are not part of outputs can be elided
// by checking if their num_uses are confined to the subgraph
static inline bool ggml_can_fuse_subgraph(const struct ggml_cgraph * cgraph,
                                          int                        node_idx,
                                          int                        count,
                                          const enum ggml_op *       ops,
                                          const int *                outputs,
                                          int                        num_outputs) {
    GGML_ASSERT(count < 32);
    if (node_idx + count > cgraph->n_nodes) {
        return false;  // 返回
    }

    int idxs[32];

    for (int i = 0; i < count; ++i) {
        idxs[i] = node_idx + i;
    }

    return ggml_can_fuse_subgraph_ext(cgraph, idxs, count, ops, outputs, num_outputs);  // ggml_can_fuse_subgraph_ext
}

#ifdef __cplusplus  // 如果定义了 __cplusplus 则编译
}
#endif  // 条件编译结束

#ifdef __cplusplus  // 如果定义了 __cplusplus 则编译
#include <array>  // 引入 array 头文件
#include <initializer_list>  // 引入 initializer_list 头文件
#include <vector>  // 引入 vector 头文件

// nicer C++ syntax for ggml_can_fuse
inline bool ggml_can_fuse(const struct ggml_cgraph * cgraph, int node_idx, std::initializer_list<enum ggml_op> ops) {
    return ggml_can_fuse(cgraph, node_idx, ops.begin(), (int)ops.size());  // ggml_can_fuse
}

inline bool ggml_can_fuse_subgraph(const struct ggml_cgraph *          cgraph,
                                   int                                 start_idx,
                                   std::initializer_list<enum ggml_op> ops,
                                   std::initializer_list<int>          outputs = {}) {
    return ggml_can_fuse_subgraph(cgraph, start_idx, ops.size(), ops.begin(), outputs.begin(), outputs.size());  // ggml_can_fuse_subgraph
}

// Return true if the edges in the graph match expectations.
inline bool ggml_check_edges(const struct ggml_cgraph *                cgraph,
                             int                                       start_idx,
                             std::initializer_list<std::array<int, 3>> edges) {
    for (const auto & edge : edges) {
        int dst_node = edge[0];
        int src_idx  = edge[1];
        int src_node = edge[2];
        if (cgraph->nodes[start_idx + dst_node]->src[src_idx] != cgraph->nodes[start_idx + src_node]) {
            return false;  // 返回
        }
    }
    return true;  // 返回
}

// expose GGUF internals for test code
GGML_API size_t gguf_type_size(enum gguf_type type);
GGML_API void gguf_write_to_buf(const struct gguf_context * ctx, std::vector<int8_t> & buf, bool only_meta);
#endif // __cplusplus  // 条件编译结束
