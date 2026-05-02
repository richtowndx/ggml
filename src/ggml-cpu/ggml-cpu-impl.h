#pragma once  // 防止重复包含

// GGML CPU internal header

#include "ggml.h"  // 引入 ggml.h 头文件
#include "ggml-impl.h"  // 引入 ggml-impl.h 头文件

#include <stdlib.h> // load `stdlib.h` before other headers to work around MinGW bug: https://sourceforge.net/p/mingw-w64/bugs/192/  // 引入 stdlib.h 头文件
//#include <stddef.h>
#include <stdbool.h>  // 引入 stdbool.h 头文件
#include <string.h> // memcpy  // 引入 string.h 头文件
#include <math.h>   // fabsf  // 引入 math.h 头文件

#ifdef __cplusplus  // 如果定义了 __cplusplus 则编译
extern "C" {  // C 链接声明
#endif  // 条件编译结束

struct ggml_compute_params {  // 结构体定义
    // ith = thread index, nth = number of threads
    int ith, nth;

    // work buffer for all threads
    size_t wsize;
    void * wdata;

    struct ggml_threadpool * threadpool;

    // use reference implementation
    bool use_ref;
};


#if defined(_MSC_VER)  // 条件编译

#define m512bh(p) p  // 宏定义 m512bh
#define m512i(p) p  // 宏定义 m512i

#else  // 否则

#define m512bh(p) (__m512bh)(p)  // 宏定义 m512bh
#define m512i(p) (__m512i)(p)  // 宏定义 m512i

#endif  // 条件编译结束

// __FMA__ and __F16C__ are not defined in MSVC, however they are implied with AVX2/AVX512
#if defined(_MSC_VER) && (defined(__AVX2__) || defined(__AVX512F__))  // 条件编译
#ifndef __FMA__  // 如果未定义 __FMA__ 则编译
#define __FMA__  // 宏定义 __FMA__
#endif  // 条件编译结束
#ifndef __F16C__  // 如果未定义 __F16C__ 则编译
#define __F16C__  // 宏定义 __F16C__
#endif  // 条件编译结束
#endif  // 条件编译结束

// __SSE3__ and __SSSE3__ are not defined in MSVC, but SSE3/SSSE3 are present when AVX/AVX2/AVX512 are available
#if defined(_MSC_VER) && (defined(__AVX__) || defined(__AVX2__) || defined(__AVX512F__))  // 条件编译
#ifndef __SSE3__  // 如果未定义 __SSE3__ 则编译
#define __SSE3__  // 宏定义 __SSE3__
#endif  // 条件编译结束
#ifndef __SSSE3__  // 如果未定义 __SSSE3__ 则编译
#define __SSSE3__  // 宏定义 __SSSE3__
#endif  // 条件编译结束
#endif  // 条件编译结束

#if defined(__s390x__) && defined(__VEC__)  // 条件编译
#ifndef __VXE__  // 如果未定义 __VXE__ 则编译
#define __VXE__  // 宏定义 __VXE__
#endif  // __VXE__  // 条件编译结束
#ifndef __VXE2__  // 如果未定义 __VXE2__ 则编译
#define __VXE2__  // 宏定义 __VXE2__
#endif  // __VXE2__  // 条件编译结束
#endif  // __s390x__ && __VEC__  // 条件编译结束

#if defined(__ARM_FEATURE_SVE) && defined(__linux__)  // 条件编译
#include <sys/prctl.h>  // 引入 sys/prctl.h 头文件
#endif  // 条件编译结束

#if defined(__ARM_NEON)  // 条件编译

// ref: https://github.com/ggml-org/llama.cpp/pull/5404
#ifdef _MSC_VER  // 如果定义了 _MSC_VER 则编译
#define ggml_vld1q_u32(w,x,y,z) { ((w) + ((uint64_t)(x) << 32)), ((y) + ((uint64_t)(z) << 32)) }  // 宏定义 ggml_vld1q_u32
#else  // 否则
#define ggml_vld1q_u32(w,x,y,z) { (w), (x), (y), (z) }  // 宏定义 ggml_vld1q_u32
#endif // _MSC_VER  // 条件编译结束

#if !defined(__aarch64__)  // 条件编译

// 32-bit ARM compatibility

// vaddlvq_s16
// vpaddq_s16
// vpaddq_s32
// vaddvq_s32
// vaddvq_f32
// vmaxvq_f32
// vcvtnq_s32_f32
// vzip1_u8
// vzip2_u8

inline static int32_t vaddlvq_s16(int16x8_t v) {
    int32x4_t v0 = vreinterpretq_s32_s64(vpaddlq_s32(vpaddlq_s16(v)));
    return vgetq_lane_s32(v0, 0) + vgetq_lane_s32(v0, 2);  // vgetq_lane_s32
}

inline static int16x8_t vpaddq_s16(int16x8_t a, int16x8_t b) {
    int16x4_t a0 = vpadd_s16(vget_low_s16(a), vget_high_s16(a));
    int16x4_t b0 = vpadd_s16(vget_low_s16(b), vget_high_s16(b));
    return vcombine_s16(a0, b0);  // vcombine_s16
}

inline static int32x4_t vpaddq_s32(int32x4_t a, int32x4_t b) {
    int32x2_t a0 = vpadd_s32(vget_low_s32(a), vget_high_s32(a));
    int32x2_t b0 = vpadd_s32(vget_low_s32(b), vget_high_s32(b));
    return vcombine_s32(a0, b0);  // vcombine_s32
}

inline static int32_t vaddvq_s32(int32x4_t v) {
    return vgetq_lane_s32(v, 0) + vgetq_lane_s32(v, 1) + vgetq_lane_s32(v, 2) + vgetq_lane_s32(v, 3);  // vgetq_lane_s32
}

inline static float vaddvq_f32(float32x4_t v) {
    return vgetq_lane_f32(v, 0) + vgetq_lane_f32(v, 1) + vgetq_lane_f32(v, 2) + vgetq_lane_f32(v, 3);  // vgetq_lane_f32
}

inline static float vmaxvq_f32(float32x4_t v) {
    return
        MAX(MAX(vgetq_lane_f32(v, 0), vgetq_lane_f32(v, 1)),
            MAX(vgetq_lane_f32(v, 2), vgetq_lane_f32(v, 3)));
}

inline static int32x4_t vcvtnq_s32_f32(float32x4_t v) {
    int32x4_t res;

    res[0] = roundf(vgetq_lane_f32(v, 0));
    res[1] = roundf(vgetq_lane_f32(v, 1));
    res[2] = roundf(vgetq_lane_f32(v, 2));
    res[3] = roundf(vgetq_lane_f32(v, 3));

    return res;  // 返回
}

inline static uint8x8_t vzip1_u8(uint8x8_t a, uint8x8_t b) {
    uint8x8_t res;

    res[0] = a[0]; res[1] = b[0];
    res[2] = a[1]; res[3] = b[1];
    res[4] = a[2]; res[5] = b[2];
    res[6] = a[3]; res[7] = b[3];

    return res;  // 返回
}

inline static uint8x8_t vzip2_u8(uint8x8_t a, uint8x8_t b) {
    uint8x8_t res;

    res[0] = a[4]; res[1] = b[4];
    res[2] = a[5]; res[3] = b[5];
    res[4] = a[6]; res[5] = b[6];
    res[6] = a[7]; res[7] = b[7];

    return res;  // 返回
}

// vld1q_s16_x2
// vld1q_u8_x2
// vld1q_u8_x4
// vld1q_s8_x2
// vld1q_s8_x4
// TODO: double-check these work correctly

typedef struct ggml_int16x8x2_t {  // 类型定义
    int16x8_t val[2];
} ggml_int16x8x2_t;

inline static ggml_int16x8x2_t ggml_vld1q_s16_x2(const int16_t * ptr) {
    ggml_int16x8x2_t res;

    res.val[0] = vld1q_s16(ptr + 0);
    res.val[1] = vld1q_s16(ptr + 8);

    return res;  // 返回
}

typedef struct ggml_uint8x16x2_t {  // 类型定义
    uint8x16_t val[2];
} ggml_uint8x16x2_t;

inline static ggml_uint8x16x2_t ggml_vld1q_u8_x2(const uint8_t * ptr) {
    ggml_uint8x16x2_t res;

    res.val[0] = vld1q_u8(ptr + 0);
    res.val[1] = vld1q_u8(ptr + 16);

    return res;  // 返回
}

typedef struct ggml_uint8x16x4_t {  // 类型定义
    uint8x16_t val[4];
} ggml_uint8x16x4_t;

inline static ggml_uint8x16x4_t ggml_vld1q_u8_x4(const uint8_t * ptr) {
    ggml_uint8x16x4_t res;

    res.val[0] = vld1q_u8(ptr + 0);
    res.val[1] = vld1q_u8(ptr + 16);
    res.val[2] = vld1q_u8(ptr + 32);
    res.val[3] = vld1q_u8(ptr + 48);

    return res;  // 返回
}

typedef struct ggml_int8x16x2_t {  // 类型定义
    int8x16_t val[2];
} ggml_int8x16x2_t;

inline static ggml_int8x16x2_t ggml_vld1q_s8_x2(const int8_t * ptr) {
    ggml_int8x16x2_t res;

    res.val[0] = vld1q_s8(ptr + 0);
    res.val[1] = vld1q_s8(ptr + 16);

    return res;  // 返回
}

typedef struct ggml_int8x16x4_t {  // 类型定义
    int8x16_t val[4];
} ggml_int8x16x4_t;

inline static ggml_int8x16x4_t ggml_vld1q_s8_x4(const int8_t * ptr) {
    ggml_int8x16x4_t res;

    res.val[0] = vld1q_s8(ptr + 0);
    res.val[1] = vld1q_s8(ptr + 16);
    res.val[2] = vld1q_s8(ptr + 32);
    res.val[3] = vld1q_s8(ptr + 48);

    return res;  // 返回
}

// NOTE: not tested
inline static int8x16_t ggml_vqtbl1q_s8(int8x16_t a, uint8x16_t b) {
    int8x16_t res;

    res[ 0] = a[b[ 0]];
    res[ 1] = a[b[ 1]];
    res[ 2] = a[b[ 2]];
    res[ 3] = a[b[ 3]];
    res[ 4] = a[b[ 4]];
    res[ 5] = a[b[ 5]];
    res[ 6] = a[b[ 6]];
    res[ 7] = a[b[ 7]];
    res[ 8] = a[b[ 8]];
    res[ 9] = a[b[ 9]];
    res[10] = a[b[10]];
    res[11] = a[b[11]];
    res[12] = a[b[12]];
    res[13] = a[b[13]];
    res[14] = a[b[14]];
    res[15] = a[b[15]];

    return res;  // 返回
}

// NOTE: not tested
inline static uint8x16_t ggml_vqtbl1q_u8(uint8x16_t a, uint8x16_t b) {
    uint8x16_t res;

    res[ 0] = a[b[ 0]];
    res[ 1] = a[b[ 1]];
    res[ 2] = a[b[ 2]];
    res[ 3] = a[b[ 3]];
    res[ 4] = a[b[ 4]];
    res[ 5] = a[b[ 5]];
    res[ 6] = a[b[ 6]];
    res[ 7] = a[b[ 7]];
    res[ 8] = a[b[ 8]];
    res[ 9] = a[b[ 9]];
    res[10] = a[b[10]];
    res[11] = a[b[11]];
    res[12] = a[b[12]];
    res[13] = a[b[13]];
    res[14] = a[b[14]];
    res[15] = a[b[15]];

    return res;  // 返回
}

#else  // 否则

#define ggml_int16x8x2_t  int16x8x2_t  // 宏定义 ggml_int16x8x2_t
#define ggml_uint8x16x2_t uint8x16x2_t  // 宏定义 ggml_uint8x16x2_t
#define ggml_uint8x16x4_t uint8x16x4_t  // 宏定义 ggml_uint8x16x4_t
#define ggml_int8x16x2_t  int8x16x2_t  // 宏定义 ggml_int8x16x2_t
#define ggml_int8x16x4_t  int8x16x4_t  // 宏定义 ggml_int8x16x4_t

#define ggml_vld1q_s16_x2 vld1q_s16_x2  // 宏定义 ggml_vld1q_s16_x2
#define ggml_vld1q_u8_x2  vld1q_u8_x2  // 宏定义 ggml_vld1q_u8_x2
#define ggml_vld1q_u8_x4  vld1q_u8_x4  // 宏定义 ggml_vld1q_u8_x4
#define ggml_vld1q_s8_x2  vld1q_s8_x2  // 宏定义 ggml_vld1q_s8_x2
#define ggml_vld1q_s8_x4  vld1q_s8_x4  // 宏定义 ggml_vld1q_s8_x4
#define ggml_vqtbl1q_s8   vqtbl1q_s8  // 宏定义 ggml_vqtbl1q_s8
#define ggml_vqtbl1q_u8   vqtbl1q_u8  // 宏定义 ggml_vqtbl1q_u8

#endif // !defined(__aarch64__)  // 条件编译结束

#if !defined(__ARM_FEATURE_DOTPROD)  // 条件编译

// NOTE: this fallback produces the same total sum as native vdotq_s32 but with different per-lane grouping — do not use when individual lane values matter.
inline static int32x4_t ggml_vdotq_s32(int32x4_t acc, int8x16_t a, int8x16_t b) {
    const int16x8_t p0 = vmull_s8(vget_low_s8 (a), vget_low_s8 (b));
    const int16x8_t p1 = vmull_s8(vget_high_s8(a), vget_high_s8(b));

    return vaddq_s32(acc, vaddq_s32(vpaddlq_s16(p0), vpaddlq_s16(p1)));  // vaddq_s32
}

#else  // 否则

#define ggml_vdotq_s32(a, b, c) vdotq_s32(a, b, c)  // 宏定义 ggml_vdotq_s32

#endif // !defined(__ARM_FEATURE_DOTPROD)  // 条件编译结束

static inline int32x4_t ggml_nvfp4_dot8(const int8x8_t q4_lo, const int8x8_t q8_lo,
                                         const int8x8_t q4_hi, const int8x8_t q8_hi) {
    const int16x8_t p_lo = vmull_s8(q4_lo, q8_lo);
    const int16x8_t p_hi = vmull_s8(q4_hi, q8_hi);
    const int32x4_t sum_lo = vpaddlq_s16(p_lo);
    const int32x4_t sum_hi = vpaddlq_s16(p_hi);
    return vaddq_s32(sum_lo, sum_hi);  // vaddq_s32
}

#endif // defined(__ARM_NEON)  // 条件编译结束

#ifdef __wasm_simd128__  // 如果定义了 __wasm_simd128__ 则编译
#include <wasm_simd128.h>  // 引入 wasm_simd128.h 头文件
#endif  // 条件编译结束

#ifdef __POWER9_VECTOR__  // 如果定义了 __POWER9_VECTOR__ 则编译
#include <altivec.h>  // 引入 altivec.h 头文件
#endif  // 条件编译结束

#if defined(_MSC_VER) || defined(__MINGW32__)  // 条件编译
#include <intrin.h>  // 引入 intrin.h 头文件
#elif defined(__SSE__) || defined(__SSE3__) || defined(__SSSE3__) || defined(__AVX__) || defined(__F16C__) || defined(__AVX2__) || defined(__AVX512F__) || defined(__AVX512BF16__)  // 否则如果
#include <immintrin.h>  // 引入 immintrin.h 头文件
#endif  // 条件编译结束

#ifdef __riscv_v_intrinsic  // 如果定义了 __riscv_v_intrinsic 则编译
#include <riscv_vector.h>  // 引入 riscv_vector.h 头文件
#endif  // 条件编译结束

#if defined(__loongarch64)  // 条件编译
#if defined(__loongarch_asx)  // 条件编译
#include <lasxintrin.h>  // 引入 lasxintrin.h 头文件
#endif  // 条件编译结束
#if defined(__loongarch_sx)  // 条件编译
#include <lsxintrin.h>  // 引入 lsxintrin.h 头文件
#endif  // 条件编译结束
#endif  // 条件编译结束

#if defined(__VXE__) || defined(__VXE2__)  // 条件编译
#include <vecintrin.h>  // 引入 vecintrin.h 头文件

#define vec_neg(a)    (-(a))                // Vector Negate  // 宏定义 vec_neg
#define vec_add(a, b) ((a) + (b))           // Vector Add  // 宏定义 vec_add
#define vec_sub(a, b) ((a) - (b))           // Vector Subtract  // 宏定义 vec_sub
#define vec_mul(a, b) ((a) * (b))           // Vector Multiply  // 宏定义 vec_mul
#define vec_div(a, b) ((a) / (b))           // Vector Divide  // 宏定义 vec_div
#define vec_sl(a, b)  ((a) << (b))          // Vector Shift Left  // 宏定义 vec_sl
#define vec_sra(a, b) ((a) >> (b))          // Vector Shift Right  // 宏定义 vec_sra
#define vec_sr(a, b)  ((a) >> (b))          // Vector Shift Right Algebraic  // 宏定义 vec_sr
#define vec_slo(a, b) vec_slb(a, (b) << 64) // Vector Shift Left by Octet  // 宏定义 vec_slo
#define vec_sro(a, b) vec_srb(a, (b) << 64) // Vector Shift Right by Octet  // 宏定义 vec_sro

#ifndef vec_and  // 如果未定义 vec_and 则编译
#define vec_and(a, b) ((a) & (b)) // Vector AND  // 宏定义 vec_and
#endif  // 条件编译结束

#ifndef vec_or  // 如果未定义 vec_or 则编译
#define vec_or(a, b)  ((a) | (b)) // Vector OR  // 宏定义 vec_or
#endif  // 条件编译结束

#ifndef vec_xor  // 如果未定义 vec_xor 则编译
#define vec_xor(a, b) ((a) ^ (b)) // Vector XOR  // 宏定义 vec_xor
#endif  // 条件编译结束

typedef signed   char char8x16_t  __attribute__((vector_size(16)));  // 类型定义
typedef unsigned char uchar8x16_t __attribute__((vector_size(16)));  // 类型定义

typedef int8_t  int8x16_t __attribute__((vector_size(16)));  // 类型定义
typedef int16_t int16x8_t __attribute__((vector_size(16)));  // 类型定义
typedef int32_t int32x4_t __attribute__((vector_size(16)));  // 类型定义

typedef uint8_t  uint8x16_t __attribute__((vector_size(16)));  // 类型定义
typedef uint16_t uint16x8_t __attribute__((vector_size(16)));  // 类型定义
typedef uint32_t uint32x4_t __attribute__((vector_size(16)));  // 类型定义

typedef float  float32x4_t  __attribute__((vector_size(16)));  // 类型定义
typedef double double64x2_t __attribute__((vector_size(16)));  // 类型定义

typedef signed   long long long64x2_t  __attribute__((vector_size(16)));  // 类型定义
typedef unsigned long long ulong64x2_t __attribute__((vector_size(16)));  // 类型定义

typedef struct ggml_uint8x16x2_t {  // 类型定义
    uint8x16_t val[2];
} ggml_uint8x16x2_t;

inline static ggml_uint8x16x2_t ggml_vec_xl_u8x2(const uint8_t * ptr) {
    ggml_uint8x16x2_t res;

    res.val[0] = vec_xl( 0, ptr);
    res.val[1] = vec_xl(16, ptr);

    return res;  // 返回
}

typedef struct ggml_uint8x16x4_t {  // 类型定义
    uint8x16_t val[4];
} ggml_uint8x16x4_t;

inline static ggml_uint8x16x4_t ggml_vec_xl_u8x4(const uint8_t * ptr) {
    ggml_uint8x16x4_t res;

    res.val[0] = vec_xl( 0, ptr);
    res.val[1] = vec_xl(16, ptr);
    res.val[2] = vec_xl(32, ptr);
    res.val[3] = vec_xl(48, ptr);

    return res;  // 返回
}

typedef struct ggml_int8x16x4_t {  // 类型定义
    int8x16_t val[4];
} ggml_int8x16x4_t;

inline static ggml_int8x16x4_t ggml_vec_xl_s8x4(const int8_t * ptr) {
    ggml_int8x16x4_t res;

    res.val[0] = vec_xl( 0, ptr);
    res.val[1] = vec_xl(16, ptr);
    res.val[2] = vec_xl(32, ptr);
    res.val[3] = vec_xl(48, ptr);

    return res;  // 返回
}

typedef struct ggml_int16x8x2_t {  // 类型定义
    int16x8_t val[2];
} ggml_int16x8x2_t;

inline static ggml_int16x8x2_t ggml_vec_xl_s16x2(const int16_t * ptr) {
    ggml_int16x8x2_t res;

    res.val[0] = vec_xl( 0, ptr);
    res.val[1] = vec_xl(16, ptr);

    return res;  // 返回
}

/*
    ! WARNING: Very slow. Use vec_perm if possible. Refer to iq4_xs
    !          or iq4_nl for example implementation.
*/
inline static int8x16_t ggml_vec_tbl(int8x16_t a, uint8x16_t b) {
    int8x16_t res;

    res[ 0] = a[b[ 0]];
    res[ 1] = a[b[ 1]];
    res[ 2] = a[b[ 2]];
    res[ 3] = a[b[ 3]];
    res[ 4] = a[b[ 4]];
    res[ 5] = a[b[ 5]];
    res[ 6] = a[b[ 6]];
    res[ 7] = a[b[ 7]];
    res[ 8] = a[b[ 8]];
    res[ 9] = a[b[ 9]];
    res[10] = a[b[10]];
    res[11] = a[b[11]];
    res[12] = a[b[12]];
    res[13] = a[b[13]];
    res[14] = a[b[14]];
    res[15] = a[b[15]];

    return res;  // 返回
}

inline static int16x8_t vec_padd_s16(int16x8_t a, int16x8_t b) {
    const uchar8x16_t v_maske = {  0,  1,  4,  5,  8,  9, 12, 13,
                                  16, 17, 20, 21, 24, 25, 28, 29 };

    const int16x8_t v_abo = vec_pack((int32x4_t)a, (int32x4_t)b);
    const int16x8_t v_abe = vec_perm(a, b, v_maske);
    return v_abo + v_abe;  // 返回
}

/**
 * @see https://github.com/ggml-org/llama.cpp/pull/14037
 */
inline static float vec_hsum_f32x4(float32x4_t v) {
    float32x4_t v_temp = v + vec_reve(v);
    return v_temp[0] + v_temp[1];  // 返回
}

inline static int32_t vec_hsum_i32x4(int32x4_t v) {
    int32x4_t v_temp = v + vec_reve(v);
    return v_temp[0] + v_temp[1];  // 返回
}

inline static int32x4_t ggml_vec_dot(int32x4_t acc, int8x16_t a, int8x16_t b) {
    const int16x8_t p = vec_mule(a, b) + vec_mulo(a, b);
    return acc + (vec_unpackh(p) + vec_unpackl(p));
}

#endif  // 条件编译结束

#if defined(__loongarch_sx)  // 条件编译
/* float type data load instructions */
static __m128 __lsx_vreplfr2vr_s(const float val) {
    v4f32 res = {val, val, val, val};
    return (__m128)res;
}
#endif  // 条件编译结束

#if defined(__loongarch_asx)  // 条件编译
static __m256 __lasx_xvreplfr2vr_s(const float val) {
    v8f32 res = {val, val, val, val, val, val, val, val};
    return (__m256)res;
}
#endif  // 条件编译结束

// TODO: move to ggml-threading
void ggml_barrier(struct ggml_threadpool * tp);  // ggml_barrier

void ggml_threadpool_chunk_set(struct ggml_threadpool * tp, int value);  // ggml_threadpool_chunk_set
int  ggml_threadpool_chunk_add(struct ggml_threadpool * tp, int value);  // ggml_threadpool_chunk_add

#ifdef __cplusplus  // 如果定义了 __cplusplus 则编译
}
#endif  // 条件编译结束
