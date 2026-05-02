//
// MIT license
// Copyright (C) 2024 Intel Corporation
// SPDX-License-Identifier: MIT
//

//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//

#ifndef GGML_SYCL_COMMON_HPP  // 如果未定义 GGML_SYCL_COMMON_HPP 则编译
#define GGML_SYCL_COMMON_HPP  // 宏定义 GGML_SYCL_COMMON_HPP

#include <cstddef>  // 引入 cstddef 头文件
#include <fstream>  // 引入 fstream 头文件
#include <iostream>  // 引入 iostream 头文件
#include <string>  // 引入 string 头文件

#include "dpct/helper.hpp"  // 引入 dpct/helper.hpp 头文件
#include "ggml.h"  // 引入 ggml.h 头文件
#include "ggml-impl.h"  // 引入 ggml-impl.h 头文件
#include "ggml-sycl.h"  // 引入 ggml-sycl.h 头文件
#include "presets.hpp"  // 引入 presets.hpp 头文件
#include "type.hpp"  // 引入 type.hpp 头文件
#include "sycl_hw.hpp"  // 引入 sycl_hw.hpp 头文件

namespace syclexp = sycl::ext::oneapi::experimental;  // 命名空间

#if defined(__INTEL_LLVM_COMPILER) && __has_include(<sycl/ext/oneapi/bfloat16.hpp>)  // 条件编译
    #include <sycl/ext/oneapi/bfloat16.hpp>  // 引入 sycl/ext/oneapi/bfloat16.hpp 头文件
    #ifndef GGML_SYCL_HAS_BF16  // 如果未定义 GGML_SYCL_HAS_BF16 则编译
        #define GGML_SYCL_HAS_BF16  // 宏定义 GGML_SYCL_HAS_BF16
    #endif  // 条件编译结束
#endif  // 条件编译结束

#if GGML_SYCL_DNNL  // 条件编译
#include "dnnl.hpp"  // 引入 dnnl.hpp 头文件
#include "dnnl_sycl.hpp"  // 引入 dnnl_sycl.hpp 头文件
#endif  // 条件编译结束

#define GGML_COMMON_DECL_SYCL  // 宏定义 GGML_COMMON_DECL_SYCL
#define GGML_COMMON_IMPL_SYCL  // 宏定义 GGML_COMMON_IMPL_SYCL
#define SYCL_FLASH_ATTN //remove it to disable FLASH_ATTENTION in building.  // 宏定义 SYCL_FLASH_ATTN
#define SYCL_FAST_FP16  //don't change. remove it will break fattn-tile.hpp building  // 宏定义 SYCL_FAST_FP16

/* suppress warning spam */
#pragma clang diagnostic push
#pragma clang diagnostic ignored "-Wnested-anon-types"
#include "ggml-common.h"  // 引入 ggml-common.h 头文件
#pragma clang diagnostic pop
#include "ggml-impl.h"  // 引入 ggml-impl.h 头文件

void* ggml_sycl_host_malloc(size_t size);  // ggml_sycl_host_malloc
void ggml_sycl_host_free(void* ptr);  // ggml_sycl_host_free


extern int g_ggml_sycl_debug;
extern int g_ggml_sycl_disable_optimize;
extern int g_ggml_sycl_prioritize_dmmv;
extern int g_ggml_sycl_enable_flash_attention;


#if defined(__clang__) && __has_builtin(__builtin_expect)  // 条件编译
// Hint the optimizer to pipeline the more likely following instruction in branches
#    define LIKELY(expr)   __builtin_expect(expr, true)  // 宏定义 LIKELY
#    define UNLIKELY(expr) __builtin_expect(expr, false)  // 宏定义 UNLIKELY
#else  // 否则
#    define LIKELY(expr)   (expr)  // 宏定义 LIKELY
#    define UNLIKELY(expr) (expr)  // 宏定义 UNLIKELY
#endif  // 条件编译结束

#define GGML_SYCL_DEBUG(...)              \
    do {                                  \
        if (UNLIKELY(g_ggml_sycl_debug))  \
            fprintf(stderr, __VA_ARGS__); \
    } while (0)

#define CHECK_TRY_ERROR(expr)                                            \
  [&]() {                                                                \
    try {                                                                \
      expr;                                                              \
      return dpct::success;                                              \
    } catch (std::exception const& e) {                                  \
      std::cerr << e.what() << "\nException caught at file:" << __FILE__ \
                << ", line:" << __LINE__ << ", func:" << __func__        \
                << std::endl;                                            \
      return dpct::default_error;                                        \
    }                                                                    \
  }()


#define __SYCL_ARCH__ DPCT_COMPATIBILITY_TEMP  // 宏定义 __SYCL_ARCH__
#define VER_4VEC 610 // todo for hardware optimize.  // 宏定义 VER_4VEC
#define VER_GEN9 700 // todo for hardware optimize.  // 宏定义 VER_GEN9
#define VER_GEN12 1000000 // todo for hardware optimize.  // 宏定义 VER_GEN12
#define VER_GEN13 (VER_GEN12 + 1030) // todo for hardware optimize.  // 宏定义 VER_GEN13

#define GGML_SYCL_MAX_NODES 8192 // TODO: adapt to hardwares  // 宏定义 GGML_SYCL_MAX_NODES

// define for XMX in Intel GPU
// TODO: currently, it's not used for XMX really.
#if !defined(GGML_SYCL_FORCE_MMQ)  // 条件编译
    #define SYCL_USE_XMX  // 宏定义 SYCL_USE_XMX
#endif  // 条件编译结束

// max batch size to use MMQ kernels when tensor cores are available
#define MMQ_MAX_BATCH_SIZE 32  // 宏定义 MMQ_MAX_BATCH_SIZE

// dmmv = dequantize_mul_mat_vec
#ifndef GGML_SYCL_DMMV_X  // 如果未定义 GGML_SYCL_DMMV_X 则编译
#define GGML_SYCL_DMMV_X 32  // 宏定义 GGML_SYCL_DMMV_X
#endif  // 条件编译结束
#ifndef GGML_SYCL_MMV_Y  // 如果未定义 GGML_SYCL_MMV_Y 则编译
#define GGML_SYCL_MMV_Y 1  // 宏定义 GGML_SYCL_MMV_Y
#endif  // 条件编译结束

typedef sycl::queue *queue_ptr;  // 类型定义

enum ggml_sycl_backend_gpu_mode {  // 枚举定义
  SYCL_UNSET_GPU_MODE = -1,
  SYCL_SINGLE_GPU_MODE = 0,
  SYCL_MUL_GPU_MODE
};

static_assert(sizeof(sycl::half) == sizeof(ggml_fp16_t), "wrong fp16 size");

static void crash() {
  int* ptr = NULL;
  *ptr = 0;
}

[[noreturn]] static void ggml_sycl_error(
    const char* stmt,
    const char* func,
    const char* file,
    const int line,
    const char* msg) {
  fprintf(stderr, "SYCL error: %s: %s\n", stmt, msg);
  fprintf(stderr, "  in function %s at %s:%d\n", func, file, line);
  GGML_ABORT("SYCL error");
}

#define SYCL_CHECK(err)                                                                                    \
    do {                                                                                                   \
        auto err_ = (err);                                                                                 \
        if (err_ != 0)                                                                                     \
            ggml_sycl_error(#err, __func__, __FILE__, __LINE__, "Exception caught in this line of code."); \
    } while (0)

#if DPCT_COMPAT_RT_VERSION >= 11100  // 条件编译
#define GGML_SYCL_ASSUME(x) __builtin_assume(x)  // 宏定义 GGML_SYCL_ASSUME
#else  // 否则
#define GGML_SYCL_ASSUME(x)  // 宏定义 GGML_SYCL_ASSUME
#endif // DPCT_COMPAT_RT_VERSION >= 11100  // 条件编译结束

#ifdef GGML_SYCL_F16  // 如果定义了 GGML_SYCL_F16 则编译
typedef sycl::half dfloat; // dequantize float  // 类型定义
typedef sycl::half2 dfloat2;  // 类型定义
#else  // 否则
typedef float dfloat; // dequantize float  // 类型定义
typedef sycl::float2 dfloat2;  // 类型定义
#endif // GGML_SYCL_F16  // 条件编译结束

#define MMVQ_MAX_BATCH_SIZE  8  // 宏定义 MMVQ_MAX_BATCH_SIZE

static int g_all_sycl_device_count = -1;
static bool g_ggml_backend_sycl_buffer_type_initialized = false;

static ggml_sycl_backend_gpu_mode g_ggml_sycl_backend_gpu_mode =
    SYCL_UNSET_GPU_MODE;

static void* g_scratch_buffer = nullptr;
static size_t g_scratch_size = 0; // disabled by default
static size_t g_scratch_offset = 0;

[[noreturn]] static inline void bad_arch(const sycl::stream& stream_ct1) {
  stream_ct1 << "ERROR: ggml-sycl was compiled without support for the "
                "current GPU architecture.\n";
  // __trap();
  std::exit(1);

  (void)bad_arch; // suppress unused function warning
}

int get_current_device_id();  // get_current_device_id

inline int ggml_sycl_get_device() {
    return get_current_device_id();  // get_current_device_id
}

inline dpct::err0 ggml_sycl_set_device(const int device) try {
  int current_device_id;
  SYCL_CHECK(CHECK_TRY_ERROR(current_device_id = get_current_device_id()));

  // GGML_SYCL_DEBUG("ggml_sycl_set_device device_id=%d,
  // current_device_id=%d\n", device, current_device);
  if (device == current_device_id) {
    return 0;  // 返回
  }

  return CHECK_TRY_ERROR(dpct::select_device(device));  // CHECK_TRY_ERROR
} catch (sycl::exception const& exc) {
  std::cerr << exc.what() << "Exception caught at file:" << __FILE__
            << ", line:" << __LINE__ << std::endl;
  crash();
  std::exit(1);
}

//////////////////////
struct optimize_feature {  // 结构体定义
    bool reorder=false;
};

struct sycl_device_info {  // 结构体定义
    int cc;  // compute capability
    int nsm; // number of streaming multiprocessors (CUDA) maps to the maximum
             // number of compute units on a SYCL device.
    // size_t  smpb;               // max. shared memory per block
    size_t  smpbo;              // max. shared memory per block (with opt-in)
    int warp_size;     // WARP_SIZE(16)|WARP_32_SIZE(32)|WARP_16_SIZE(16). For Intel GPU, 16 is better in most cases. Some OP support 32 only.
    int max_wg_per_cu; // max work groups per compute unit - refer to
                       // cudaOccupancyMaxActiveBlocksPerMultiprocessor
    bool    vmm;                // virtual memory support
    size_t  total_vram;
    sycl_hw_info hw_info;
    optimize_feature opt_feature;
};


struct ggml_sycl_device_info {  // 结构体定义
    int device_count;

    sycl_device_info devices[GGML_SYCL_MAX_DEVICES] = {};

    std::array<float, GGML_SYCL_MAX_DEVICES> default_tensor_split = {};

    int max_work_group_sizes[GGML_SYCL_MAX_DEVICES] = {0};
};

const ggml_sycl_device_info & ggml_sycl_info();

struct ggml_sycl_pool {  // 结构体定义
    virtual ~ggml_sycl_pool() = default;

    virtual void * alloc(size_t size, size_t * actual_size) = 0;
    virtual void free(void * ptr, size_t size) = 0;
};

template<typename T>  // 模板
struct ggml_sycl_pool_alloc {  // 结构体定义
    ggml_sycl_pool * pool = nullptr;
    T * ptr = nullptr;
    size_t actual_size = 0;

    explicit ggml_sycl_pool_alloc(ggml_sycl_pool & pool) : pool(&pool) {
    }

    ggml_sycl_pool_alloc(ggml_sycl_pool & pool, size_t size) : pool(&pool) {
        alloc(size);
    }

    ~ggml_sycl_pool_alloc() {
        if (ptr != nullptr) {
            pool->free(ptr, actual_size);
        }
    }

    T * realloc(size_t size) {
        GGML_ASSERT(pool != nullptr);
        if (ptr)
            pool->free(ptr, actual_size);
        ptr = (T *) pool->alloc(size * sizeof(T), &this->actual_size);
        return ptr;  // 返回
    }

    // size is in number of elements
    T * alloc(size_t size) {
        GGML_ASSERT(pool != nullptr);
        GGML_ASSERT(ptr == nullptr);
        ptr = (T *) pool->alloc(size * sizeof(T), &this->actual_size);
        return ptr;  // 返回
    }

    T * alloc(ggml_sycl_pool & pool, size_t size) {
        this->pool = &pool;
        return alloc(size);  // alloc
    }

    T * get() {
        return ptr;  // 返回
    }

    ggml_sycl_pool_alloc() = default;
    ggml_sycl_pool_alloc(const ggml_sycl_pool_alloc &) = delete;
    ggml_sycl_pool_alloc(ggml_sycl_pool_alloc &&) = delete;
    ggml_sycl_pool_alloc& operator=(const ggml_sycl_pool_alloc &) = delete;
    ggml_sycl_pool_alloc& operator=(ggml_sycl_pool_alloc &&) = delete;
};

// backend interface

struct ggml_tensor_extra_gpu {  // 结构体定义
  void* data_device[GGML_SYCL_MAX_DEVICES]; // 1 pointer for each device for split
                                       // tensors
  dpct::event_ptr events[GGML_SYCL_MAX_DEVICES]
                        [GGML_SYCL_MAX_STREAMS]; // events for synchronizing multiple GPUs
  optimize_feature optimized_feature;
};

void release_extra_gpu(ggml_tensor_extra_gpu * extra, std::vector<queue_ptr> streams={});

namespace sycl_ex = sycl::ext::oneapi::experimental;  // 命名空间
struct ggml_backend_sycl_context {  // 结构体定义
    int device;
    std::string name;
    optimize_feature opt_feature;

    queue_ptr qptrs[GGML_SYCL_MAX_DEVICES][GGML_SYCL_MAX_STREAMS] = { { nullptr } };

    explicit ggml_backend_sycl_context(int device) :  // ggml_backend_sycl_context
        device(device),
        name(GGML_SYCL_NAME + std::to_string(device)) {
        opt_feature = ggml_sycl_info().devices[device].opt_feature;
    }

    queue_ptr stream(int device, int stream) {
        if (qptrs[device][stream] == nullptr) {
            qptrs[device][stream] = &(dpct::get_device(device).default_queue());
        }
        return qptrs[device][stream];  // 返回
    }

    queue_ptr stream() {
        return stream(device, 0);  // stream
    }

#if GGML_SYCL_DNNL  // 条件编译
    dnnl::engine make_engine(sycl::queue* q) {
        // Get the device associated with the queue
        sycl::device dev = q->get_device();
        // Get the context associated with the queue
        sycl::context ctx = q->get_context();
        const dnnl::engine eng = dnnl::sycl_interop::make_engine(dev, ctx);
        return eng;  // 返回
    }

    std::unordered_map<sycl::queue*, dnnl::stream> stream_map;
    std::unordered_map<sycl::queue*, dnnl::engine> engine_map;
    dnnl::stream stream_dnnl(int device, int _stream) {
        auto q = stream(device, _stream);
        return stream_dnnl(q);  // stream_dnnl
    }
    dnnl::engine engine_dnnl(sycl::queue* qptr) {
        auto it = engine_map.find(qptr);
        if (it == engine_map.end()) {
            auto eng = make_engine(qptr);
            engine_map[qptr] = eng;
            return eng;  // 返回
        }
        else
        {
            return it->second;  // 返回
        }
    }
    dnnl::stream stream_dnnl(sycl::queue* qptr) {
        auto it = stream_map.find(qptr);
        if (it == stream_map.end()) {
            auto eng = engine_dnnl(qptr);
            auto stream = dnnl::sycl_interop::make_stream(eng, *qptr);
            stream_map[qptr] = stream;
            return stream;  // 返回
        }
        else
        {
            return it->second;  // 返回
        }
    }
    dnnl::stream stream_dnnl() {
        return stream_dnnl(device, 0);  // stream_dnnl
    }
    dnnl::memory get_scratchpad_mem(const dnnl::memory::desc & scratchpad_md,
                                    const dnnl::engine & eng, const queue_ptr q) {
        ggml_sycl_pool_alloc<uint8_t> * pool;
        auto it = scratchpad_map.find(q);
        if (it == scratchpad_map.end()) {
            scratchpad_map[q] = std::make_unique<ggml_sycl_pool_alloc<uint8_t>>(this->pool());
            pool = scratchpad_map[q].get();
        } else {
            pool = it->second.get();
        }

        size_t scratchpad_size = scratchpad_md.get_size();
        if (scratchpad_size > pool->actual_size) {
            pool->realloc(scratchpad_size);
        }
        void * mem_ptr = pool->get();
        return dnnl::memory(scratchpad_md, eng, mem_ptr);
    }
#endif  // 条件编译结束

    // pool
    std::unique_ptr<ggml_sycl_pool> pools[GGML_SYCL_MAX_DEVICES];
    std::unordered_map<sycl::queue *, std::unique_ptr<ggml_sycl_pool_alloc<uint8_t>>> scratchpad_map;

    std::unique_ptr<ggml_sycl_pool> host_pools[GGML_SYCL_MAX_DEVICES];

    static std::unique_ptr<ggml_sycl_pool> new_pool_for_device(queue_ptr qptr, int device);

    static std::unique_ptr<ggml_sycl_pool> new_pool_for_host(queue_ptr qptr, int device);

    ggml_sycl_pool & pool(int device) {
        if (pools[device] == nullptr) {
            pools[device] = new_pool_for_device(stream(device,0), device);
        }
        return *pools[device];  // 返回
    }

    ggml_sycl_pool & pool() {
        return pool(device);  // pool
    }

#ifdef GGML_SYCL_GRAPH  // 如果定义了 GGML_SYCL_GRAPH 则编译
    std::unique_ptr<sycl_ex::command_graph<sycl_ex::graph_state::executable>> exec_graph = nullptr;
#endif  // 条件编译结束

    ggml_sycl_pool & host_pool(int device) {
        if (host_pools[device] == nullptr) {
            host_pools[device] = new_pool_for_host(stream(device, 0), device);
        }
        return *host_pools[device];  // 返回
    }

    ggml_sycl_pool & host_pool() { return host_pool(device); }
};

// common device functions

static __dpct_inline__ float warp_reduce_sum(float x,
    const sycl::nd_item<3>& item_ct1) {
#pragma unroll
    for (int mask = WARP_SIZE / 2; mask > 0; mask >>= 1) {
        x += dpct::permute_sub_group_by_xor(item_ct1.get_sub_group(), x, mask);
    }
    return x;  // 返回
}

static __dpct_inline__ sycl::float2
warp_reduce_sum(sycl::float2 a, const sycl::nd_item<3>& item_ct1) {
#pragma unroll
    for (int mask = WARP_SIZE / 2; mask > 0; mask >>= 1) {
        a.x() += dpct::permute_sub_group_by_xor(item_ct1.get_sub_group(), a.x(),
            mask);
        a.y() += dpct::permute_sub_group_by_xor(item_ct1.get_sub_group(), a.y(),
            mask);
    }
    return a;  // 返回
}

/* use WARP_SIZE or WARP_32_SIZE*/
template <int width>  // 模板
static __dpct_inline__ int warp_reduce_sum(int x) {
  return sycl::reduce_over_group(  // 返回
      sycl::ext::oneapi::this_work_item::get_sub_group(), x, sycl::plus<>());
}

/* use WARP_SIZE or WARP_32_SIZE*/
template <int width>  // 模板
static __dpct_inline__ float warp_reduce_sum(float x) {
#pragma unroll
  for (int offset = width / 2; offset > 0; offset >>= 1) {
    x += dpct::permute_sub_group_by_xor(
        sycl::ext::oneapi::this_work_item::get_sub_group(), x, offset, width);
  }
  return x;  // 返回
}

/* use WARP_SIZE or WARP_32_SIZE*/
template <int width>  // 模板
static __dpct_inline__ float warp_reduce_sum(float x, const sycl::nd_item<3>& item_ct1) {
#pragma unroll
  for (int offset = width / 2; offset > 0; offset >>= 1) {
    x += dpct::permute_sub_group_by_xor(
        item_ct1.get_sub_group(), x, offset);
  }
  return x;  // 返回
}

/* use WARP_SIZE or WARP_32_SIZE*/
template <int width>  // 模板
static __dpct_inline__ sycl::float2 warp_reduce_sum(sycl::float2 a) {
#pragma unroll
  for (int offset = width / 2; offset > 0; offset >>= 1) {
    a.x() += dpct::permute_sub_group_by_xor(
        sycl::ext::oneapi::this_work_item::get_sub_group(), a.x(), offset,
        width);
    a.y() += dpct::permute_sub_group_by_xor(
        sycl::ext::oneapi::this_work_item::get_sub_group(), a.y(), offset,
        width);
  }
  return a;  // 返回
}

/* use WARP_SIZE or WARP_32_SIZE*/
template <int width>  // 模板
static __dpct_inline__ sycl::half2 warp_reduce_sum(sycl::half2 a) {
#pragma unroll
  for (int offset = width / 2; offset > 0; offset >>= 1) {
    a = a + dpct::permute_sub_group_by_xor(
                sycl::ext::oneapi::this_work_item::get_sub_group(), a, offset,
                width);
  }
  return a;  // 返回
}

static constexpr int ggml_sycl_get_physical_warp_size() {
  // todo: for old iGPU + dGPU case, need to be changed.
  return WARP_SIZE;  // 返回
}

/* use WARP_SIZE or WARP_32_SIZE*/
template <int width>  // 模板
static __dpct_inline__ int warp_reduce_all(int x) {
    if (width == ggml_sycl_get_physical_warp_size()) {
        return sycl::all_of_group(  // 返回
            sycl::ext::oneapi::this_work_item::get_sub_group(),
            (~0xffffffff &
             (0x1 << sycl::ext::oneapi::this_work_item::get_sub_group()
                         .get_local_linear_id())) ||
                x);
    } else {
#pragma unroll
        for (int offset = width / 2; offset > 0; offset >>= 1) {
            x = dpct::permute_sub_group_by_xor(
                    sycl::ext::oneapi::this_work_item::get_sub_group(), x,
                    offset, width) &&
                x;
        }
        return x;  // 返回
    }
}

/* use WARP_SIZE or WARP_32_SIZE*/
template <int width>  // 模板
static __dpct_inline__ int warp_reduce_any(int x) {
    if (width == ggml_sycl_get_physical_warp_size()) {
        return sycl::any_of_group(  // 返回
            sycl::ext::oneapi::this_work_item::get_sub_group(),
            (0xffffffff &
             (0x1 << sycl::ext::oneapi::this_work_item::get_sub_group()
                         .get_local_linear_id())) &&
                x);
    } else {
#pragma unroll
        for (int offset = width / 2; offset > 0; offset >>= 1) {
            x = dpct::permute_sub_group_by_xor(
                    sycl::ext::oneapi::this_work_item::get_sub_group(), x,
                    offset, width) ||
                x;
        }
        return x;  // 返回
    }
}

/* use WARP_SIZE or WARP_32_SIZE*/
template <int width>  // 模板
static __dpct_inline__ float warp_reduce_max(float x) {
#pragma unroll
  for (int offset = width / 2; offset > 0; offset >>= 1) {
    x = sycl::fmax(x, dpct::permute_sub_group_by_xor(
                          sycl::ext::oneapi::this_work_item::get_sub_group(), x,
                          offset, width));
  }
  return x;  // 返回
}

static __dpct_inline__ float warp_reduce_max(float x,
    const sycl::nd_item<3>& item_ct1) {
#pragma unroll
    for (int mask = WARP_SIZE / 2; mask > 0; mask >>= 1) {
        x = sycl::fmax(x, dpct::permute_sub_group_by_xor(
            item_ct1.get_sub_group(), x, mask));
    }
    return x;  // 返回
}

/* Helper for Computing the linear offset of a ggml_tensor given
per-dimension sizes, strides, and indices */
template<int N>  // 模板
__dpct_inline__ size_t calculate_offset(const std::array<int, N> & strides, const std::array<int, N> & indices) {
    size_t offset = 0;
#pragma unroll
    for (int i = 0; i < N; i++) {
        auto index_i = indices[i];
        offset += strides[i] * index_i;
    }
    return offset;  // 返回
}

// Helper for vec loading aligned data
template <typename Tp, int n>  // 模板
inline sycl::vec<Tp, n> vec_aligned_load(const Tp* aligned_ptr) {
    return *reinterpret_cast<const sycl::vec<Tp, n>*>(aligned_ptr);
}

// Helper for accessing pointers with no warnings
template <typename Tp, int dim>  // 模板
static __dpct_inline__ Tp* get_pointer(sycl::local_accessor<Tp, dim> acc) {
    return acc.template get_multi_ptr<sycl::access::decorated::no>().get();
}

int64_t downsample_sycl_global_range(int64_t accumulate_block_num, int64_t block_size);  // downsample_sycl_global_range

constexpr size_t ceil_div(const size_t m, const size_t n) {
    return (m + n - 1) / n;
}

bool gpu_has_xmx(sycl::device &dev);  // gpu_has_xmx

template <int N, class T> std::string debug_get_array_str(const std::string & prefix, const T array[N]) {  // 模板
    if (LIKELY(!g_ggml_sycl_debug)) {
        return "";  // 返回
    }
    std::stringstream ss;
    ss << prefix << "=[";
    for (std::size_t i = 0; i < N - 1; ++i) {
        ss << array[i] << ", ";
    }
    if constexpr (N > 0) {
        ss << array[N - 1];
    }
    ss << "]";
    return ss.str();
}

inline std::string debug_get_tensor_str(const std::string &prefix,
        const ggml_tensor *tensor, const std::string &suffix = "") {
    std::stringstream ss;
    if (LIKELY(!g_ggml_sycl_debug)) { return ss.str(); }
    ss << prefix.c_str() << "=";
    if (tensor) {
        ss << "'" << tensor->name << "':type=" << ggml_type_name(tensor->type);
        ss << debug_get_array_str<GGML_MAX_DIMS>(";ne", tensor->ne);
        ss << debug_get_array_str<GGML_MAX_DIMS>(";nb", tensor->nb);

        if (!ggml_is_contiguous(tensor)) { ss << ";strided"; }
        if (ggml_is_permuted(tensor)) { ss << ";permuted"; }
    } else {
        ss << "nullptr";
    }
    ss << suffix;
    return ss.str();
}

// Use scope_op_debug_print to log operations coming from running a model
struct scope_op_debug_print {  // 结构体定义
    // Use string_views to avoid the cost of creating a string and concatenating them
    // string_views must be alive for as long as the object is alive
    // scope_op_debug_print are used with string literals in practice which are stored in constant space so always accessible
    scope_op_debug_print(const std::string_view & func, const std::string_view & func_suffix, const ggml_tensor * dst,
                         std::size_t num_src, const std::string_view & suffix = "") :
        func(func),
        func_suffix(func_suffix) {
        if (LIKELY(!g_ggml_sycl_debug)) {
            return;  // 返回
        }
        GGML_SYCL_DEBUG("[SYCL][OP] call %s%s:", func.data(), func_suffix.data());
        GGML_SYCL_DEBUG("%s", debug_get_tensor_str(" dst", dst).c_str());
        if (dst) {
            for (std::size_t i = 0; i < num_src; ++i) {
                GGML_SYCL_DEBUG("%s", debug_get_tensor_str("\tsrc" + std::to_string(i), dst->src[i]).c_str());
            }
        }
        GGML_SYCL_DEBUG("%s\n", suffix.data());
    }

    scope_op_debug_print(const std::string_view & func, const ggml_tensor * dst, std::size_t num_src,
                         const std::string_view & suffix = "") :
        scope_op_debug_print(func, "", dst, num_src, suffix) {}

    ~scope_op_debug_print() { GGML_SYCL_DEBUG("[SYCL][OP] call %s%s done\n", func.data(), func_suffix.data()); }

  private:
    std::string_view func;
    std::string_view func_suffix;
};

static __dpct_inline__ float get_alibi_slope(const float    max_bias,
                                             const uint32_t h,
                                             const uint32_t n_head_log2,
                                             const float    m0,
                                             const float    m1) {
    if (max_bias <= 0.0f) {
        return 1.0f;  // 返回
    }
    const float base = h < n_head_log2 ? m0 : m1;
    const int   exph = h < n_head_log2 ? h + 1 : 2*(h - n_head_log2) + 1;

    return dpct::pow(base, exph);
}

static const sycl::uint3 init_fastdiv_values(uint32_t d) {
    GGML_ASSERT(d != 0);

    uint32_t L = 0;
    while (L < 32 && (uint32_t{ 1 } << L) < d) {
        L++;
    }

    uint32_t mp = (uint32_t) ((uint64_t{ 1 } << 32) * ((uint64_t{ 1 } << L) - d) / d + 1);
    return sycl::uint3(mp, L, d);
}

// Maximum number of bytes that can be copied in a single instruction.
// Set by test result.
static constexpr int ggml_sycl_get_max_cpy_bytes() {
    return 16;  // 返回
}

// Aligned memory transfers of 8/16 bytes can be faster than 2 transfers with 4 bytes.
template <int nbytes, int alignment = 0>  // 模板
static __dpct_inline__ void ggml_sycl_memcpy_1(void * dst, const void * src) {
    if constexpr (alignment != 0) {
        static_assert(nbytes % alignment == 0, "bad alignment");
    }
    constexpr int nb_per_cpy = alignment == 0 ? nbytes : alignment;

#pragma unroll
    for (int i = 0; i < nbytes/nb_per_cpy; ++i) {
        if constexpr (nb_per_cpy == 1) {
            ((char *) dst)[i] = ((const char *) src)[i];
        } else if constexpr (nb_per_cpy == 2) {
            ((short *) dst)[i] = ((const short *) src)[i];
        } else if constexpr (nb_per_cpy == 4) {
            ((int *) dst)[i] = ((const int *) src)[i];
        } else if constexpr (nb_per_cpy == 8) {
            ((sycl::int2 *) dst)[i] = ((const sycl::int2 *) src)[i];
        } else if constexpr (nb_per_cpy == 16) {
            ((sycl::int4 *) dst)[i] = ((const sycl::int4 *) src)[i];
        } else {
            static_assert(nbytes == 0 && nbytes == -1, "bad nbytes");
        }
    }
}
template <typename T>  // 模板
sycl::half2 __dpct_inline__ make_half2( T x, T y) {
    sycl::half2 res(static_cast<sycl::half>(x),static_cast<sycl::half>(y));
    return res;  // 返回
}

static __dpct_inline__ uint32_t fastdiv(uint32_t n, const sycl::uint3 fastdiv_values) {
    const uint32_t hi = sycl::mul_hi<unsigned>(n, fastdiv_values.x());
    return (hi + n) >> fastdiv_values.y();
}


template <typename T>  // 模板
sycl::float2 __dpct_inline__ make_float2( T x, T y) {
    sycl::float2 res(static_cast<float>(x),static_cast<float>(y));
    return res;  // 返回
}

sycl::float2 __dpct_inline__ __half22float2(sycl::half2 &H) {
    sycl::float2 float2_value(static_cast<float>(H.x()), static_cast<float>(H.y()));
    return float2_value;  // 返回
}

static __dpct_inline__ sycl::uint2 fast_div_modulo(uint32_t n, const sycl::uint3 fastdiv_values) {
    const uint32_t div_val = fastdiv(n, fastdiv_values);
    const uint32_t mod_val = n - div_val * fastdiv_values.z();
    return sycl::uint2(div_val, mod_val);
}

static __dpct_inline__ int ggml_sycl_dp4a(const int a, const int b, int c) {
    return dpct::dp4a(a, b, c);
}

static __dpct_inline__ float ggml_sycl_e8m0_to_fp32(uint8_t x) {
    uint32_t bits;
    if (x == 0) {
        bits = 0x00400000;
    } else {
        bits = (uint32_t) x << 23;
    }

    float result;
    memcpy(&result, &bits, sizeof(float));
    return result;  // 返回
}

sycl::float2 __dpct_inline__ __half22float2(const sycl::half2 &H) {
    sycl::float2 float2_value(static_cast<float>(H.x()), static_cast<float>(H.y()));
    return float2_value;  // 返回
}

float __dpct_inline__ __half2float(sycl::half H) {
    return static_cast<float>(H);
}

static __dpct_inline__ void ggml_sycl_mad(float & acc, const float v, const float u) {
    acc += v*u;
}

static __dpct_inline__ void ggml_sycl_mad(float & acc, const sycl::float2 v, const sycl::float2 u) {
    acc += v.x() * u.x();
    acc += v.y() * u.y();
}

static __dpct_inline__ void ggml_sycl_mad(float & acc, const sycl::half2 v, const sycl::half2 u) {
#ifdef GGML_SYCL_F16  // 如果定义了 GGML_SYCL_F16 则编译
    const sycl::float2 tmp = (v * u).template convert<float, sycl::rounding_mode::automatic>();
    acc += tmp.x() + tmp.y();
#else  // 否则
    const sycl::float2 tmpv = __half22float2(v);
    const sycl::float2 tmpu = __half22float2(u);
    acc += tmpv.x() * tmpu.x();
    acc += tmpv.y() * tmpu.y();
#endif // GGML_SYCL_F16  // 条件编译结束
}

static __dpct_inline__ void ggml_sycl_mad(sycl::half2 & acc, const sycl::half2 v, const sycl::half2 u) {
#ifdef GGML_SYCL_F16  // 如果定义了 GGML_SYCL_F16 则编译
    acc += v*u;
#else  // 否则
    const sycl::float2 tmpv = __half22float2(v);
    const sycl::float2 tmpu = __half22float2(u);
    sycl::float2 tmpacc = __half22float2(acc);
    // tmpacc.x += tmpv.x() * tmpu.x();
    // tmpacc.y += tmpv.y() * tmpu.y();
    sycl::float2 tmp1(tmpacc.x() + tmpv.x() * tmpu.x(), tmpacc.y() + tmpv.y() * tmpu.y());
    acc = make_half2(tmp1.x(), tmp1.y());
#endif // GGML_SYCL_F16  // 条件编译结束
}

template <int n>  // 模板
struct ggml_sycl_unroll {  // 结构体定义
    template <typename Func, typename... Args>  // 模板
    void operator()(const Func & f, Args... args) const {
        f(n - 1, args...);
        ggml_sycl_unroll<n - 1>{}(f, args...);
    }
};

template <>  // 模板
struct ggml_sycl_unroll<1> {  // 结构体定义
    template <typename Func, typename... Args>  // 模板
    void operator()(const Func & f, Args... args) const {
        f(0, args...);
    }
};

static __dpct_inline__ sycl::half2 ggml_sycl_hmax2(const sycl::half2 a, const sycl::half2 b) {
    sycl::half2 ret;
    reinterpret_cast<sycl::half &>(ret.x()) =
        sycl::vec<float, 1>(sycl::fmax(a[0], b[0])).convert<sycl::half, sycl::rounding_mode::automatic>()[0];
    reinterpret_cast<sycl::half &>(ret.y()) =
        sycl::vec<float, 1>(sycl::fmax(a[1], b[1])).convert<sycl::half, sycl::rounding_mode::automatic>()[0];
    return ret;  // 返回
}

static __dpct_inline__ sycl::half ggml_sycl_hmax(const sycl::half a, const sycl::half b) {
    return sycl::vec<float, 1>(  // 返回
               sycl::fmax(sycl::vec<sycl::half, 1>(a).convert<float, sycl::rounding_mode::automatic>()[0],
                          sycl::vec<sycl::half, 1>(b).convert<float, sycl::rounding_mode::automatic>()[0]))
        .convert<sycl::half, sycl::rounding_mode::automatic>()[0];
}

static __dpct_inline__ uint32_t __hgt2_mask(const sycl::half2 a, const sycl::half2 b) {
    const uint32_t mask_low  = 0x0000FFFF * (float(a[0]) > float(b[0]));
    const uint32_t mask_high = 0xFFFF0000 * (float(a[1]) > float(b[1]));
    return mask_low | mask_high;  // 返回
}

static __dpct_inline__ uint32_t fastmodulo(uint32_t n, const sycl::uint3 fastdiv_values) {
    // expects  fastdiv_values to contain <mp, L, divisor> in <x, y, z> (see init_fastdiv_values)
    return n - fastdiv(n, fastdiv_values) * fastdiv_values.z();
}

static bool fast_fp16_available(const int cc) {
    GGML_UNUSED(cc);
    return true;   //Intel GPUs always support FP16.  // 返回
}

enum class block_reduce_method {  // 枚举定义
    MAX,
    SUM,
};

template<block_reduce_method method_t, typename T, int warp_size>  // 模板
struct block_reduce_policy;

template <typename T, typename... Ts>  // 模板
inline constexpr bool is_any = (std::is_same_v<T, Ts> || ...);

template<typename...>  // 模板
inline constexpr bool ggml_sycl_dependent_false_v = false;

#define WARP_32_SIZE 32  // 宏定义 WARP_32_SIZE

template <typename T, int warp_size> struct block_reduce_policy<block_reduce_method::SUM, T, warp_size> {  // 模板
    static T reduce(T val) {
        if constexpr (is_any<T, float, sycl::float2, sycl::half2, int>) {
            return warp_reduce_sum<warp_size>(val);
        } else {
            static_assert(ggml_sycl_dependent_false_v<T>, "Unsupported type for block reduce sum");
        }
    }

    static T sentinel() {
        if constexpr (std::is_same_v<T, float>) {
            return 0.0f;  // 返回
        } else if constexpr (std::is_same_v<T, sycl::float2>) {
            return sycl::float2(0.0f, 0.0f);
        } else if constexpr (std::is_same_v<T, sycl::half2>) {
            return sycl::half2(0.0f, 0.0f);
        } else if constexpr (std::is_same_v<T, int>) {
            return 0;  // 返回
        } else {
            static_assert(ggml_sycl_dependent_false_v<T>, "Unsupported type for block reduce sum");
        }
    }
};

template <typename T, int warp_size> struct block_reduce_policy<block_reduce_method::MAX, T, warp_size> {  // 模板
    static T reduce(T val) {
        if constexpr (is_any<T, float, sycl::half2>) {
            return warp_reduce_max<warp_size>(val);
        } else {
            static_assert(ggml_sycl_dependent_false_v<T>, "Unsupported type for block reduce max");
        }
    }

    static T sentinel() {
        if constexpr (std::is_same_v<T, float>) {
            return -INFINITY;  // 返回
        } else if constexpr (std::is_same_v<T, sycl::half2>) {
            return sycl::half2(-INFINITY, -INFINITY);
        } else {
            static_assert(ggml_sycl_dependent_false_v<T>, "Unsupported type for block reduce max");
        }
    }
};


template <block_reduce_method reduce_method_t, int warp_size, typename T>  // 模板
static T block_reduce(T val, T * shared_vals, int block_size_template) {
    auto item_ct1                 = sycl::ext::oneapi::this_work_item::get_nd_item<3>();
    val                           = block_reduce_policy<reduce_method_t, T,warp_size>::reduce(val);
    const int block_size = block_size_template == 0 ? item_ct1.get_local_range(2) : block_size_template;
    const int nthreads = item_ct1.get_local_range(2);
    const int nwarps = nthreads / WARP_SIZE;

    if (block_size > warp_size) {
        assert((block_size <= 1024) && (block_size % warp_size) == 0);
        const int warp_id = item_ct1.get_local_id(2) / warp_size;
        const int lane_id = item_ct1.get_local_id(2) % warp_size;
        if (lane_id == 0) {
            shared_vals[warp_id] = val;
        }
        item_ct1.barrier(sycl::access::fence_space::local_space);

        size_t nreduce = nwarps / WARP_SIZE;
        float tmp = 0.f;
        if (lane_id < (static_cast<int>(block_size) / warp_size)) {
            for (size_t i = 0; i < nreduce; i += 1)
            {
                tmp += shared_vals[lane_id + i * WARP_SIZE];
            }
        }
        return block_reduce_policy<reduce_method_t, T, warp_size>::reduce(tmp);
    }
    return val;  // 返回
}

static __dpct_inline__ float ggml_sycl_ue4m3_to_fp32(uint8_t x) {
    const uint32_t bits = x * (x != 0x7F && x != 0xFF);
    const __nv_fp8_e4m3 xf = *reinterpret_cast<const __nv_fp8_e4m3 *>(&bits);
    return static_cast<float>(xf) / 2;
}

#endif // GGML_SYCL_COMMON_HPP  // 条件编译结束
