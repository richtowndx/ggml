#pragma once  // 防止重复包含

#include "ggml.h"  // 引入 ggml.h 头文件
#include "ggml-cpu-impl.h"  // 引入 ggml-cpu-impl.h 头文件

#include <algorithm>  // 引入 algorithm 头文件
#include <memory>  // 引入 memory 头文件
#include <type_traits>  // 引入 type_traits 头文件

#if defined(GGML_USE_OPENMP)  // 条件编译
#include <omp.h>  // 引入 omp.h 头文件
#else  // 否则
#include <thread>  // 引入 thread 头文件
#endif  // 条件编译结束

#define TILE_M 16  // 宏定义 TILE_M
#define TILE_N 16  // 宏定义 TILE_N
#define TILE_K 32  // 宏定义 TILE_K
#define VNNI_BLK 4  // 宏定义 VNNI_BLK

#define AMX_BLK_SIZE 32  // 宏定义 AMX_BLK_SIZE

#define TMM0 0  // 宏定义 TMM0
#define TMM1 1  // 宏定义 TMM1
#define TMM2 2  // 宏定义 TMM2
#define TMM3 3  // 宏定义 TMM3
#define TMM4 4  // 宏定义 TMM4
#define TMM5 5  // 宏定义 TMM5
#define TMM6 6  // 宏定义 TMM6
#define TMM7 7  // 宏定义 TMM7

// parallel routines
template <typename T, typename std::enable_if<std::is_integral<T>::value, int>::type = 0>  // 模板
inline T div_up(T x, T y) { return (x + y - 1) / y; }

template <typename T>  // 模板
inline void balance211(T n, T nth, T ith, T& n_start, T& n_end) {
#if 0  // 条件编译
    // onednn partition pattern
    T& n_my = n_end;
    if (nth <= 1 || n == 0) {
        n_start = 0;
        n_my = n;
    } else {
        T n1 = div_up(n, nth);
        T n2 = n1 - 1;
        T T1 = n - n2 * nth;
        n_my = ith < T1 ? n1 : n2;
        n_start = ith <= T1 ? ith*n1 : T1 * n1 + (ith - T1) * n2;
    }
    n_end += n_start;
#else  // 否则
    // pytorch aten partition pattern
    T n_my = div_up(n, nth);
    n_start = ith * n_my;
    n_end = std::min(n_start + n_my, n);
#endif  // 条件编译结束
}

template <typename func_t>  // 模板
inline void parallel_for(int n, const func_t & f) {
    if (n <= 0) {
        return;  // 返回
    }
#if defined(GGML_USE_OPENMP)  // 条件编译
    #pragma omp parallel
    {
        int nth = omp_get_num_threads();
        int ith = omp_get_thread_num();
        int tbegin, tend;
        balance211(n, nth, ith, tbegin, tend);
        f(tbegin, tend);
    }
#else  // 否则
    int nth = std::thread::hardware_concurrency();
    if (nth <= 1) {
        f(0, n);
        return;  // 返回
    }
    if (nth > n) {
        nth = n;
    }
    std::vector<std::thread> threads;
    threads.reserve(nth);
    for (int ith = 0; ith < nth; ++ith) {
        threads.emplace_back([&f, n, ith, nth] {
            int tbegin, tend;
            balance211(n, nth, ith, tbegin, tend);
            f(tbegin, tend);
        });
    }
    for (auto & t : threads) {
        t.join();
    }
#endif  // 条件编译结束
}

template <typename func_t>  // 模板
inline void parallel_for_ggml(const ggml_compute_params * params, int n, const func_t & f) {
    int tbegin, tend;
    balance211(n, params->nth, params->ith, tbegin, tend);
    f(tbegin, tend);
}

// quantized types that have AMX support
inline bool qtype_has_amx_kernels(const enum ggml_type type) {
    // TODO: fix padding for vnni format
    return (type == GGML_TYPE_Q4_0) ||
        (type == GGML_TYPE_Q4_1) ||
        (type == GGML_TYPE_Q8_0) ||
        (type == GGML_TYPE_Q4_K) ||
        (type == GGML_TYPE_Q5_K) ||
        (type == GGML_TYPE_Q6_K) ||
        (type == GGML_TYPE_IQ4_XS);
}
