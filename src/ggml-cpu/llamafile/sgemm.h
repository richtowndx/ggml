#pragma once  // 防止重复包含
#include <stdint.h>  // 引入 stdint.h 头文件
#include <stdbool.h>  // 引入 stdbool.h 头文件

#if defined(__VXE__) || defined(__VXE2__)  // 条件编译
#include <vecintrin.h>  // 引入 vecintrin.h 头文件
#endif  // 条件编译结束

#ifdef _MSC_VER  // 如果定义了 _MSC_VER 则编译
#define NOINLINE __declspec(noinline)  // 宏定义 NOINLINE
#else  // 否则
#define NOINLINE __attribute__((__noinline__))  // 宏定义 NOINLINE
#endif  // 条件编译结束

#ifdef __cplusplus  // 如果定义了 __cplusplus 则编译
extern "C" {  // C 链接声明
#endif  // 条件编译结束

bool llamafile_sgemm(const struct ggml_compute_params * params, int64_t, int64_t, int64_t,
                     const void *, int64_t, const void *, int64_t, void *, int64_t,
                     int, int, int);

#ifdef __cplusplus  // 如果定义了 __cplusplus 则编译
}
#endif  // 条件编译结束
