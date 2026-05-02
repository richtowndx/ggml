#ifndef HEX_UTILS_H  // 如果未定义 HEX_UTILS_H 则编译
#define HEX_UTILS_H  // 宏定义 HEX_UTILS_H

#include <stdbool.h>  // 引入 stdbool.h 头文件
#include <stdint.h>  // 引入 stdint.h 头文件
#include <qurt_memory.h>  // 引入 qurt_memory.h 头文件
#include <qurt.h>  // 引入 qurt.h 头文件

#include "hexagon_types.h"  // 引入 hexagon_types.h 头文件
#include "hexagon_protos.h"  // 引入 hexagon_protos.h 头文件

#include "hex-fastdiv.h"  // 引入 hex-fastdiv.h 头文件
#include "hex-dump.h"  // 引入 hex-dump.h 头文件

#ifndef MAX  // 如果未定义 MAX 则编译
#define MAX(a, b) ((a) > (b) ? (a) : (b))  // 宏定义 MAX
#endif  // 条件编译结束

#ifndef MIN  // 如果未定义 MIN 则编译
#define MIN(a, b) ((a) < (b) ? (a) : (b))  // 宏定义 MIN
#endif  // 条件编译结束

static inline uint64_t hex_get_cycles() {
    uint64_t cycles = 0;
    asm volatile(" %0 = c15:14\n" : "=r"(cycles));  // volatile
    return cycles;  // 返回
}

static inline uint64_t hex_get_pktcnt() {
    uint64_t pktcnt;
    asm volatile(" %0 = c19:18\n" : "=r"(pktcnt));  // volatile
    return pktcnt;  // 返回
}

static inline uint32_t hex_ceil_pow2(uint32_t x) {
    if (x <= 1) { return 1; }
    int p = 2;
    x--;
    while (x >>= 1) { p <<= 1; }
    return p;  // 返回
}

static inline size_t hmx_ceil_div(size_t num, size_t den) {
    return (num + den - 1) / den;
}

static inline int32_t hex_is_aligned(const void * addr, uint32_t align) {
    return ((size_t) addr & (align - 1)) == 0;
}

static inline size_t hex_align_up(size_t v, size_t align) {
    return hmx_ceil_div(v, align) * align;  // hmx_ceil_div
}

static inline size_t hex_align_down(size_t v, size_t align) {
    return (v / align) * align;
}

static inline int32_t hex_is_one_chunk(void * addr, uint32_t n, uint32_t chunk_size) {
    uint32_t left_off  = (size_t) addr & (chunk_size - 1);
    uint32_t right_off = left_off + n;
    return right_off <= chunk_size;  // 返回
}

static inline uint32_t hex_round_up(uint32_t n, uint32_t m) {
    return m * ((n + m - 1) / m);
}

static inline size_t hex_smin(size_t a, size_t b) {
    return a < b ? a : b;  // 返回
}

static inline size_t hex_smax(size_t a, size_t b) {
    return a > b ? a : b;  // 返回
}

static inline void hex_swap_ptr(void ** p1, void ** p2) {
    void * t = *p1;
    *p1      = *p2;
    *p2      = t;
}

static inline void hex_l2fetch(const void * p, uint32_t width, uint32_t stride, uint32_t height) {
    const uint64_t control = Q6_P_combine_RR(stride, Q6_R_combine_RlRl(width, height));
    Q6_l2fetch_AP((void *) p, control);
}

#define HEX_L2_LINE_SIZE  64  // 宏定义 HEX_L2_LINE_SIZE
#define HEX_L2_FLUSH_SIZE (128 * 1024)  // 宏定义 HEX_L2_FLUSH_SIZE

static inline void hex_l2flush(void * addr, size_t size) {
    if (size > HEX_L2_FLUSH_SIZE) {
        qurt_mem_cache_clean((qurt_addr_t) 0, 0, QURT_MEM_CACHE_FLUSH_INVALIDATE_ALL, QURT_MEM_DCACHE);
    } else {
        const uint32_t s = (uint32_t) addr;
        const uint32_t e = s + size;
        for (uint32_t i = s; i < e; i += HEX_L2_LINE_SIZE * 4) {
            Q6_dccleaninva_A((void *) i + HEX_L2_LINE_SIZE * 0);
            Q6_dccleaninva_A((void *) i + HEX_L2_LINE_SIZE * 1);
            Q6_dccleaninva_A((void *) i + HEX_L2_LINE_SIZE * 2);
            Q6_dccleaninva_A((void *) i + HEX_L2_LINE_SIZE * 3);
        }
    }
}

static inline void hex_pause() {
    asm volatile(" pause(#255)\n");  // volatile
}

#ifndef HEX_NUM_PMU_COUNTERS  // 如果未定义 HEX_NUM_PMU_COUNTERS 则编译
#define HEX_NUM_PMU_COUNTERS 8  // 宏定义 HEX_NUM_PMU_COUNTERS
#endif  // 条件编译结束

static inline void hex_get_pmu(uint32_t counters[]) {
#if __HVX_ARCH__ >= 79  // 条件编译
    asm volatile("%0 = upmucnt0" : "=r"(counters[0]));  // volatile
    asm volatile("%0 = upmucnt1" : "=r"(counters[1]));  // volatile
    asm volatile("%0 = upmucnt2" : "=r"(counters[2]));  // volatile
    asm volatile("%0 = upmucnt3" : "=r"(counters[3]));  // volatile
    asm volatile("%0 = upmucnt4" : "=r"(counters[4]));  // volatile
    asm volatile("%0 = upmucnt5" : "=r"(counters[5]));  // volatile
    asm volatile("%0 = upmucnt6" : "=r"(counters[6]));  // volatile
    asm volatile("%0 = upmucnt7" : "=r"(counters[7]));  // volatile
#else  // 否则
    counters[0] = qurt_pmu_get(QURT_PMUCNT0);
    counters[1] = qurt_pmu_get(QURT_PMUCNT1);
    counters[2] = qurt_pmu_get(QURT_PMUCNT2);
    counters[3] = qurt_pmu_get(QURT_PMUCNT3);
    counters[4] = qurt_pmu_get(QURT_PMUCNT4);
    counters[5] = qurt_pmu_get(QURT_PMUCNT5);
    counters[6] = qurt_pmu_get(QURT_PMUCNT6);
    counters[7] = qurt_pmu_get(QURT_PMUCNT7);
    // qurt_pmu_get_pmucnt(counters);
#endif  // 条件编译结束
}

#endif /* HEX_UTILS_H */  // 条件编译结束
