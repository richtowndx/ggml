#pragma once  // 防止重复包含

#include <atomic>  // 引入 atomic 头文件
#include <cassert>  // 引入 cassert 头文件
#include <cerrno>  // 引入 cerrno 头文件
#include <cstdarg>  // 引入 cstdarg 头文件
#include <cstddef>  // 引入 cstddef 头文件
#include <cstdint>  // 引入 cstdint 头文件
#include <cstdio>  // 引入 cstdio 头文件
#include <cstdlib>  // 引入 cstdlib 头文件
#include <ctime>  // 引入 ctime 头文件

#define unlikely(x) __builtin_expect(!!(x), 0)  // 宏定义 unlikely
#define likely(x)   __builtin_expect(!!(x), 1)  // 宏定义 likely

#ifndef UNUSED  // 如果未定义 UNUSED 则编译
#    define UNUSED(x) (void) (x)  // 宏定义 UNUSED
#endif  // 条件编译结束

/** Checks is a value is a power of two. Does not handle zero. */
#define IS_POT(v) (((v) & ((v) - 1)) == 0)  // 宏定义 IS_POT

/** Checks is a value is a power of two. Zero handled. */
#define IS_POT_NONZERO(v) ((v) != 0 && IS_POT(v))  // 宏定义 IS_POT_NONZERO

/** Align a value to a power of two */
#define ALIGN_POT(x, pot_align) (((x) + (pot_align) - 1) & ~((pot_align) - 1))  // 宏定义 ALIGN_POT

#define p_atomic_read(_v) __atomic_load_n((_v), __ATOMIC_ACQUIRE)  // 宏定义 p_atomic_read

static inline bool util_is_power_of_two_nonzero64(uint64_t v) {
    return IS_POT_NONZERO(v);  // IS_POT_NONZERO
}

static inline uint64_t align64(uint64_t value, uint64_t alignment) {
    assert(util_is_power_of_two_nonzero64(alignment));
    return ALIGN_POT(value, alignment);  // ALIGN_POT
}

struct list_head {  // 结构体定义
    list_head * prev;
    list_head * next;
};

struct util_sparse_array {  // 结构体定义
    size_t   elem_size;
    unsigned node_size_log2;

    uintptr_t root;
};

void * util_sparse_array_get(util_sparse_array * arr, uint64_t idx);  // util_sparse_array_get
void   util_sparse_array_init(util_sparse_array * arr, size_t elem_size, size_t node_size);  // util_sparse_array_init

inline void os_time_sleep(int64_t usecs) {
    timespec time;
    time.tv_sec  = usecs / 1000000;
    time.tv_nsec = (usecs % 1000000) * 1000;
    while (clock_nanosleep(CLOCK_MONOTONIC, 0, &time, &time) == EINTR)
        ;
}

struct timer_data {  // 结构体定义
    long long start;
    long long total;
    long long count;
};

static inline void start_timer(timer_data * timer) {
    timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    timer->start = (long long) ts.tv_sec * 1000000000LL + ts.tv_nsec;
}

// returns the duration in ns
static inline long long stop_timer(timer_data * timer) {
    timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    long long timer_end = (long long) ts.tv_sec * 1000000000LL + ts.tv_nsec;

    long long duration = (timer_end - timer->start);
    timer->total += duration;
    timer->count += 1;

    return duration;  // 返回
}
