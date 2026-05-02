// Conditional fine-grained profiling macros for HMX operations.
//
// Define ENABLE_PROFILE_TIMERS (via compiler flag or before including this
// header) to instrument sub-operation latencies with HAP qtimer.  When the
// macro is not defined the TIMER_* helpers expand to nothing so there is zero
// overhead.
//
// Usage:
//   TIMER_DEFINE(my_phase);          // declare accumulator variable
//   TIMER_START(my_phase);           // snapshot start time
//   ... work ...
//   TIMER_STOP(my_phase);            // accumulate elapsed ticks
//   FARF(ALWAYS, "my_phase: %lld us", TIMER_US(my_phase));

#ifndef HMX_PROFILE_H  // 如果未定义 HMX_PROFILE_H 则编译
#define HMX_PROFILE_H  // 宏定义 HMX_PROFILE_H

#include <HAP_perf.h>  // 引入 HAP_perf.h 头文件

// #define ENABLE_PROFILE_TIMERS

#if defined(ENABLE_PROFILE_TIMERS)  // 条件编译
#  define TIMER_DEFINE(name) int64_t name##_ticks = 0  // 宏定义 TIMER_DEFINE
#  define TIMER_START(name)  int64_t name##_t0 = HAP_perf_get_qtimer_count()  // 宏定义 TIMER_START
#  define TIMER_STOP(name)   name##_ticks += HAP_perf_get_qtimer_count() - name##_t0  // 宏定义 TIMER_STOP
#  define TIMER_US(name)     HAP_perf_qtimer_count_to_us(name##_ticks)  // 宏定义 TIMER_US
#else  // 否则
#  define TIMER_DEFINE(name)  // 宏定义 TIMER_DEFINE
#  define TIMER_START(name)  // 宏定义 TIMER_START
#  define TIMER_STOP(name)  // 宏定义 TIMER_STOP
#  define TIMER_US(name)     0LL  // 宏定义 TIMER_US
#endif  // 条件编译结束

#endif // HMX_PROFILE_H  // 条件编译结束
