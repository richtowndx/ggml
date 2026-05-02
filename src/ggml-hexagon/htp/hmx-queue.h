#ifndef HMX_QUEUE_H  // 如果未定义 HMX_QUEUE_H 则编译
#define HMX_QUEUE_H  // 宏定义 HMX_QUEUE_H

#include <stdbool.h>  // 引入 stdbool.h 头文件
#include <stdint.h>  // 引入 stdint.h 头文件
#include <stdatomic.h>  // 引入 stdatomic.h 头文件

#include <hexagon_types.h>  // 引入 hexagon_types.h 头文件
#include <qurt_thread.h>  // 引入 qurt_thread.h 头文件
#include <qurt_futex.h>  // 引入 qurt_futex.h 头文件
#include <HAP_farf.h>  // 引入 HAP_farf.h 头文件

#include "hex-utils.h"  // 引入 hex-utils.h 头文件

#ifdef __cplusplus  // 如果定义了 __cplusplus 则编译
extern "C" {  // C 链接声明
#endif  // 条件编译结束

#define HMX_QUEUE_THREAD_STACK_SIZE (16 * 1024)  // 宏定义 HMX_QUEUE_THREAD_STACK_SIZE
#define HMX_QUEUE_POLL_COUNT        2000  // 宏定义 HMX_QUEUE_POLL_COUNT

typedef void (*hmx_queue_func)(void *);  // 类型定义

// Dummy funcs used as signals
enum hmx_queue_signal {  // 枚举定义
    HMX_QUEUE_NOOP = 0, // aka NULL
    HMX_QUEUE_SUSPEND,
    HMX_QUEUE_KILL
};

struct hmx_queue_desc {  // 结构体定义
    hmx_queue_func   func;
    void *           data;
    atomic_uint      done;
};

struct hmx_queue {  // 结构体定义
    struct hmx_queue_desc * desc;
    atomic_uint      idx_write; // updated by producer (push)
    atomic_uint      idx_read;  // updated by consumer (process)
    unsigned int     idx_pop;   // updated by producer (pop)
    uint32_t         idx_mask;
    uint32_t         capacity;

    atomic_uint      seqn;      // incremented for all pushes, used with futex
    qurt_thread_t    thread;
    void *           stack;
    uint32_t         hap_rctx;
    bool             hmx_locked;
};

struct hmx_queue * hmx_queue_create(size_t capacity, uint32_t hap_rctx);  // hmx_queue_create
void hmx_queue_delete(struct hmx_queue * q);  // hmx_queue_delete

static inline struct hmx_queue_desc hmx_queue_make_desc(hmx_queue_func func, void * data) {
    struct hmx_queue_desc d = { func, data };  // 结构体定义
    return d;  // 返回
}

static inline bool hmx_queue_push(struct hmx_queue * q, struct hmx_queue_desc d) {
    unsigned int ir = atomic_load(&q->idx_read);
    unsigned int iw = q->idx_write;

    if (((iw + 1) & q->idx_mask) == ir) {
        FARF(HIGH, "hmx-queue-push: queue is full\n");
        return false;  // 返回
    }

    atomic_store(&d.done, 0);

    FARF(HIGH, "hmx-queue-push: iw %u func %p data %p\n", iw, d.func, d.data);

    q->desc[iw] = d;
    atomic_store(&q->idx_write, (iw + 1) & q->idx_mask);
    // wake up our thread
    atomic_fetch_add(&q->seqn, 1);
    qurt_futex_wake(&q->seqn, 1);

    return true;  // 返回
}

static inline bool hmx_queue_signal(struct hmx_queue *q, enum hmx_queue_signal sig) {
    return hmx_queue_push(q, hmx_queue_make_desc((hmx_queue_func) sig, NULL));  // hmx_queue_push
}

static inline bool hmx_queue_empty(struct hmx_queue * q) {
    return q->idx_pop == q->idx_write;  // 返回
}

static inline uint32_t hmx_queue_depth(struct hmx_queue * q) {
    return (q->idx_read - q->idx_read) & q->idx_mask;
}

static inline uint32_t hmx_queue_capacity(struct hmx_queue * q) {
    return q->capacity;  // 返回
}

static inline struct hmx_queue_desc hmx_queue_pop(struct hmx_queue * q) {
    unsigned int ip = q->idx_pop;
    unsigned int iw = q->idx_write;

    struct hmx_queue_desc rd = { NULL, NULL };  // 结构体定义
    if (ip == iw) {
        return rd;  // 返回
    }

    // Wait for desc to complete
    struct hmx_queue_desc * d = &q->desc[ip];
    while (!atomic_load(&d->done)) {
        FARF(HIGH, "hmx-queue-pop: waiting for HMX queue : %u\n", ip);
        hex_pause();
    }

    rd = *d;
    q->idx_pop = (ip + 1) & q->idx_mask;

    FARF(HIGH, "hmx-queue-pop: ip %u func %p data %p\n", ip, rd.func, rd.data);
    return rd;  // 返回
}

static inline void hmx_queue_flush(struct hmx_queue * q) {
    while (hmx_queue_pop(q).func != NULL) ;
}

static inline void hmx_queue_suspend(struct hmx_queue *q) {
    hmx_queue_signal(q, HMX_QUEUE_SUSPEND);
    hmx_queue_flush(q);
}

#ifdef __cplusplus  // 如果定义了 __cplusplus 则编译
}  // extern "C"  // C 链接声明
#endif  // 条件编译结束

#endif /* HMX_QUEUE_H */  // 条件编译结束
