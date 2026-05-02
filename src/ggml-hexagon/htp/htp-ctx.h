#ifndef HTP_CTX_H  // 如果未定义 HTP_CTX_H 则编译
#define HTP_CTX_H  // 宏定义 HTP_CTX_H

#include "hex-dma.h"  // 引入 hex-dma.h 头文件
#include "hmx-queue.h"  // 引入 hmx-queue.h 头文件
#include "htp-ops.h"  // 引入 htp-ops.h 头文件
#include "worker-pool.h"  // 引入 worker-pool.h 头文件

#include <assert.h>  // 引入 assert.h 头文件
#include <dspqueue.h>  // 引入 dspqueue.h 头文件
#include <stdatomic.h>  // 引入 stdatomic.h 头文件
#include <stdint.h>  // 引入 stdint.h 头文件
#include <stdbool.h>  // 引入 stdbool.h 头文件

#define HTP_MAX_NTHREADS 10  // 宏定义 HTP_MAX_NTHREADS
#define HTP_MAX_MMAPS    16  // 宏定义 HTP_MAX_MMAPS

// Memory mapping
struct htp_mmap {  // 结构体定义
    uint64_t size;
    uint64_t base;
    uint32_t fd;
    uint32_t reserved;
};

// Scratchpad state
struct htp_spad {  // 结构体定义
    const struct htp_tensor * src;             // original src of the data (for reuse)
    uint8_t *                 data;            // pointer to an area in vtcm
    uint32_t                  stride;          // stride used inside this spad
    uint32_t                  size;            // total size
    uint32_t                  size_per_thread; // size per thread
};

struct htp_context;

// Context while processing an Op
// TODO: fold this into the main context
struct htp_ops_context {  // 结构体定义
    struct htp_context * ctx;

    enum htp_op_code    op; // FIXME: rename to opcode
    int32_t             op_params[HTP_OP_MAX_PARAMS];

    const struct htp_tensor * src[HTP_OP_MAX_INPUTS];
    const struct htp_tensor * dst;

    // TODO convert these to an array
    struct htp_spad src0_spad;
    struct htp_spad src1_spad;
    struct htp_spad src2_spad;
    struct htp_spad src3_spad;
    struct htp_spad dst_spad;

    uint32_t n_threads;
    uint32_t flags;
};

// Main context for htp DSP backend
struct htp_context {  // 结构体定义
    dspqueue_t             queue;
    dma_queue *            dma[HTP_MAX_NTHREADS];
    struct htp_mmap        mmap[HTP_MAX_MMAPS];
    worker_pool_context_t  worker_pool;
    uint32_t               n_threads;

    int                    thread_id;
    int                    thread_prio;

    bool                   hmx_enabled;
    bool                   etm;
    uint32_t               profiler;

    uint8_t *              vtcm_base;
    size_t                 vtcm_size;
    uint32_t               vtcm_rctx;
    atomic_bool            vtcm_valid;
    atomic_bool            vtcm_needs_release;

    uint64_t               max_vmem;

    struct htp_ops_context octx;

#ifdef HTP_HAS_HMX  // 如果定义了 HTP_HAS_HMX 则编译
    struct hmx_queue *     hmx_queue; // Async HMX queue for pipeline overlap
#endif  // 条件编译结束
};

int op_matmul(struct htp_ops_context * octx);  // op_matmul
int op_matmul_id(struct htp_ops_context * octx);  // op_matmul_id
int op_binary(struct htp_ops_context * octx);  // op_binary
int op_unary(struct htp_ops_context * octx);  // op_unary
int op_sum_rows(struct htp_ops_context * octx);  // op_sum_rows
int op_activations(struct htp_ops_context * octx);  // op_activations
int op_softmax(struct htp_ops_context * octx);  // op_softmax
int op_add_id(struct htp_ops_context * octx);  // op_add_id
int op_rope(struct htp_ops_context * octx);  // op_rope
int op_flash_attn_ext(struct htp_ops_context * octx);  // op_flash_attn_ext
int op_set_rows(struct htp_ops_context * octx);  // op_set_rows
int op_get_rows(struct htp_ops_context * octx);  // op_get_rows
int op_cpy(struct htp_ops_context * octx);  // op_cpy
int op_repeat(struct htp_ops_context * octx);  // op_repeat
int op_argsort(struct htp_ops_context * octx);  // op_argsort
int op_ssm_conv(struct htp_ops_context * octx);  // op_ssm_conv
int op_cumsum(struct htp_ops_context * octx);  // op_cumsum
int op_fill(struct htp_ops_context * octx);  // op_fill
int op_diag(struct htp_ops_context * octx);  // op_diag
int op_solve_tri(struct htp_ops_context * octx);  // op_solve_tri

#endif /* HTP_CTX_H */  // 条件编译结束
