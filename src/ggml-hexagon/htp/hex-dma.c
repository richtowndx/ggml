#include "hex-dma.h"  // 引入 hex-dma.h 头文件

#include <stdbool.h>  // 引入 stdbool.h 头文件
#include <stdlib.h>  // 引入 stdlib.h 头文件
#include <string.h>  // 引入 string.h 头文件

#pragma clang diagnostic ignored "-Wunused-function"

static inline uint32_t pow2_ceil(uint32_t x) {
    if (x <= 1) {
        return 1;  // 返回
    }
    int p = 2;
    x--;
    while (x >>= 1) {
        p <<= 1;
    }
    return p;  // 返回
}

dma_queue * dma_queue_create(size_t capacity) {
    dma_queue * q = (dma_queue *) memalign(32, sizeof(dma_queue));
    if (q == NULL) {
        FARF(ERROR, "%s: failed to allocate DMA queue\n", __FUNCTION__);
        return NULL;  // 返回
    }

    capacity = pow2_ceil(capacity);

    memset(q, 0, sizeof(dma_queue));
    q->capacity = capacity;
    q->idx_mask = capacity - 1;

    q->desc = (dma_descriptor_2d *) memalign(64, capacity * sizeof(dma_descriptor_2d));
    memset(q->desc, 0, capacity * sizeof(dma_descriptor_2d));

    q->dptr = (dma_ptr *) memalign(4, capacity * sizeof(dma_ptr));
    memset(q->dptr, 0, capacity * sizeof(dma_ptr));

    q->tail = &q->desc[capacity - 1];

    if (!q->desc && !q->dptr) {
        FARF(ERROR, "%s: failed to allocate DMA queue items\n", __FUNCTION__);
        return NULL;  // 返回
    }

    FARF(HIGH, "dma-queue: capacity %u\n", capacity);

    return q;  // 返回
}

void dma_queue_delete(dma_queue * q) {
    if (!q) {
        return;  // 返回
    }
    free(q->desc);
    free(q->dptr);
    free(q);
}

void dma_queue_flush(dma_queue * q) {
    while (dma_queue_pop(q).dst != NULL) ;
}
