#ifndef VTCM_UTILS_H  // 如果未定义 VTCM_UTILS_H 则编译
#define VTCM_UTILS_H  // 宏定义 VTCM_UTILS_H

#include "hex-utils.h"  // 引入 hex-utils.h 头文件

#include <assert.h>  // 引入 assert.h 头文件
#include <stdint.h>  // 引入 stdint.h 头文件
#include <hexagon_types.h>  // 引入 hexagon_types.h 头文件

static inline uint8_t *vtcm_seq_alloc(uint8_t **vtcm_ptr, size_t size) {
    uint8_t *p = *vtcm_ptr;
    *vtcm_ptr += size;
    return p;  // 返回
}

#endif // VTCM_UTILS_H  // 条件编译结束
