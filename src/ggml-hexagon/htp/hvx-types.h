#ifndef HVX_TYPES_H  // 如果未定义 HVX_TYPES_H 则编译
#define HVX_TYPES_H  // 宏定义 HVX_TYPES_H

#include <stdbool.h>  // 引入 stdbool.h 头文件
#include <stdint.h>  // 引入 stdint.h 头文件

#include <hexagon_types.h>  // 引入 hexagon_types.h 头文件

#define SIZEOF_FP32 (4)  // 宏定义 SIZEOF_FP32
#define SIZEOF_FP16 (2)  // 宏定义 SIZEOF_FP16
#define VLEN        (128)  // 宏定义 VLEN
#define VLEN_FP32   (VLEN / SIZEOF_FP32)  // 宏定义 VLEN_FP32
#define VLEN_FP16   (VLEN / SIZEOF_FP16)  // 宏定义 VLEN_FP16

typedef union {  // 类型定义
    HVX_Vector v;
    uint8_t    b[VLEN];
    uint16_t   h[VLEN_FP16];
    uint32_t   w[VLEN_FP32];
    __fp16     fp16[VLEN_FP16];
    float      fp32[VLEN_FP32];
} __attribute__((aligned(VLEN), packed)) HVX_VectorAlias;

typedef struct {  // 类型定义
    HVX_Vector v[2];
} HVX_Vector_x2;

typedef struct {  // 类型定义
    HVX_Vector v[4];
} HVX_Vector_x4;

typedef struct {  // 类型定义
    HVX_Vector v[8];
} HVX_Vector_x8;

#endif /* HVX_TYPES_H */  // 条件编译结束
