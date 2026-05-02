#ifndef GGML_METAL_IMPL  // 如果未定义 GGML_METAL_IMPL 则编译
#define GGML_METAL_IMPL  // 宏定义 GGML_METAL_IMPL

// kernel parameters for mat-mat threadgroups
//
// TODO: become function constants

#define SZ_SIMDGROUP 16  // 宏定义 SZ_SIMDGROUP
#define N_MM_NK 2  // 宏定义 N_MM_NK
#define N_MM_NK_TOTAL (SZ_SIMDGROUP * N_MM_NK)  // 宏定义 N_MM_NK_TOTAL

#define N_MM_BLOCK_X 4  // 宏定义 N_MM_BLOCK_X
#define N_MM_BLOCK_Y 2  // 宏定义 N_MM_BLOCK_Y
#define N_MM_SIMD_GROUP_X 2  // 宏定义 N_MM_SIMD_GROUP_X
#define N_MM_SIMD_GROUP_Y 2  // 宏定义 N_MM_SIMD_GROUP_Y

// kernel parameters for mat-vec threadgroups
//
// N_R0: number of src0 rows to process per simdgroup
// N_SG: number of simdgroups per threadgroup
//
// TODO: for optimal performance, become function of the device and work size

#define N_R0_Q1_0 8  // 宏定义 N_R0_Q1_0
#define N_SG_Q1_0 2  // 宏定义 N_SG_Q1_0

#define N_R0_Q4_0 4  // 宏定义 N_R0_Q4_0
#define N_SG_Q4_0 2  // 宏定义 N_SG_Q4_0

#define N_R0_Q4_1 4  // 宏定义 N_R0_Q4_1
#define N_SG_Q4_1 2  // 宏定义 N_SG_Q4_1

#define N_R0_Q5_0 4  // 宏定义 N_R0_Q5_0
#define N_SG_Q5_0 2  // 宏定义 N_SG_Q5_0

#define N_R0_Q5_1 4  // 宏定义 N_R0_Q5_1
#define N_SG_Q5_1 2  // 宏定义 N_SG_Q5_1

#define N_R0_Q8_0 2  // 宏定义 N_R0_Q8_0
#define N_SG_Q8_0 4  // 宏定义 N_SG_Q8_0

#define N_R0_MXFP4 2  // 宏定义 N_R0_MXFP4
#define N_SG_MXFP4 2  // 宏定义 N_SG_MXFP4

#define N_R0_Q2_K 4  // 宏定义 N_R0_Q2_K
#define N_SG_Q2_K 2  // 宏定义 N_SG_Q2_K

#define N_R0_Q3_K 2  // 宏定义 N_R0_Q3_K
#define N_SG_Q3_K 2  // 宏定义 N_SG_Q3_K

#define N_R0_Q4_K 2  // 宏定义 N_R0_Q4_K
#define N_SG_Q4_K 2  // 宏定义 N_SG_Q4_K

#define N_R0_Q5_K 1  // 宏定义 N_R0_Q5_K
#define N_SG_Q5_K 2  // 宏定义 N_SG_Q5_K

#define N_R0_Q6_K 2  // 宏定义 N_R0_Q6_K
#define N_SG_Q6_K 2  // 宏定义 N_SG_Q6_K

#define N_R0_IQ1_S 4  // 宏定义 N_R0_IQ1_S
#define N_SG_IQ1_S 2  // 宏定义 N_SG_IQ1_S

#define N_R0_IQ1_M 4  // 宏定义 N_R0_IQ1_M
#define N_SG_IQ1_M 2  // 宏定义 N_SG_IQ1_M

#define N_R0_IQ2_XXS 4  // 宏定义 N_R0_IQ2_XXS
#define N_SG_IQ2_XXS 2  // 宏定义 N_SG_IQ2_XXS

#define N_R0_IQ2_XS 4  // 宏定义 N_R0_IQ2_XS
#define N_SG_IQ2_XS 2  // 宏定义 N_SG_IQ2_XS

#define N_R0_IQ2_S 4  // 宏定义 N_R0_IQ2_S
#define N_SG_IQ2_S 2  // 宏定义 N_SG_IQ2_S

#define N_R0_IQ3_XXS 4  // 宏定义 N_R0_IQ3_XXS
#define N_SG_IQ3_XXS 2  // 宏定义 N_SG_IQ3_XXS

#define N_R0_IQ3_S 4  // 宏定义 N_R0_IQ3_S
#define N_SG_IQ3_S 2  // 宏定义 N_SG_IQ3_S

#define N_R0_IQ4_NL 2  // 宏定义 N_R0_IQ4_NL
#define N_SG_IQ4_NL 2  // 宏定义 N_SG_IQ4_NL

#define N_R0_IQ4_XS 2  // 宏定义 N_R0_IQ4_XS
#define N_SG_IQ4_XS 2  // 宏定义 N_SG_IQ4_XS

// function constants offsets
#define FC_FLASH_ATTN_EXT_PAD          100  // 宏定义 FC_FLASH_ATTN_EXT_PAD
#define FC_FLASH_ATTN_EXT_BLK          200  // 宏定义 FC_FLASH_ATTN_EXT_BLK
#define FC_FLASH_ATTN_EXT              300  // 宏定义 FC_FLASH_ATTN_EXT
#define FC_FLASH_ATTN_EXT_VEC          400  // 宏定义 FC_FLASH_ATTN_EXT_VEC
#define FC_FLASH_ATTN_EXT_VEC_REDUCE   500  // 宏定义 FC_FLASH_ATTN_EXT_VEC_REDUCE
#define FC_MUL_MV                      600  // 宏定义 FC_MUL_MV
#define FC_MUL_MM                      700  // 宏定义 FC_MUL_MM
#define FC_ROPE                        800  // 宏定义 FC_ROPE
#define FC_SSM_CONV                    900  // 宏定义 FC_SSM_CONV
#define FC_SOLVE_TRI                   1000  // 宏定义 FC_SOLVE_TRI
#define FC_COUNT_EQUAL                 1100  // 宏定义 FC_COUNT_EQUAL
#define FC_UNARY                       1200  // 宏定义 FC_UNARY
#define FC_BIN                         1300  // 宏定义 FC_BIN
#define FC_SUM_ROWS                    1400  // 宏定义 FC_SUM_ROWS
#define FC_UPSCALE                     1500  // 宏定义 FC_UPSCALE
#define FC_GATED_DELTA_NET             1600  // 宏定义 FC_GATED_DELTA_NET

// op-specific constants
#define OP_FLASH_ATTN_EXT_NQPSG 8  // 宏定义 OP_FLASH_ATTN_EXT_NQPSG
#define OP_FLASH_ATTN_EXT_NCPSG 64  // 宏定义 OP_FLASH_ATTN_EXT_NCPSG

#define OP_FLASH_ATTN_EXT_VEC_NQPSG 1  // 宏定义 OP_FLASH_ATTN_EXT_VEC_NQPSG
#define OP_FLASH_ATTN_EXT_VEC_NCPSG 32  // 宏定义 OP_FLASH_ATTN_EXT_VEC_NCPSG

#define OP_UNARY_NUM_SCALE      10  // 宏定义 OP_UNARY_NUM_SCALE
#define OP_UNARY_NUM_FILL       11  // 宏定义 OP_UNARY_NUM_FILL
#define OP_UNARY_NUM_CLAMP      12  // 宏定义 OP_UNARY_NUM_CLAMP
#define OP_UNARY_NUM_SQR        13  // 宏定义 OP_UNARY_NUM_SQR
#define OP_UNARY_NUM_SQRT       14  // 宏定义 OP_UNARY_NUM_SQRT
#define OP_UNARY_NUM_SIN        15  // 宏定义 OP_UNARY_NUM_SIN
#define OP_UNARY_NUM_COS        16  // 宏定义 OP_UNARY_NUM_COS
#define OP_UNARY_NUM_LOG        17  // 宏定义 OP_UNARY_NUM_LOG
#define OP_UNARY_NUM_LEAKY_RELU 18  // 宏定义 OP_UNARY_NUM_LEAKY_RELU

#define OP_UNARY_NUM_TANH        100  // 宏定义 OP_UNARY_NUM_TANH
#define OP_UNARY_NUM_RELU        101  // 宏定义 OP_UNARY_NUM_RELU
#define OP_UNARY_NUM_SIGMOID     102  // 宏定义 OP_UNARY_NUM_SIGMOID
#define OP_UNARY_NUM_GELU        103  // 宏定义 OP_UNARY_NUM_GELU
#define OP_UNARY_NUM_GELU_ERF    104  // 宏定义 OP_UNARY_NUM_GELU_ERF
#define OP_UNARY_NUM_GELU_QUICK  105  // 宏定义 OP_UNARY_NUM_GELU_QUICK
#define OP_UNARY_NUM_SILU        106  // 宏定义 OP_UNARY_NUM_SILU
#define OP_UNARY_NUM_ELU         107  // 宏定义 OP_UNARY_NUM_ELU
#define OP_UNARY_NUM_NEG         108  // 宏定义 OP_UNARY_NUM_NEG
#define OP_UNARY_NUM_ABS         109  // 宏定义 OP_UNARY_NUM_ABS
#define OP_UNARY_NUM_SGN         110  // 宏定义 OP_UNARY_NUM_SGN
#define OP_UNARY_NUM_STEP        111  // 宏定义 OP_UNARY_NUM_STEP
#define OP_UNARY_NUM_HARDSWISH   112  // 宏定义 OP_UNARY_NUM_HARDSWISH
#define OP_UNARY_NUM_HARDSIGMOID 113  // 宏定义 OP_UNARY_NUM_HARDSIGMOID
#define OP_UNARY_NUM_EXP         114  // 宏定义 OP_UNARY_NUM_EXP
#define OP_UNARY_NUM_SOFTPLUS    115  // 宏定义 OP_UNARY_NUM_SOFTPLUS
#define OP_UNARY_NUM_EXPM1       116  // 宏定义 OP_UNARY_NUM_EXPM1
#define OP_UNARY_NUM_FLOOR       117  // 宏定义 OP_UNARY_NUM_FLOOR
#define OP_UNARY_NUM_CEIL        118  // 宏定义 OP_UNARY_NUM_CEIL
#define OP_UNARY_NUM_ROUND       119  // 宏定义 OP_UNARY_NUM_ROUND
#define OP_UNARY_NUM_TRUNC       120  // 宏定义 OP_UNARY_NUM_TRUNC
#define OP_UNARY_NUM_XIELU       121  // 宏定义 OP_UNARY_NUM_XIELU

#define OP_SUM_ROWS_NUM_SUM_ROWS 10  // 宏定义 OP_SUM_ROWS_NUM_SUM_ROWS
#define OP_SUM_ROWS_NUM_MEAN     11  // 宏定义 OP_SUM_ROWS_NUM_MEAN

// kernel argument structs
//
// - element counters (e.g. ne00) typically use int32_t to reduce register usage
//   however, be careful from int overflows when using those in the kernel implementation
//
// - strides (e.g. nb00) use uint64_t

typedef struct {  // 类型定义
    int32_t  ne00;
    int32_t  ne01;
    int32_t  ne02;
    int32_t  ne03;
    uint64_t nb00;
    uint64_t nb01;
    uint64_t nb02;
    uint64_t nb03;
    int32_t  ne10;
    int32_t  ne11;
    int32_t  ne12;
    int32_t  ne13;
    uint64_t nb10;
    uint64_t nb11;
    uint64_t nb12;
    uint64_t nb13;
    int32_t  ne0;
    int32_t  ne1;
    int32_t  ne2;
    int32_t  ne3;
    uint64_t nb0;
    uint64_t nb1;
    uint64_t nb2;
    uint64_t nb3;
    int32_t  dim;
} ggml_metal_kargs_concat;

typedef struct {  // 类型定义
    int32_t  ne00;
    int32_t  ne01;
    int32_t  ne02;
    int32_t  ne03;
    uint64_t nb00;
    uint64_t nb01;
    uint64_t nb02;
    uint64_t nb03;
    int32_t  ne0;
    int32_t  ne1;
    int32_t  ne2;
    int32_t  ne3;
    uint64_t nb0;
    uint64_t nb1;
    uint64_t nb2;
    uint64_t nb3;
    float    slope;
    float    scale;
    float    bias;
    float    val;
    float    min;
    float    max;
} ggml_metal_kargs_unary;

typedef struct {  // 类型定义
    int32_t  ne00;
    int32_t  ne01;
    int32_t  ne02;
    int32_t  ne03;
    uint64_t nb00;
    uint64_t nb01;
    uint64_t nb02;
    uint64_t nb03;
    int32_t  ne10;
    int32_t  ne11;
    int32_t  ne12;
    int32_t  ne13;
    uint64_t nb10;
    uint64_t nb11;
    uint64_t nb12;
    uint64_t nb13;
    int32_t  ne0;
    int32_t  ne1;
    int32_t  ne2;
    int32_t  ne3;
    uint64_t nb0;
    uint64_t nb1;
    uint64_t nb2;
    uint64_t nb3;
    uint64_t offs;
    uint64_t o1[8];
} ggml_metal_kargs_bin;

typedef struct {  // 类型定义
    int64_t ne0;
    int64_t ne1;
    size_t nb01;
    size_t nb02;
    size_t nb11;
    size_t nb21;
} ggml_metal_kargs_add_id;

typedef struct {  // 类型定义
    int32_t  ne00;
    int32_t  ne01;
    int32_t  ne02;
    int32_t  ne03;
    uint64_t nb00;
    uint64_t nb01;
    uint64_t nb02;
    uint64_t nb03;
    int32_t  ne0;
    int32_t  ne1;
    int32_t  ne2;
    int32_t  ne3;
    uint64_t nb0;
    uint64_t nb1;
    uint64_t nb2;
    uint64_t nb3;
} ggml_metal_kargs_repeat;

typedef struct {  // 类型定义
    int64_t  nk0;
    int64_t  ne00;
    int64_t  ne01;
    int64_t  ne02;
    int64_t  ne03;
    uint64_t nb00;
    uint64_t nb01;
    uint64_t nb02;
    uint64_t nb03;
    int64_t  ne0;
    int64_t  ne1;
    int64_t  ne2;
    int64_t  ne3;
    uint64_t nb0;
    uint64_t nb1;
    uint64_t nb2;
    uint64_t nb3;
} ggml_metal_kargs_cpy;

typedef struct {  // 类型定义
    int64_t  ne10;
    int64_t  ne11;
    int64_t  ne12;
    uint64_t nb10;
    uint64_t nb11;
    uint64_t nb12;
    uint64_t nb13;
    uint64_t nb1;
    uint64_t nb2;
    uint64_t nb3;
    uint64_t offs;
    bool     inplace;
} ggml_metal_kargs_set;

typedef struct {  // 类型定义
    int32_t  ne00;
    int32_t  ne01;
    int32_t  ne02;
    int32_t  ne03;
    uint64_t nb00;
    uint64_t nb01;
    uint64_t nb02;
    uint64_t nb03;
    int32_t  ne0;
    int32_t  ne1;
    int32_t  ne2;
    int32_t  ne3;
    uint64_t nb0;
    uint64_t nb1;
    uint64_t nb2;
    uint64_t nb3;
    int32_t  n_past;
    int32_t  n_dims;
    int32_t  n_ctx_orig;
    float    freq_base;
    float    freq_scale;
    float    ext_factor;
    float    attn_factor;
    float    beta_fast;
    float    beta_slow;
    int32_t  sect_0;
    int32_t  sect_1;
    int32_t  sect_2;
    int32_t  sect_3;
    bool     src2;
} ggml_metal_kargs_rope;

typedef struct {  // 类型定义
    int32_t  ne11;
    int32_t  ne_12_2; // assume K and V are same shape
    int32_t  ne_12_3;
    uint64_t nb11;
    uint64_t nb12;
    uint64_t nb13;
    uint64_t nb21;
    uint64_t nb22;
    uint64_t nb23;
    int32_t  ne31;
    int32_t  ne32;
    int32_t  ne33;
    uint64_t nb31;
    uint64_t nb32;
    uint64_t nb33;
} ggml_metal_kargs_flash_attn_ext_pad;

typedef struct {  // 类型定义
    int32_t  ne01;
    int32_t  ne30;
    int32_t  ne31;
    int32_t  ne32;
    int32_t  ne33;
    uint64_t nb31;
    uint64_t nb32;
    uint64_t nb33;
} ggml_metal_kargs_flash_attn_ext_blk;

typedef struct {  // 类型定义
    int32_t  ne01;
    int32_t  ne02;
    int32_t  ne03;
    uint64_t nb01;
    uint64_t nb02;
    uint64_t nb03;
    int32_t  ne11;
    int32_t  ne_12_2; // assume K and V are same shape
    int32_t  ne_12_3;
    int32_t  ns10;
    uint64_t nb11;
    uint64_t nb12;
    uint64_t nb13;
    int32_t  ns20;
    uint64_t nb21;
    uint64_t nb22;
    uint64_t nb23;
    int32_t  ne31;
    int32_t  ne32;
    int32_t  ne33;
    uint64_t nb31;
    uint64_t nb32;
    uint64_t nb33;
    int32_t  ne1;
    int32_t  ne2;
    int32_t  ne3;
    float    scale;
    float    max_bias;
    float    m0;
    float    m1;
    int32_t  n_head_log2;
    float    logit_softcap;
} ggml_metal_kargs_flash_attn_ext;

typedef struct {  // 类型定义
    int32_t  ne01;
    int32_t  ne02;
    int32_t  ne03;
    uint64_t nb01;
    uint64_t nb02;
    uint64_t nb03;
    int32_t  ne11;
    int32_t  ne_12_2; // assume K and V are same shape
    int32_t  ne_12_3;
    int32_t  ns10;
    uint64_t nb11;
    uint64_t nb12;
    uint64_t nb13;
    int32_t  ns20;
    uint64_t nb21;
    uint64_t nb22;
    uint64_t nb23;
    int32_t  ne31;
    int32_t  ne32;
    int32_t  ne33;
    uint64_t nb31;
    uint64_t nb32;
    uint64_t nb33;
    int32_t  ne1;
    int32_t  ne2;
    int32_t  ne3;
    float    scale;
    float    max_bias;
    float    m0;
    float    m1;
    int32_t  n_head_log2;
    float    logit_softcap;
} ggml_metal_kargs_flash_attn_ext_vec;

typedef struct {  // 类型定义
    int32_t  nrows;
} ggml_metal_kargs_flash_attn_ext_vec_reduce;

typedef struct {  // 类型定义
    int32_t  ne00;
    int32_t  ne02;
    uint64_t nb01;
    uint64_t nb02;
    uint64_t nb03;
    int32_t  ne12;
    uint64_t nb10;
    uint64_t nb11;
    uint64_t nb12;
    uint64_t nb13;
    int32_t  ne0;
    int32_t  ne1;
    int16_t  r2;
    int16_t  r3;
} ggml_metal_kargs_mul_mm;

typedef struct {  // 类型定义
    int32_t  ne00;
    int32_t  ne01;
    int32_t  ne02;
    uint64_t nb00;
    uint64_t nb01;
    uint64_t nb02;
    uint64_t nb03;
    int32_t  ne10;
    int32_t  ne11;
    int32_t  ne12;
    uint64_t nb10;
    uint64_t nb11;
    uint64_t nb12;
    uint64_t nb13;
    int32_t  ne0;
    int32_t  ne1;
    int32_t  nr0;
    int16_t  r2;
    int16_t  r3;
} ggml_metal_kargs_mul_mv;

typedef struct {  // 类型定义
    int32_t  ne00;
    int32_t  ne01;
    int32_t  ne02;
    uint64_t nb00;
    uint64_t nb01;
    uint64_t nb02;
    uint64_t nb03;
    int32_t  ne10;
    int32_t  ne11;
    int32_t  ne12;
    uint64_t nb10;
    uint64_t nb11;
    uint64_t nb12;
    uint64_t nb13;
    int32_t  ne0;
    int32_t  ne1;
    int16_t  r2;
    int16_t  r3;
} ggml_metal_kargs_mul_mv_ext;

typedef struct {  // 类型定义
    int32_t  ne02;
    int32_t  ne10;
    int32_t  ne11;  // n_expert_used (bcast)
    uint64_t nb11;
    uint64_t nb12;
    int32_t  ne21; // n_tokens
    int32_t  ne20;  // n_expert_used
    uint64_t nb21;
} ggml_metal_kargs_mul_mm_id_map0;

typedef struct {  // 类型定义
    int32_t  ne00;
    int32_t  ne02;
    uint64_t nb01;
    uint64_t nb02;
    uint64_t nb03;
    int32_t  ne11;
    uint64_t nb10;
    uint64_t nb11;
    uint64_t nb12;
    uint64_t nb13;
    int32_t  ne20;
    int32_t  ne21;
    int32_t  ne0;
    int32_t  ne1;
    int16_t  r2;
    int16_t  r3;
} ggml_metal_kargs_mul_mm_id;

typedef struct {  // 类型定义
    int32_t  nei0;
    int32_t  nei1;
    uint64_t nbi1;
    int32_t  ne00;
    int32_t  ne01;
    int32_t  ne02;
    uint64_t nb00;
    uint64_t nb01;
    uint64_t nb02;
    int32_t  ne10;
    int32_t  ne11;
    int32_t  ne12;
    int32_t  ne13;
    uint64_t nb10;
    uint64_t nb11;
    uint64_t nb12;
    int32_t  ne0;
    int32_t  ne1;
    uint64_t nb1;
    int32_t  nr0;
} ggml_metal_kargs_mul_mv_id;

// NORM
// RMS_NORM
typedef struct {  // 类型定义
    int32_t  ne00;
    int32_t  ne00_t;
    uint64_t nb1;
    uint64_t nb2;
    uint64_t nb3;
    float    eps;
    int32_t  nef1[3];
    int32_t  nef2[3];
    int32_t  nef3[3];
    uint64_t nbf1[3];
    uint64_t nbf2[3];
    uint64_t nbf3[3];
} ggml_metal_kargs_norm;

typedef struct {  // 类型定义
    int32_t  ne00;
    int32_t  ne01;
    int32_t  ne02;
    int32_t  ne03;
    uint64_t nb00;
    uint64_t nb01;
    uint64_t nb02;
    uint64_t nb03;
    int32_t  ne0;
    int32_t  ne1;
    int32_t  ne2;
    int32_t  ne3;
    uint64_t nb0;
    uint64_t nb1;
    uint64_t nb2;
    uint64_t nb3;
    float    eps;
} ggml_metal_kargs_l2_norm;

typedef struct {  // 类型定义
    int64_t  ne00;
    int64_t  ne01;
    int64_t  ne02;
    uint64_t nb00;
    uint64_t nb01;
    uint64_t nb02;
    int32_t  ngrp;
    float    eps;
} ggml_metal_kargs_group_norm;

typedef struct {  // 类型定义
    int32_t  IC;
    int32_t  IL;
    int32_t  K;
    int32_t  s0;
    uint64_t nb0;
    uint64_t nb1;
} ggml_metal_kargs_conv_transpose_1d;

typedef struct {  // 类型定义
    int32_t  IC;
    int32_t  IH;
    int32_t  IW;
    int32_t  KH;
    int32_t  KW;
    int32_t  OC;
    int32_t  s0;
    uint64_t nb0;
    uint64_t nb1;
    uint64_t nb2;
} ggml_metal_kargs_conv_transpose_2d;

typedef struct {  // 类型定义
    uint64_t nb00;
    uint64_t nb01;
    uint64_t nb02;
    uint64_t nb03;
    uint64_t nb10;
    uint64_t nb11;
    uint64_t nb12;
    uint64_t nb13;
    uint64_t nb0;
    uint64_t nb1;
    uint64_t nb2;
    uint64_t nb3;
    int32_t  IW;
    int32_t  IH;
    int32_t  KW;
    int32_t  KH;
    int32_t  IC;
    int32_t  OC;
    int32_t  OW;
    int32_t  OH;
    int32_t  N;
    int32_t  s0;
    int32_t  s1;
    int32_t  p0;
    int32_t  p1;
    int32_t  d0;
    int32_t  d1;
} ggml_metal_kargs_conv_2d;

typedef struct {  // 类型定义
    uint64_t  ofs0;
    uint64_t  ofs1;
    int32_t  IW;
    int32_t  IH;
    int32_t  CHW;
    int32_t  s0;
    int32_t  s1;
    int32_t  p0;
    int32_t  p1;
    int32_t  d0;
    int32_t  d1;
    int32_t  N;
    int32_t  KH;
    int32_t  KW;
    int32_t  KHW; // KH * KW, pre-computed on CPU to save GPU resources
} ggml_metal_kargs_im2col;

typedef struct {  // 类型定义
    int32_t  IW;
    int32_t  IH;
    int32_t  ID;
    int32_t  OW;
    int32_t  OH;
    int32_t  OD;
    int32_t  KW;
    int32_t  KH;
    int32_t  KD;
    int32_t  s0;
    int32_t  s1;
    int32_t  s2;
    int32_t  p0;
    int32_t  p1;
    int32_t  p2;
    int32_t  d0;
    int32_t  d1;
    int32_t  d2;
    int32_t  IC;
    int32_t  N;
    int32_t  OC;
    uint64_t nb00;
    uint64_t nb01;
    uint64_t nb02;
    uint64_t nb03;
    uint64_t nb10;
    uint64_t nb11;
    uint64_t nb12;
    uint64_t nb13;
    uint64_t nb0;
    uint64_t nb1;
    uint64_t nb2;
    uint64_t nb3;
} ggml_metal_kargs_conv_3d;

typedef struct{  // 类型定义
    int32_t  ne00;
    uint64_t nb01;
    int32_t  ne10;
    uint64_t nb11;
    int32_t  ne0;
    uint64_t nb1;
    int32_t  i00;
    int32_t  i10;
    float    alpha;
    float    limit;
} ggml_metal_kargs_glu;

typedef struct {  // 类型定义
    uint64_t np;
} ggml_metal_kargs_sum;

typedef struct {  // 类型定义
    int64_t  ne00;
    int64_t  ne01;
    int64_t  ne02;
    int64_t  ne03;
    uint64_t nb00;
    uint64_t nb01;
    uint64_t nb02;
    uint64_t nb03;
    int64_t  ne0;
    int64_t  ne1;
    int64_t  ne2;
    int64_t  ne3;
    uint64_t nb0;
    uint64_t nb1;
    uint64_t nb2;
    uint64_t nb3;
} ggml_metal_kargs_sum_rows;

typedef struct {  // 类型定义
    int64_t  ne00;
    int64_t  ne01;
    int64_t  ne02;
    int64_t  ne03;
    uint64_t nb00;
    uint64_t nb01;
    uint64_t nb02;
    uint64_t nb03;
    int64_t  net0;
    int64_t  net1;
    int64_t  net2;
    int64_t  net3;
    uint64_t nbt0;
    uint64_t nbt1;
    uint64_t nbt2;
    uint64_t nbt3;
    bool     outb;
} ggml_metal_kargs_cumsum_blk;

typedef struct {  // 类型定义
    int64_t  ne00;
    int64_t  ne01;
    int64_t  ne02;
    int64_t  ne03;
    uint64_t nb00;
    uint64_t nb01;
    uint64_t nb02;
    uint64_t nb03;
    int64_t  net0;
    int64_t  net1;
    int64_t  net2;
    int64_t  net3;
    uint64_t nbt0;
    uint64_t nbt1;
    uint64_t nbt2;
    uint64_t nbt3;
} ggml_metal_kargs_cumsum_add;

typedef struct {  // 类型定义
    int32_t  ne00;
    int32_t  ne01;
    int32_t  ne02;
    uint64_t nb01;
    uint64_t nb02;
    uint64_t nb03;
    int32_t  ne11;
    int32_t  ne12;
    int32_t  ne13;
    uint64_t nb11;
    uint64_t nb12;
    uint64_t nb13;
    uint64_t nb1;
    uint64_t nb2;
    uint64_t nb3;
    float    scale;
    float    max_bias;
    float    m0;
    float    m1;
    int32_t  n_head_log2;
} ggml_metal_kargs_soft_max;

typedef struct {  // 类型定义
    int64_t  ne00;
    int64_t  ne01;
    int64_t  ne02;
    uint64_t nb00;
    uint64_t nb01;
    uint64_t nb02;
    int64_t  ne10;
    int64_t  ne11;
    uint64_t nb10;
    uint64_t nb11;
    int64_t  ne0;
    int64_t  ne1;
    int64_t  ne2;
    uint64_t nb0;
    uint64_t nb1;
    uint64_t nb2;
} ggml_metal_kargs_ssm_conv;

typedef struct {  // 类型定义
    int64_t  d_state;
    int64_t  d_inner;
    int64_t  n_head;
    int64_t  n_group;
    int64_t  n_seq_tokens;
    int64_t  n_seqs;
    uint64_t s_off;
    uint64_t nb00;
    uint64_t nb01;
    uint64_t nb02;
    uint64_t nb03;
    uint64_t nb10;
    uint64_t nb11;
    uint64_t nb12;
    uint64_t ns12;
    uint64_t nb13;
    uint64_t nb20;
    uint64_t nb21;
    uint64_t ns21;
    uint64_t nb22;
    int64_t  ne30;
    uint64_t nb31;
    uint64_t nb41;
    uint64_t nb42;
    uint64_t ns42;
    uint64_t nb43;
    uint64_t nb51;
    uint64_t nb52;
    uint64_t ns52;
    uint64_t nb53;
    uint64_t nb0;
} ggml_metal_kargs_ssm_scan;

typedef struct {  // 类型定义
    int32_t  ne00;
    int32_t  ne01;
    int32_t  ne02;
    int32_t  ne03;
    uint64_t nb00;
    uint64_t nb01;
    uint64_t nb02;
    uint64_t nb03;
    int32_t  ne10;
    int32_t  ne11;
    int32_t  ne12;
    int32_t  ne13;
    uint64_t nb10;
    uint64_t nb11;
    uint64_t nb12;
    uint64_t nb13;
    int32_t  ne20;
    int32_t  ne21;
    int32_t  ne22;
    int32_t  ne23;
    uint64_t nb20;
    uint64_t nb21;
    uint64_t nb22;
    uint64_t nb23;
    int32_t  ns02;
    int32_t  ns12;
    int32_t  ns22;
    int32_t  ne0;
    int32_t  ne1;
    int32_t  ne2;
    int32_t  ne3;
    uint64_t nb0;
    uint64_t nb1;
    uint64_t nb2;
    uint64_t nb3;
} ggml_metal_kargs_gated_delta_net;

typedef struct {  // 类型定义
    int32_t  ne00;
    int32_t  ne01;
    int32_t  ne02;
    int32_t  ne03;
    uint64_t nb00;
    uint64_t nb01;
    uint64_t nb02;
    uint64_t nb03;
    int32_t  ne10;
    int32_t  ne11;
    int32_t  ne12;
    int32_t  ne13;
    uint64_t nb10;
    uint64_t nb11;
    uint64_t nb12;
    uint64_t nb13;
    int32_t  ne0;
    int32_t  ne1;
    int32_t  ne2;
    int32_t  ne3;
    uint64_t nb0;
    uint64_t nb1;
    uint64_t nb2;
    uint64_t nb3;
} ggml_metal_kargs_solve_tri;

typedef struct {  // 类型定义
    int32_t  ne00t;
    int32_t  ne00;
    uint64_t nb01;
    uint64_t nb02;
    uint64_t nb03;
    int32_t  ne10;
    uint64_t nb10;
    uint64_t nb11;
    uint64_t nb12;
    uint64_t nb1;
    uint64_t nb2;
    uint64_t nb3;
} ggml_metal_kargs_get_rows;

typedef struct {  // 类型定义
    int32_t  nk0;
    int32_t  ne01;
    uint64_t nb01;
    uint64_t nb02;
    uint64_t nb03;
    int32_t  ne11;
    int32_t  ne12;
    uint64_t nb10;
    uint64_t nb11;
    uint64_t nb12;
    uint64_t nb1;
    uint64_t nb2;
    uint64_t nb3;
} ggml_metal_kargs_set_rows;

typedef struct {  // 类型定义
    int32_t  ne00;
    int32_t  ne01;
    int32_t  ne02;
    int32_t  ne03;
    uint64_t nb00;
    uint64_t nb01;
    uint64_t nb02;
    uint64_t nb03;
    int32_t  ne0;
    int32_t  ne1;
    int32_t  ne2;
    int32_t  ne3;
    uint64_t nb0;
    uint64_t nb1;
    uint64_t nb2;
    uint64_t nb3;
} ggml_metal_kargs_diag;

typedef struct {  // 类型定义
    int64_t  ne00;
    int64_t  ne01;
    int64_t  ne02;
    int64_t  ne03;
    uint64_t nb00;
    uint64_t nb01;
    uint64_t nb02;
    uint64_t nb03;
    int64_t  ne0;
    int64_t  ne1;
    int64_t  ne2;
    int64_t  ne3;
    uint64_t nb0;
    uint64_t nb1;
    uint64_t nb2;
    uint64_t nb3;
    float    sf0;
    float    sf1;
    float    sf2;
    float    sf3;
    float    poffs;
} ggml_metal_kargs_upscale;

typedef struct {  // 类型定义
    int64_t  ne00;
    int64_t  ne01;
    int64_t  ne02;
    int64_t  ne03;
    uint64_t nb00;
    uint64_t nb01;
    uint64_t nb02;
    uint64_t nb03;
    int64_t  ne0;
    int64_t  ne1;
    int64_t  ne2;
    int64_t  ne3;
    uint64_t nb0;
    uint64_t nb1;
    uint64_t nb2;
    uint64_t nb3;
} ggml_metal_kargs_pad;

typedef struct {  // 类型定义
    int64_t  ne00;
    int64_t  ne01;
    int64_t  ne02;
    int64_t  ne03;
    uint64_t nb00;
    uint64_t nb01;
    uint64_t nb02;
    uint64_t nb03;
    int64_t  ne0;
    int64_t  ne1;
    int64_t  ne2;
    int64_t  ne3;
    uint64_t nb0;
    uint64_t nb1;
    uint64_t nb2;
    uint64_t nb3;
    int32_t  p0;
    int32_t  p1;
} ggml_metal_kargs_pad_reflect_1d;

typedef struct {  // 类型定义
    int64_t  ne00;
    int64_t  ne01;
    int64_t  ne02;
    int64_t  ne03;
    uint64_t nb00;
    uint64_t nb01;
    uint64_t nb02;
    uint64_t nb03;
    int64_t  ne0;
    int64_t  ne1;
    int64_t  ne2;
    int64_t  ne3;
    uint64_t nb0;
    uint64_t nb1;
    uint64_t nb2;
    uint64_t nb3;
    int32_t  s0;
    int32_t  s1;
    int32_t  s2;
    int32_t  s3;
} ggml_metal_kargs_roll;

typedef struct {  // 类型定义
    uint64_t nb1;
    int      dim;
    int      max_period;
} ggml_metal_kargs_timestep_embedding;

typedef struct {  // 类型定义
    int32_t  ne00;
    int32_t  ne01;
    int32_t  ne02;
    int32_t  ne03;
    uint64_t nb00;
    uint64_t nb01;
    uint64_t nb02;
    uint64_t nb03;
    int32_t  ne0;
    int32_t  ne1;
    int32_t  ne2;
    int32_t  ne3;
    uint64_t nb0;
    uint64_t nb1;
    uint64_t nb2;
    uint64_t nb3;
} ggml_metal_kargs_tri;

typedef struct {  // 类型定义
    int32_t  ne00;
    int32_t  ne01;
    int32_t  ne02;
    int32_t  ne03;
    uint64_t nb00;
    uint64_t nb01;
    uint64_t nb02;
    uint64_t nb03;
    int32_t  ne0;
    int32_t  ne1;
    int32_t  ne2;
    int32_t  ne3;
    int32_t  top_k;
} ggml_metal_kargs_argsort;

typedef struct {  // 类型定义
    int64_t  ne00;
    int64_t  ne01;
    int64_t  ne02;
    int64_t  ne03;
    uint64_t nb00;
    uint64_t nb01;
    uint64_t nb02;
    uint64_t nb03;
    int32_t  ne0;
    int32_t  ne1;
    int32_t  ne2;
    int32_t  ne3;
    int32_t  top_k;
    int32_t  len;
} ggml_metal_kargs_argsort_merge;

typedef struct {  // 类型定义
    int64_t  ne0;
    float    start;
    float    step;
} ggml_metal_kargs_arange;

typedef struct {  // 类型定义
    int64_t val;
} ggml_metal_kargs_memset;

typedef struct {  // 类型定义
    int32_t  ne00;
    int32_t  ne01;
    int32_t  ne02;
    int32_t  ne03;
    uint64_t nb00;
    uint64_t nb01;
    uint64_t nb02;
    uint64_t nb03;
    uint64_t nb10;
    uint64_t nb11;
    uint64_t nb12;
    uint64_t nb13;
} ggml_metal_kargs_count_equal;

typedef struct {  // 类型定义
    int32_t  k0;
    int32_t  k1;
    int32_t  s0;
    int32_t  s1;
    int32_t  p0;
    int32_t  p1;
    int64_t  IH;
    int64_t  IW;
    int64_t  OH;
    int64_t  OW;
    int64_t  np;
} ggml_metal_kargs_pool_2d;

typedef struct {  // 类型定义
    int32_t  k0;
    int32_t  s0;
    int32_t  p0;
    int64_t  IW;
    int64_t  OW;
    int64_t  np;
} ggml_metal_kargs_pool_1d;

typedef struct {  // 类型定义
     int64_t ne00;
    uint64_t nb01;
} ggml_metal_kargs_argmax;

typedef struct {  // 类型定义
    int64_t  np;
} ggml_metal_kargs_opt_step_adamw;

typedef struct {  // 类型定义
    int64_t  np;
} ggml_metal_kargs_opt_step_sgd;

#endif // GGML_METAL_IMPL  // 条件编译结束
