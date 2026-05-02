#pragma once  // 防止重复包含

#include "ggml.h"  // 引入 ggml.h 头文件

#ifdef __cplusplus  // 如果定义了 __cplusplus 则编译
extern "C" {  // C 链接声明
#endif  // 条件编译结束

struct ggml_metal_buffer_id {  // 结构体定义
    void * metal; // id<MTLBuffer>
    size_t offs;
};

typedef struct ggml_metal_device * ggml_metal_device_t;  // 类型定义

//
// MTLFunctionConstantValues wrapper
//

typedef struct ggml_metal_cv * ggml_metal_cv_t;  // 类型定义

ggml_metal_cv_t ggml_metal_cv_init(void);  // ggml_metal_cv_init
void ggml_metal_cv_free(ggml_metal_cv_t cv);  // ggml_metal_cv_free

void ggml_metal_cv_set_int16(ggml_metal_cv_t cv, int16_t value, int32_t idx);  // ggml_metal_cv_set_int16
void ggml_metal_cv_set_int32(ggml_metal_cv_t cv, int32_t value, int32_t idx);  // ggml_metal_cv_set_int32
void ggml_metal_cv_set_bool (ggml_metal_cv_t cv, bool    value, int32_t idx);  // ggml_metal_cv_set_bool

//
// MTLComputePipelineState wrapper
//

typedef struct ggml_metal_pipeline * ggml_metal_pipeline_t;  // 类型定义

ggml_metal_pipeline_t ggml_metal_pipeline_init(void);  // ggml_metal_pipeline_init
void ggml_metal_pipeline_free(ggml_metal_pipeline_t pipeline);  // ggml_metal_pipeline_free

// a collection of pipelines
typedef struct ggml_metal_pipelines * ggml_metal_pipelines_t;  // 类型定义

ggml_metal_pipelines_t ggml_metal_pipelines_init(void);  // ggml_metal_pipelines_init
void ggml_metal_pipelines_free(ggml_metal_pipelines_t ppls);  // ggml_metal_pipelines_free

void                  ggml_metal_pipelines_add(ggml_metal_pipelines_t ppls, const char * name, ggml_metal_pipeline_t pipeline);  // ggml_metal_pipelines_add
ggml_metal_pipeline_t ggml_metal_pipelines_get(ggml_metal_pipelines_t ppls, const char * name);  // ggml_metal_pipelines_get

struct ggml_metal_pipeline_with_params {  // 结构体定义
    ggml_metal_pipeline_t pipeline;

    int nsg;

    int nr0;
    int nr1;

    size_t smem;

    bool c4;
    bool cnt;
};

int ggml_metal_pipeline_max_theads_per_threadgroup(struct ggml_metal_pipeline_with_params pipeline);  // ggml_metal_pipeline_max_theads_per_threadgroup

//
// MTLCommandBuffer wrapper
//

typedef void * ggml_metal_cmd_buf_t;  // 类型定义

//
// MTLComputeCommandEncoder wrapper
//

typedef struct ggml_metal_encoder * ggml_metal_encoder_t;  // 类型定义

ggml_metal_encoder_t ggml_metal_encoder_init(ggml_metal_cmd_buf_t cmd_buf_raw, bool concurrent);  // ggml_metal_encoder_init
void ggml_metal_encoder_free(ggml_metal_encoder_t encoder);  // ggml_metal_encoder_free

void ggml_metal_encoder_debug_group_push(ggml_metal_encoder_t encoder, const char * name);  // ggml_metal_encoder_debug_group_push
void ggml_metal_encoder_debug_group_pop (ggml_metal_encoder_t encoder);  // ggml_metal_encoder_debug_group_pop

void ggml_metal_encoder_set_pipeline(ggml_metal_encoder_t encoder, struct ggml_metal_pipeline_with_params pipeline);  // ggml_metal_encoder_set_pipeline

void ggml_metal_encoder_set_bytes (ggml_metal_encoder_t encoder, void * data, size_t size, int idx);  // ggml_metal_encoder_set_bytes
void ggml_metal_encoder_set_buffer(ggml_metal_encoder_t encoder, struct ggml_metal_buffer_id buffer, int idx);  // ggml_metal_encoder_set_buffer

void ggml_metal_encoder_set_threadgroup_memory_size(ggml_metal_encoder_t encoder, size_t size, int idx);  // ggml_metal_encoder_set_threadgroup_memory_size

void ggml_metal_encoder_dispatch_threadgroups(ggml_metal_encoder_t encoder, int tg0, int tg1, int tg2, int tptg0, int tptg1, int tptg2);  // ggml_metal_encoder_dispatch_threadgroups

void ggml_metal_encoder_memory_barrier(ggml_metal_encoder_t encoder);  // ggml_metal_encoder_memory_barrier

void ggml_metal_encoder_end_encoding(ggml_metal_encoder_t encoder);  // ggml_metal_encoder_end_encoding

//
// MTLLibrary wrapper
//

typedef struct ggml_metal_library * ggml_metal_library_t;  // 类型定义

ggml_metal_library_t ggml_metal_library_init            (ggml_metal_device_t dev);  // ggml_metal_library_init
ggml_metal_library_t ggml_metal_library_init_from_source(ggml_metal_device_t dev, const char * source, bool verbose);  // ggml_metal_library_init_from_source

void ggml_metal_library_free(ggml_metal_library_t lib);  // ggml_metal_library_free

ggml_metal_device_t ggml_metal_library_get_device(ggml_metal_library_t lib);  // ggml_metal_library_get_device

struct ggml_metal_pipeline_with_params ggml_metal_library_get_pipeline    (ggml_metal_library_t lib, const char * name);  // ggml_metal_library_get_pipeline
struct ggml_metal_pipeline_with_params ggml_metal_library_compile_pipeline(ggml_metal_library_t lib, const char * base, const char * name, ggml_metal_cv_t cv);  // ggml_metal_library_compile_pipeline

struct ggml_metal_pipeline_with_params ggml_metal_library_get_pipeline_base              (ggml_metal_library_t lib, enum ggml_op op);  // ggml_metal_library_get_pipeline_base
struct ggml_metal_pipeline_with_params ggml_metal_library_get_pipeline_cpy               (ggml_metal_library_t lib, enum ggml_type tsrc, enum ggml_type tdst);  // ggml_metal_library_get_pipeline_cpy
struct ggml_metal_pipeline_with_params ggml_metal_library_get_pipeline_pool_1d           (ggml_metal_library_t lib, const struct ggml_tensor * op, enum ggml_op_pool op_pool);  // ggml_metal_library_get_pipeline_pool_1d
struct ggml_metal_pipeline_with_params ggml_metal_library_get_pipeline_pool_2d           (ggml_metal_library_t lib, const struct ggml_tensor * op, enum ggml_op_pool op_pool);  // ggml_metal_library_get_pipeline_pool_2d
struct ggml_metal_pipeline_with_params ggml_metal_library_get_pipeline_get_rows          (ggml_metal_library_t lib, enum ggml_type tsrc);  // ggml_metal_library_get_pipeline_get_rows
struct ggml_metal_pipeline_with_params ggml_metal_library_get_pipeline_set_rows          (ggml_metal_library_t lib, enum ggml_type tidx, enum ggml_type tdst);  // ggml_metal_library_get_pipeline_set_rows
struct ggml_metal_pipeline_with_params ggml_metal_library_get_pipeline_diag              (ggml_metal_library_t lib, const struct ggml_tensor * op);  // ggml_metal_library_get_pipeline_diag
struct ggml_metal_pipeline_with_params ggml_metal_library_get_pipeline_repeat            (ggml_metal_library_t lib, enum ggml_type tsrc);  // ggml_metal_library_get_pipeline_repeat
struct ggml_metal_pipeline_with_params ggml_metal_library_get_pipeline_unary             (ggml_metal_library_t lib, const struct ggml_tensor * op);  // ggml_metal_library_get_pipeline_unary
struct ggml_metal_pipeline_with_params ggml_metal_library_get_pipeline_glu               (ggml_metal_library_t lib, const struct ggml_tensor * op);  // ggml_metal_library_get_pipeline_glu
struct ggml_metal_pipeline_with_params ggml_metal_library_get_pipeline_sum               (ggml_metal_library_t lib, const struct ggml_tensor * op);  // ggml_metal_library_get_pipeline_sum
struct ggml_metal_pipeline_with_params ggml_metal_library_get_pipeline_sum_rows          (ggml_metal_library_t lib, const struct ggml_tensor * op);  // ggml_metal_library_get_pipeline_sum_rows
struct ggml_metal_pipeline_with_params ggml_metal_library_get_pipeline_cumsum_blk        (ggml_metal_library_t lib, const struct ggml_tensor * op);  // ggml_metal_library_get_pipeline_cumsum_blk
struct ggml_metal_pipeline_with_params ggml_metal_library_get_pipeline_cumsum_add        (ggml_metal_library_t lib, const struct ggml_tensor * op);  // ggml_metal_library_get_pipeline_cumsum_add
struct ggml_metal_pipeline_with_params ggml_metal_library_get_pipeline_tri               (ggml_metal_library_t lib, const struct ggml_tensor * op);  // ggml_metal_library_get_pipeline_tri
struct ggml_metal_pipeline_with_params ggml_metal_library_get_pipeline_soft_max          (ggml_metal_library_t lib, const struct ggml_tensor * op);  // ggml_metal_library_get_pipeline_soft_max
struct ggml_metal_pipeline_with_params ggml_metal_library_get_pipeline_ssm_conv          (ggml_metal_library_t lib, const struct ggml_tensor * op);  // ggml_metal_library_get_pipeline_ssm_conv
struct ggml_metal_pipeline_with_params ggml_metal_library_get_pipeline_ssm_conv_batched  (ggml_metal_library_t lib, const struct ggml_tensor * op, int ssm_conv_bs);  // ggml_metal_library_get_pipeline_ssm_conv_batched
struct ggml_metal_pipeline_with_params ggml_metal_library_get_pipeline_ssm_scan          (ggml_metal_library_t lib, const struct ggml_tensor * op);  // ggml_metal_library_get_pipeline_ssm_scan
struct ggml_metal_pipeline_with_params ggml_metal_library_get_pipeline_rwkv              (ggml_metal_library_t lib, const struct ggml_tensor * op);  // ggml_metal_library_get_pipeline_rwkv
struct ggml_metal_pipeline_with_params ggml_metal_library_get_pipeline_gated_delta_net   (ggml_metal_library_t lib, const struct ggml_tensor * op);  // ggml_metal_library_get_pipeline_gated_delta_net
struct ggml_metal_pipeline_with_params ggml_metal_library_get_pipeline_solve_tri         (ggml_metal_library_t lib, const struct ggml_tensor * op);  // ggml_metal_library_get_pipeline_solve_tri
struct ggml_metal_pipeline_with_params ggml_metal_library_get_pipeline_mul_mv_ext        (ggml_metal_library_t lib, enum ggml_type tsrc0, enum ggml_type tsrc1, int nsg, int nxpsg, int r1ptg);  // ggml_metal_library_get_pipeline_mul_mv_ext
struct ggml_metal_pipeline_with_params ggml_metal_library_get_pipeline_mul_mm            (ggml_metal_library_t lib, const struct ggml_tensor * op);  // ggml_metal_library_get_pipeline_mul_mm
struct ggml_metal_pipeline_with_params ggml_metal_library_get_pipeline_mul_mv            (ggml_metal_library_t lib, const struct ggml_tensor * op);  // ggml_metal_library_get_pipeline_mul_mv
struct ggml_metal_pipeline_with_params ggml_metal_library_get_pipeline_mul_mm_id_map0    (ggml_metal_library_t lib, int ne02, int ne20);  // ggml_metal_library_get_pipeline_mul_mm_id_map0
struct ggml_metal_pipeline_with_params ggml_metal_library_get_pipeline_mul_mm_id         (ggml_metal_library_t lib, const struct ggml_tensor * op);  // ggml_metal_library_get_pipeline_mul_mm_id
struct ggml_metal_pipeline_with_params ggml_metal_library_get_pipeline_mul_mv_id         (ggml_metal_library_t lib, const struct ggml_tensor * op);  // ggml_metal_library_get_pipeline_mul_mv_id
struct ggml_metal_pipeline_with_params ggml_metal_library_get_pipeline_argmax            (ggml_metal_library_t lib, const struct ggml_tensor * op);  // ggml_metal_library_get_pipeline_argmax
struct ggml_metal_pipeline_with_params ggml_metal_library_get_pipeline_argsort           (ggml_metal_library_t lib, const struct ggml_tensor * op);  // ggml_metal_library_get_pipeline_argsort
struct ggml_metal_pipeline_with_params ggml_metal_library_get_pipeline_argsort_merge     (ggml_metal_library_t lib, const struct ggml_tensor * op);  // ggml_metal_library_get_pipeline_argsort_merge
struct ggml_metal_pipeline_with_params ggml_metal_library_get_pipeline_top_k             (ggml_metal_library_t lib, const struct ggml_tensor * op);  // ggml_metal_library_get_pipeline_top_k
struct ggml_metal_pipeline_with_params ggml_metal_library_get_pipeline_top_k_merge       (ggml_metal_library_t lib, const struct ggml_tensor * op);  // ggml_metal_library_get_pipeline_top_k_merge
struct ggml_metal_pipeline_with_params ggml_metal_library_get_pipeline_bin               (ggml_metal_library_t lib, const struct ggml_tensor * op, int32_t n_fuse );  // ggml_metal_library_get_pipeline_bin
struct ggml_metal_pipeline_with_params ggml_metal_library_get_pipeline_bin_one           (ggml_metal_library_t lib, enum ggml_op op);  // ggml_metal_library_get_pipeline_bin_one
struct ggml_metal_pipeline_with_params ggml_metal_library_get_pipeline_l2_norm           (ggml_metal_library_t lib, const struct ggml_tensor * op);  // ggml_metal_library_get_pipeline_l2_norm
struct ggml_metal_pipeline_with_params ggml_metal_library_get_pipeline_group_norm        (ggml_metal_library_t lib, const struct ggml_tensor * op);  // ggml_metal_library_get_pipeline_group_norm
struct ggml_metal_pipeline_with_params ggml_metal_library_get_pipeline_norm              (ggml_metal_library_t lib, const struct ggml_tensor * op, int32_t n_fuse);  // ggml_metal_library_get_pipeline_norm
struct ggml_metal_pipeline_with_params ggml_metal_library_get_pipeline_rope              (ggml_metal_library_t lib, const struct ggml_tensor * op);  // ggml_metal_library_get_pipeline_rope
struct ggml_metal_pipeline_with_params ggml_metal_library_get_pipeline_im2col            (ggml_metal_library_t lib, const struct ggml_tensor * op);  // ggml_metal_library_get_pipeline_im2col
struct ggml_metal_pipeline_with_params ggml_metal_library_get_pipeline_conv_transpose_1d (ggml_metal_library_t lib, const struct ggml_tensor * op);  // ggml_metal_library_get_pipeline_conv_transpose_1d
struct ggml_metal_pipeline_with_params ggml_metal_library_get_pipeline_conv_transpose_2d (ggml_metal_library_t lib, const struct ggml_tensor * op);  // ggml_metal_library_get_pipeline_conv_transpose_2d
struct ggml_metal_pipeline_with_params ggml_metal_library_get_pipeline_conv_2d           (ggml_metal_library_t lib, const struct ggml_tensor * op);  // ggml_metal_library_get_pipeline_conv_2d
struct ggml_metal_pipeline_with_params ggml_metal_library_get_pipeline_conv_3d           (ggml_metal_library_t lib, const struct ggml_tensor * op);  // ggml_metal_library_get_pipeline_conv_3d
struct ggml_metal_pipeline_with_params ggml_metal_library_get_pipeline_upscale           (ggml_metal_library_t lib, const struct ggml_tensor * op);  // ggml_metal_library_get_pipeline_upscale
struct ggml_metal_pipeline_with_params ggml_metal_library_get_pipeline_pad               (ggml_metal_library_t lib, const struct ggml_tensor * op);  // ggml_metal_library_get_pipeline_pad
struct ggml_metal_pipeline_with_params ggml_metal_library_get_pipeline_pad_reflect_1d    (ggml_metal_library_t lib, const struct ggml_tensor * op);  // ggml_metal_library_get_pipeline_pad_reflect_1d
struct ggml_metal_pipeline_with_params ggml_metal_library_get_pipeline_roll              (ggml_metal_library_t lib, const struct ggml_tensor * op);  // ggml_metal_library_get_pipeline_roll
struct ggml_metal_pipeline_with_params ggml_metal_library_get_pipeline_arange            (ggml_metal_library_t lib, const struct ggml_tensor * op);  // ggml_metal_library_get_pipeline_arange
struct ggml_metal_pipeline_with_params ggml_metal_library_get_pipeline_timestep_embedding(ggml_metal_library_t lib, const struct ggml_tensor * op);  // ggml_metal_library_get_pipeline_timestep_embedding
struct ggml_metal_pipeline_with_params ggml_metal_library_get_pipeline_opt_step_adamw    (ggml_metal_library_t lib, const struct ggml_tensor * op);  // ggml_metal_library_get_pipeline_opt_step_adamw
struct ggml_metal_pipeline_with_params ggml_metal_library_get_pipeline_opt_step_sgd      (ggml_metal_library_t lib, const struct ggml_tensor * op);  // ggml_metal_library_get_pipeline_opt_step_sgd
struct ggml_metal_pipeline_with_params ggml_metal_library_get_pipeline_memset            (ggml_metal_library_t lib, const struct ggml_tensor * op);  // ggml_metal_library_get_pipeline_memset
struct ggml_metal_pipeline_with_params ggml_metal_library_get_pipeline_count_equal       (ggml_metal_library_t lib, const struct ggml_tensor * op);  // ggml_metal_library_get_pipeline_count_equal

struct ggml_metal_pipeline_with_params ggml_metal_library_get_pipeline_flash_attn_ext_pad(
        ggml_metal_library_t lib,
        const struct ggml_tensor * op,
        bool    has_mask,
        int32_t ncpsg);

struct ggml_metal_pipeline_with_params ggml_metal_library_get_pipeline_flash_attn_ext_blk(
        ggml_metal_library_t lib,
        const struct ggml_tensor * op,
        int32_t nqptg,
        int32_t ncpsg);

struct ggml_metal_pipeline_with_params ggml_metal_library_get_pipeline_flash_attn_ext(
        ggml_metal_library_t lib,
        const struct ggml_tensor * op,
        bool    has_mask,
        bool    has_sinks,
        bool    has_bias,
        bool    has_scap,
        bool    has_kvpad,
        int32_t nsg);

struct ggml_metal_pipeline_with_params ggml_metal_library_get_pipeline_flash_attn_ext_vec(
        ggml_metal_library_t lib,
        const struct ggml_tensor * op,
        bool    has_mask,
        bool    has_sinks,
        bool    has_bias,
        bool    has_scap,
        bool    has_kvpad,
        int32_t nsg,
        int32_t nwg);

struct ggml_metal_pipeline_with_params ggml_metal_library_get_pipeline_flash_attn_ext_vec_reduce(
        ggml_metal_library_t lib,
        const struct ggml_tensor * op,
        int32_t dv,
        int32_t nwg);

// MTLResidencySet wrapper

typedef void * ggml_metal_rset_t;  // 类型定义

// a collection of residency sets (non-owning)
typedef struct ggml_metal_rsets * ggml_metal_rsets_t;  // 类型定义

ggml_metal_rsets_t ggml_metal_rsets_init(void);  // ggml_metal_rsets_init
void ggml_metal_rsets_free(ggml_metal_rsets_t rsets);  // ggml_metal_rsets_free

//
// device
//

struct ggml_metal_device_props {  // 结构体定义
    int device;
    char name[128];
    char desc[128];

    size_t max_buffer_size;
    size_t max_working_set_size;
    size_t max_theadgroup_memory_size;

    bool has_simdgroup_reduction;
    bool has_simdgroup_mm;
    bool has_unified_memory;
    bool has_bfloat;
    bool has_tensor;
    bool use_residency_sets;
    bool use_shared_buffers;

    bool supports_gpu_family_apple7;

    int op_offload_min_batch_size;
};

typedef struct ggml_metal_event * ggml_metal_event_t;  // 类型定义

void ggml_metal_event_encode_signal(ggml_metal_event_t ev, ggml_metal_cmd_buf_t cmd_buf);  // ggml_metal_event_encode_signal
void ggml_metal_event_encode_wait  (ggml_metal_event_t ev, ggml_metal_cmd_buf_t cmd_buf);  // ggml_metal_event_encode_wait

ggml_metal_device_t ggml_metal_device_init(int device);  // ggml_metal_device_init
void ggml_metal_device_free(ggml_metal_device_t dev);  // ggml_metal_device_free

ggml_metal_device_t ggml_metal_device_get(int device);  // ggml_metal_device_get

void * ggml_metal_device_get_obj  (ggml_metal_device_t dev); // id<MTLDevice>  // ggml_metal_device_get_obj
void * ggml_metal_device_get_queue(ggml_metal_device_t dev); // id<MTLCommandQueue>  // ggml_metal_device_get_queue

ggml_metal_library_t ggml_metal_device_get_library(ggml_metal_device_t dev);  // ggml_metal_device_get_library

void ggml_metal_device_rsets_add(ggml_metal_device_t dev, ggml_metal_rset_t rset);  // ggml_metal_device_rsets_add
void ggml_metal_device_rsets_rm (ggml_metal_device_t dev, ggml_metal_rset_t rset);  // ggml_metal_device_rsets_rm

void ggml_metal_device_rsets_keep_alive(ggml_metal_device_t dev);  // ggml_metal_device_rsets_keep_alive

ggml_metal_event_t ggml_metal_device_event_init(ggml_metal_device_t dev);  // ggml_metal_device_event_init
void ggml_metal_device_event_free(ggml_metal_device_t dev, ggml_metal_event_t ev);  // ggml_metal_device_event_free
void ggml_metal_device_event_synchronize(ggml_metal_device_t dev, ggml_metal_event_t ev);  // ggml_metal_device_event_synchronize

void ggml_metal_device_get_memory(ggml_metal_device_t dev, size_t * free, size_t * total);  // ggml_metal_device_get_memory
bool ggml_metal_device_supports_op(ggml_metal_device_t dev, const struct ggml_tensor * op);  // ggml_metal_device_supports_op

const struct ggml_metal_device_props * ggml_metal_device_get_props(ggml_metal_device_t dev);  // ggml_metal_device_get_props

//
// device buffers
//

typedef struct ggml_metal_buffer * ggml_metal_buffer_t;  // 类型定义

ggml_metal_buffer_t ggml_metal_buffer_init(ggml_metal_device_t dev, size_t size, bool shared);  // ggml_metal_buffer_init
ggml_metal_buffer_t ggml_metal_buffer_map (ggml_metal_device_t dev, void * ptr, size_t size, size_t max_tensor_size);  // ggml_metal_buffer_map

void   ggml_metal_buffer_free     (ggml_metal_buffer_t buf);  // ggml_metal_buffer_free
void * ggml_metal_buffer_get_base (ggml_metal_buffer_t buf);  // ggml_metal_buffer_get_base
bool   ggml_metal_buffer_is_shared(ggml_metal_buffer_t buf);  // ggml_metal_buffer_is_shared

void   ggml_metal_buffer_memset_tensor(ggml_metal_buffer_t buf, struct ggml_tensor * tensor, uint8_t value, size_t offset, size_t size);  // ggml_metal_buffer_memset_tensor
void   ggml_metal_buffer_set_tensor   (ggml_metal_buffer_t buf, struct ggml_tensor * tensor, const void * data, size_t offset, size_t size);  // ggml_metal_buffer_set_tensor
void   ggml_metal_buffer_get_tensor   (ggml_metal_buffer_t buf, const struct ggml_tensor * tensor, void * data, size_t offset, size_t size);  // ggml_metal_buffer_get_tensor
void   ggml_metal_buffer_clear        (ggml_metal_buffer_t buf, uint8_t value);  // ggml_metal_buffer_clear

// finds the Metal buffer that contains the tensor data on the GPU device
// the assumption is that there is 1-to-1 mapping between the host and device memory buffers, so we can find the
// Metal buffer based on the host memory pointer
//
struct ggml_metal_buffer_id ggml_metal_buffer_get_id(ggml_metal_buffer_t buf, const struct ggml_tensor * t);  // ggml_metal_buffer_get_id

#ifdef __cplusplus  // 如果定义了 __cplusplus 则编译
}
#endif  // 条件编译结束
