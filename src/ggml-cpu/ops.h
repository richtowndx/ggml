#pragma once  // 防止重复包含

#include "ggml.h"  // 引入 ggml.h 头文件

//
// cache line
//

#if defined(__cpp_lib_hardware_interference_size)  // 条件编译
#define CACHE_LINE_SIZE std::hardware_destructive_interference_size  // 宏定义 CACHE_LINE_SIZE
#else  // 否则
#if defined(__POWER9_VECTOR__)  // 条件编译
#define CACHE_LINE_SIZE 128  // 宏定义 CACHE_LINE_SIZE
#elif defined(__VXE__) || defined(__VXE2__)  // 否则如果
#define CACHE_LINE_SIZE 256  // 宏定义 CACHE_LINE_SIZE
#else  // 否则
#define CACHE_LINE_SIZE 64  // 宏定义 CACHE_LINE_SIZE
#endif  // 条件编译结束
#endif  // 条件编译结束

static const size_t CACHE_LINE_SIZE_F32 = CACHE_LINE_SIZE/sizeof(float);

// Work buffer size for im2col operations in CONV2D
#define GGML_IM2COL_WORK_SIZE (16 * 1024 * 1024)  // 宏定义 GGML_IM2COL_WORK_SIZE

#ifdef __cplusplus  // 如果定义了 __cplusplus 则编译
extern "C" {  // C 链接声明
#endif  // 条件编译结束

void ggml_compute_forward_dup(const struct ggml_compute_params * params, struct ggml_tensor * dst);  // ggml_compute_forward_dup
void ggml_compute_forward_add(const struct ggml_compute_params * params, struct ggml_tensor * dst);  // ggml_compute_forward_add
void ggml_compute_forward_add_id(const struct ggml_compute_params * params, struct ggml_tensor * dst);  // ggml_compute_forward_add_id
void ggml_compute_forward_add1(const struct ggml_compute_params * params, struct ggml_tensor * dst);  // ggml_compute_forward_add1
void ggml_compute_forward_acc(const struct ggml_compute_params * params, struct ggml_tensor * dst);  // ggml_compute_forward_acc
void ggml_compute_forward_sum(const struct ggml_compute_params * params, struct ggml_tensor * dst);  // ggml_compute_forward_sum
void ggml_compute_forward_sum_rows(const struct ggml_compute_params * params, struct ggml_tensor * dst);  // ggml_compute_forward_sum_rows
void ggml_compute_forward_cumsum(const struct ggml_compute_params * params, struct ggml_tensor * dst);  // ggml_compute_forward_cumsum
void ggml_compute_forward_mean(const struct ggml_compute_params * params, struct ggml_tensor * dst);  // ggml_compute_forward_mean
void ggml_compute_forward_argmax(const struct ggml_compute_params * params, struct ggml_tensor * dst);  // ggml_compute_forward_argmax
void ggml_compute_forward_count_equal(const struct ggml_compute_params * params, struct ggml_tensor * dst);  // ggml_compute_forward_count_equal
void ggml_compute_forward_repeat(const struct ggml_compute_params * params, struct ggml_tensor * dst);  // ggml_compute_forward_repeat
void ggml_compute_forward_repeat_back(const struct ggml_compute_params * params, struct ggml_tensor * dst);  // ggml_compute_forward_repeat_back
void ggml_compute_forward_concat(const struct ggml_compute_params * params, struct ggml_tensor * dst);  // ggml_compute_forward_concat
void ggml_compute_forward_silu_back(const struct ggml_compute_params * params, struct ggml_tensor * dst);  // ggml_compute_forward_silu_back
void ggml_compute_forward_norm(const struct ggml_compute_params * params, struct ggml_tensor * dst);  // ggml_compute_forward_norm
void ggml_compute_forward_rms_norm(const struct ggml_compute_params * params, struct ggml_tensor * dst);  // ggml_compute_forward_rms_norm
void ggml_compute_forward_rms_norm_back(const struct ggml_compute_params * params, struct ggml_tensor * dst);  // ggml_compute_forward_rms_norm_back
void ggml_compute_forward_group_norm(const struct ggml_compute_params * params, struct ggml_tensor * dst);  // ggml_compute_forward_group_norm
void ggml_compute_forward_l2_norm(const struct ggml_compute_params * params, struct ggml_tensor * dst);  // ggml_compute_forward_l2_norm
void ggml_compute_forward_out_prod(const struct ggml_compute_params * params, struct ggml_tensor * dst);  // ggml_compute_forward_out_prod
void ggml_compute_forward_scale(const struct ggml_compute_params * params, struct ggml_tensor * dst);  // ggml_compute_forward_scale
void ggml_compute_forward_set(const struct ggml_compute_params * params, struct ggml_tensor * dst);  // ggml_compute_forward_set
void ggml_compute_forward_cpy(const struct ggml_compute_params * params, struct ggml_tensor * dst);  // ggml_compute_forward_cpy
void ggml_compute_forward_cont(const struct ggml_compute_params * params, struct ggml_tensor * dst);  // ggml_compute_forward_cont
void ggml_compute_forward_get_rows(const struct ggml_compute_params * params, struct ggml_tensor * dst);  // ggml_compute_forward_get_rows
void ggml_compute_forward_get_rows_back(const struct ggml_compute_params * params, struct ggml_tensor * dst);  // ggml_compute_forward_get_rows_back
void ggml_compute_forward_set_rows(const struct ggml_compute_params * params, struct ggml_tensor * dst);  // ggml_compute_forward_set_rows
void ggml_compute_forward_diag(const struct ggml_compute_params * params, struct ggml_tensor * dst);  // ggml_compute_forward_diag
void ggml_compute_forward_diag_mask_inf(const struct ggml_compute_params * params, struct ggml_tensor * dst);  // ggml_compute_forward_diag_mask_inf
void ggml_compute_forward_diag_mask_zero(const struct ggml_compute_params * params, struct ggml_tensor * dst);  // ggml_compute_forward_diag_mask_zero
void ggml_compute_forward_soft_max(const struct ggml_compute_params * params, struct ggml_tensor * dst);  // ggml_compute_forward_soft_max
void ggml_compute_forward_soft_max_ext_back(const struct ggml_compute_params * params, struct ggml_tensor * dst);  // ggml_compute_forward_soft_max_ext_back
void ggml_compute_forward_rope(const struct ggml_compute_params * params, struct ggml_tensor * dst);  // ggml_compute_forward_rope
void ggml_compute_forward_rope_back(const struct ggml_compute_params * params, struct ggml_tensor * dst);  // ggml_compute_forward_rope_back
void ggml_compute_forward_clamp(const struct ggml_compute_params * params, struct ggml_tensor * dst);  // ggml_compute_forward_clamp
void ggml_compute_forward_conv_transpose_1d(const struct ggml_compute_params * params, struct ggml_tensor * dst);  // ggml_compute_forward_conv_transpose_1d
void ggml_compute_forward_im2col(const struct ggml_compute_params * params, struct ggml_tensor * dst);  // ggml_compute_forward_im2col
void ggml_compute_forward_im2col_back_f32(const struct ggml_compute_params * params, struct ggml_tensor * dst);  // ggml_compute_forward_im2col_back_f32
void ggml_compute_forward_im2col_3d(const struct ggml_compute_params * params, struct ggml_tensor * dst);  // ggml_compute_forward_im2col_3d
void ggml_compute_forward_conv_2d(const struct ggml_compute_params * params, struct ggml_tensor * dst);  // ggml_compute_forward_conv_2d
void ggml_compute_forward_conv_3d(const struct ggml_compute_params * params, struct ggml_tensor * dst);  // ggml_compute_forward_conv_3d
void ggml_compute_forward_conv_transpose_2d(const struct ggml_compute_params * params, struct ggml_tensor * dst);  // ggml_compute_forward_conv_transpose_2d
void ggml_compute_forward_conv_2d_dw(const struct ggml_compute_params * params, struct ggml_tensor * dst);  // ggml_compute_forward_conv_2d_dw
void ggml_compute_forward_pool_1d(const struct ggml_compute_params * params, struct ggml_tensor * dst);  // ggml_compute_forward_pool_1d
void ggml_compute_forward_pool_2d(const struct ggml_compute_params * params, struct ggml_tensor * dst);  // ggml_compute_forward_pool_2d
void ggml_compute_forward_pool_2d_back(const struct ggml_compute_params * params, struct ggml_tensor * dst);  // ggml_compute_forward_pool_2d_back
void ggml_compute_forward_upscale(const struct ggml_compute_params * params, struct ggml_tensor * dst);  // ggml_compute_forward_upscale
void ggml_compute_forward_pad(const struct ggml_compute_params * params, struct ggml_tensor * dst);  // ggml_compute_forward_pad
void ggml_compute_forward_pad_reflect_1d(const struct ggml_compute_params * params, struct ggml_tensor * dst);  // ggml_compute_forward_pad_reflect_1d
void ggml_compute_forward_roll(const struct ggml_compute_params * params, struct ggml_tensor * dst);  // ggml_compute_forward_roll
void ggml_compute_forward_arange(const struct ggml_compute_params * params, struct ggml_tensor * dst);  // ggml_compute_forward_arange
void ggml_compute_forward_timestep_embedding(const struct ggml_compute_params * params, struct ggml_tensor * dst);  // ggml_compute_forward_timestep_embedding
void ggml_compute_forward_argsort(const struct ggml_compute_params * params, struct ggml_tensor * dst);  // ggml_compute_forward_argsort
void ggml_compute_forward_top_k(const struct ggml_compute_params * params, struct ggml_tensor * dst);  // ggml_compute_forward_top_k
void ggml_compute_forward_leaky_relu(const struct ggml_compute_params * params, struct ggml_tensor * dst);  // ggml_compute_forward_leaky_relu
void ggml_compute_forward_tri(const struct ggml_compute_params * params, struct ggml_tensor * dst);  // ggml_compute_forward_tri
void ggml_compute_forward_fill(const struct ggml_compute_params * params, struct ggml_tensor * dst);  // ggml_compute_forward_fill
void ggml_compute_forward_flash_attn_ext(const struct ggml_compute_params * params, struct ggml_tensor * dst);  // ggml_compute_forward_flash_attn_ext
void ggml_compute_forward_flash_attn_back(
        const struct ggml_compute_params * params,
        const bool masked,
        struct ggml_tensor * dst);
void ggml_compute_forward_ssm_conv(const struct ggml_compute_params * params, struct ggml_tensor * dst);  // ggml_compute_forward_ssm_conv
void ggml_compute_forward_ssm_scan(const struct ggml_compute_params * params, struct ggml_tensor * dst);  // ggml_compute_forward_ssm_scan
void ggml_compute_forward_win_part(const struct ggml_compute_params * params, struct ggml_tensor * dst);  // ggml_compute_forward_win_part
void ggml_compute_forward_win_unpart(const struct ggml_compute_params * params, struct ggml_tensor * dst);  // ggml_compute_forward_win_unpart
void ggml_compute_forward_unary(const struct ggml_compute_params * params, struct ggml_tensor * dst);  // ggml_compute_forward_unary
void ggml_compute_forward_glu(const struct ggml_compute_params * params, struct ggml_tensor * dst);  // ggml_compute_forward_glu
void ggml_compute_forward_get_rel_pos(const struct ggml_compute_params * params, struct ggml_tensor * dst);  // ggml_compute_forward_get_rel_pos
void ggml_compute_forward_add_rel_pos(const struct ggml_compute_params * params, struct ggml_tensor * dst);  // ggml_compute_forward_add_rel_pos
void ggml_compute_forward_rwkv_wkv6(const struct ggml_compute_params * params, struct ggml_tensor * dst);  // ggml_compute_forward_rwkv_wkv6
void ggml_compute_forward_rwkv_wkv7(const struct ggml_compute_params * params, struct ggml_tensor * dst);  // ggml_compute_forward_rwkv_wkv7
void ggml_compute_forward_solve_tri(const struct ggml_compute_params * params, struct ggml_tensor * dst);  // ggml_compute_forward_solve_tri
void ggml_compute_forward_gla(const struct ggml_compute_params * params, struct ggml_tensor * dst);  // ggml_compute_forward_gla
void ggml_compute_forward_gated_delta_net(const struct ggml_compute_params * params, struct ggml_tensor * dst);  // ggml_compute_forward_gated_delta_net
void ggml_compute_forward_map_custom1(const struct ggml_compute_params * params, struct ggml_tensor * dst);  // ggml_compute_forward_map_custom1
void ggml_compute_forward_map_custom2(const struct ggml_compute_params * params, struct ggml_tensor * dst);  // ggml_compute_forward_map_custom2
void ggml_compute_forward_map_custom3(const struct ggml_compute_params * params, struct ggml_tensor * dst);  // ggml_compute_forward_map_custom3
void ggml_compute_forward_custom(const struct ggml_compute_params * params, struct ggml_tensor * dst);  // ggml_compute_forward_custom
void ggml_compute_forward_cross_entropy_loss(const struct ggml_compute_params * params, struct ggml_tensor * dst);  // ggml_compute_forward_cross_entropy_loss
void ggml_compute_forward_cross_entropy_loss_back(const struct ggml_compute_params * params, struct ggml_tensor * dst);  // ggml_compute_forward_cross_entropy_loss_back
void ggml_compute_forward_opt_step_adamw(const struct ggml_compute_params * params, struct ggml_tensor * dst);  // ggml_compute_forward_opt_step_adamw
void ggml_compute_forward_mul_mat(const struct ggml_compute_params * params, struct ggml_tensor * dst);  // ggml_compute_forward_mul_mat
void ggml_compute_forward_opt_step_sgd(const struct ggml_compute_params * params, struct ggml_tensor * dst);  // ggml_compute_forward_opt_step_sgd
#ifdef __cplusplus  // 如果定义了 __cplusplus 则编译
}
#endif  // 条件编译结束
