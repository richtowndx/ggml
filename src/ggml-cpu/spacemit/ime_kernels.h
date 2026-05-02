#pragma once  // 防止重复包含

#include <cstddef>  // 引入 cstddef 头文件

namespace sqnbitgemm_spacemit_ime {  // 命名空间
namespace ime1 {  // 命名空间
size_t gemm_kernel_i8i4(size_t            blk_len,
                        const std::byte * quant_a_ptr,
                        const std::byte * quant_b_data,
                        const float *     quant_b_scale,
                        const std::byte * quant_b_zp,
                        float *           c_ptr,
                        size_t            count_m,
                        size_t            count_n,
                        size_t            count_k,
                        size_t            block_count_k,
                        size_t            ldc,
                        const float *     bias,
                        const size_t      scale_stride);

void quantize_a_row_i8(size_t blk_len, const float * a_ptr, size_t count_k, std::byte * quant_a_ptr);  // quantize_a_row_i8

void quantize_a_4row_i8(size_t blk_len, const float * a_ptr, size_t count_k, std::byte * quant_a_ptr);  // quantize_a_4row_i8

}  // namespace ime1
}  // namespace sqnbitgemm_spacemit_ime
