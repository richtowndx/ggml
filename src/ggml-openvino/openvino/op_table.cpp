#include "op_table.h"  // 引入 op_table.h 头文件

#include "utils.h"  // 引入 utils.h 头文件

#include <openvino/op/add.hpp>  // 引入 openvino/op/add.hpp 头文件
#include <openvino/op/divide.hpp>  // 引入 openvino/op/divide.hpp 头文件
#include <openvino/op/gather.hpp>  // 引入 openvino/op/gather.hpp 头文件
#include <openvino/op/matmul.hpp>  // 引入 openvino/op/matmul.hpp 头文件
#include <openvino/op/multiply.hpp>  // 引入 openvino/op/multiply.hpp 头文件
#include <openvino/op/subtract.hpp>  // 引入 openvino/op/subtract.hpp 头文件

namespace ov {  // 命名空间
namespace frontend {  // 命名空间
namespace ggml {  // 命名空间

std::unordered_map<std::string, CreatorFunction> get_supported_ops() {
    using namespace ov::op;  // using 声明
    return {  // 返回
        {"GGML_OP_ADD",            op::translate_1to1_match_2_inputs<v1::Add>     },
        {"GGML_OP_ADD1",           op::translate_1to1_match_2_inputs<v1::Add>     },
        {"GGML_OP_CONT",           op::translate_cont                             },
        {"GGML_OP_DIV",            op::translate_1to1_match_2_inputs<v1::Divide>  },
        {"GGML_OP_GET_ROWS",       op::translate_get_rows                         },
        {"GGML_OP_MUL",            op::translate_1to1_match_2_inputs<v1::Multiply>},
        {"GGML_OP_MUL_MAT",        op::translate_mulmat                           },
        {"GGML_OP_PERMUTE",        op::translate_permute                          },
        {"GGML_OP_RESHAPE",        op::translate_reshape                          },
        {"GGML_OP_RMS_NORM",       op::translate_rms_norm                         },
        {"GGML_OP_ROPE",           op::translate_rope                             },
        {"GGML_OP_SCALE",          op::translate_scale                            },
        {"GGML_OP_SOFT_MAX",       op::translate_soft_max                         },
        {"GGML_OP_SUB",            op::translate_1to1_match_2_inputs<v1::Subtract>},
        {"GGML_OP_TRANSPOSE",      op::translate_transpose                        },
        {"GGML_UNARY_OP_GELU",     op::translate_unary_gelu                       },
        {"GGML_UNARY_OP_SILU",     op::translate_unary_silu                       },
        {"GGML_OP_VIEW",           op::translate_view                             },
        {"GGML_GLU_OP_SWIGLU",     op::translate_glu_swiglu                       },
        {"GGML_GLU_OP_GEGLU",      op::translate_glu_geglu                        },
        {"GGML_OP_SET_ROWS",       op::translate_set_rows                         },
        {"GGML_OP_CPY",            op::translate_cpy                              },
        {"GGML_OP_FLASH_ATTN_EXT", op::translate_flash_attn_ext                   },
    };
}

}  // namespace ggml
}  // namespace frontend
}  // namespace ov
