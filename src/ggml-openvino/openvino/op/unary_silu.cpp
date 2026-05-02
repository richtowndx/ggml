#include "../node_context.h"  // 引入 ../node_context.h 头文件
#include "../op_table.h"  // 引入 ../op_table.h 头文件
#include "../utils.h"  // 引入 ../utils.h 头文件

#include <openvino/core/node_output.hpp>  // 引入 openvino/core/node_output.hpp 头文件
#include <openvino/op/multiply.hpp>  // 引入 openvino/op/multiply.hpp 头文件
#include <openvino/op/sigmoid.hpp>  // 引入 openvino/op/sigmoid.hpp 头文件

namespace ov {  // 命名空间
namespace frontend {  // 命名空间
namespace ggml {  // 命名空间
namespace op {  // 命名空间

OutputVector translate_unary_silu(const NodeContext & context) {
    num_inputs_check(context, 1, 1);

    auto input = context.get_input(0);
    auto sigmoid = std::make_shared<ov::op::v0::Sigmoid>(input);
    auto res = std::make_shared<ov::op::v1::Multiply>(input, sigmoid);

    return rename_outputs_with_suffix({res}, context.get_name());  // 返回
}

}  // namespace op
}  // namespace ggml
}  // namespace frontend
}  // namespace ov
