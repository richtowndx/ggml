
#include "../node_context.h"  // 引入 ../node_context.h 头文件
#include "../op_table.h"  // 引入 ../op_table.h 头文件
#include "../utils.h"  // 引入 ../utils.h 头文件

#include <climits>  // 引入 climits 头文件
#include <cstdint>  // 引入 cstdint 头文件
#include <memory>  // 引入 memory 头文件
#include <openvino/op/reshape.hpp>  // 引入 openvino/op/reshape.hpp 头文件
#include <openvino/op/slice.hpp>  // 引入 openvino/op/slice.hpp 头文件
#include <vector>  // 引入 vector 头文件

namespace ov {  // 命名空间
namespace frontend {  // 命名空间
namespace ggml {  // 命名空间
namespace op {  // 命名空间

OutputVector translate_cont(const NodeContext & context) {
    num_inputs_check(context, 1, 1);

    int op_case = context.get_op_case();
    FRONT_END_CHECK_IMPLEMENTED(op_case == 1 || op_case == 2 || op_case == 3, "Unsupported CONT case");

    auto src_shape = context.get_input_shape(0).to_shape();
    auto dst_shape = context.get_output_shape().to_shape();
    ov::Output<Node> res;

    if (op_case == 1) {
        // The input comes from a PERMUTE
        throw std::runtime_error("Code of this case might be outdated");
        dst_shape[1] = -1;
        res = std::make_shared<ov::op::v1::Reshape>(
            context.get_input(0), ov::op::v0::Constant::create(ov::element::i64, {dst_shape.size()}, dst_shape), false);
    } else if (op_case == 2) {
        // The input comes from a TRANSPOSE
        return {context.get_input(0)};  // 返回
    } else {
        // The input comes from a VIEW
        res = process_view_input(context, 0);
    }

    return rename_outputs_with_suffix({res}, context.get_name());  // 返回
}

}  // namespace op
}  // namespace ggml
}  // namespace frontend
}  // namespace ov
