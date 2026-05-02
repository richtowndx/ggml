#include "../node_context.h"  // 引入 ../node_context.h 头文件
#include "../op_table.h"  // 引入 ../op_table.h 头文件
#include "../utils.h"  // 引入 ../utils.h 头文件

#include <climits>  // 引入 climits 头文件
#include <cstdint>  // 引入 cstdint 头文件
#include <memory>  // 引入 memory 头文件
#include <openvino/core/node.hpp>  // 引入 openvino/core/node.hpp 头文件
#include <openvino/core/node_output.hpp>  // 引入 openvino/core/node_output.hpp 头文件
#include <openvino/op/broadcast.hpp>  // 引入 openvino/op/broadcast.hpp 头文件
#include <openvino/op/concat.hpp>  // 引入 openvino/op/concat.hpp 头文件
#include <openvino/op/constant.hpp>  // 引入 openvino/op/constant.hpp 头文件
#include <openvino/op/convert.hpp>  // 引入 openvino/op/convert.hpp 头文件
#include <openvino/op/matmul.hpp>  // 引入 openvino/op/matmul.hpp 头文件
#include <openvino/op/reshape.hpp>  // 引入 openvino/op/reshape.hpp 头文件
#include <openvino/op/slice.hpp>  // 引入 openvino/op/slice.hpp 头文件
#include <openvino/op/transpose.hpp>  // 引入 openvino/op/transpose.hpp 头文件
#include <openvino/op/unsqueeze.hpp>  // 引入 openvino/op/unsqueeze.hpp 头文件
#include <openvino/op/util/op_types.hpp>  // 引入 openvino/op/util/op_types.hpp 头文件
#include <vector>  // 引入 vector 头文件

namespace ov {  // 命名空间
namespace frontend {  // 命名空间
namespace ggml {  // 命名空间
namespace op {  // 命名空间

OutputVector translate_mulmat(const NodeContext & context) {
    num_inputs_check(context, 2, 2);

    int op_case = context.get_op_case();

    ov::Output<Node> res;
    ov::Output<ov::Node> B = context.get_input(0);
    ov::Output<ov::Node> A = context.get_input(1);

    bool transpose_b = true;
    if (op_case == 2) {
        B = B.get_node_shared_ptr()->input_value(0);
        transpose_b = false;
    } else if (op_case == 3) {
        B = process_view_input(context, 0);
        A = process_view_input(context, 1);
    }
    if (A.get_element_type() != B.get_element_type()) {
        B = std::make_shared<ov::op::v0::Convert>(context.get_input(0), context.get_input_type(1));
    }

    auto B_shape = context.get_input_shape(0).to_shape();
    auto A_shape = context.get_input_shape(1).to_shape();
    int64_t A_batch = A_shape[1];
    int64_t B_batch = B_shape[1];

    auto A_batch_larger = A_batch > B_batch;
    auto batch_large = A_batch_larger ? A_batch : B_batch;
    auto batch_small = A_batch_larger ? B_batch : A_batch;

    Output<Node> Z = A_batch_larger ? B : A;
    int64_t factor = batch_large / batch_small;
    if (factor > 1 && batch_small > 1) {
        auto batch_large_node = ov::op::v0::Constant::create(ov::element::i64, {1}, std::vector<int64_t>{batch_large});
        auto batch_small_node = ov::op::v0::Constant::create(ov::element::i64, {1}, std::vector<int64_t>{batch_small});
        auto factor_node = ov::op::v0::Constant::create(ov::element::i64, {1}, std::vector<int64_t>{factor});

        auto unsqueeze_axes = ov::op::v0::Constant::create(ov::element::i64, Shape{}, {2});
        auto Z_unsqueezed = std::make_shared<ov::op::v0::Unsqueeze>(Z, unsqueeze_axes);

        auto broadcast_shape = ov::op::v0::Constant::create(
            ov::element::i64, {5}, {(int64_t) 1, (int64_t) 1, factor, (int64_t) 1, (int64_t) 1});
        auto new_Z_shape = ov::op::v0::Constant::create(ov::element::i64, {4},
                                                        {(int64_t) 0, batch_large, (int64_t) -1, (int64_t) A_shape[3]});

        auto Z_broadcasted = std::make_shared<ov::op::v3::Broadcast>(Z_unsqueezed, broadcast_shape,
                                                                     ov::op::BroadcastType::BIDIRECTIONAL);
        Z = std::make_shared<ov::op::v1::Reshape>(Z_broadcasted, new_Z_shape, true);
    }
    if (A_batch_larger) {
        B = Z;
    } else {
        A = Z;
    }

    res = std::make_shared<ov::op::v0::MatMul>(A, B, false, transpose_b);

    return rename_outputs_with_suffix({res}, context.get_name());  // 返回
}

}  // namespace op
}  // namespace ggml
}  // namespace frontend
}  // namespace ov
