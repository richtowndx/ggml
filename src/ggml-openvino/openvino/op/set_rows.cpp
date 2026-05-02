#include "../node_context.h"  // 引入 ../node_context.h 头文件
#include "../op_table.h"  // 引入 ../op_table.h 头文件
#include "../utils.h"  // 引入 ../utils.h 头文件

#include <cassert>  // 引入 cassert 头文件
#include <cstdint>  // 引入 cstdint 头文件
#include <memory>  // 引入 memory 头文件
#include <openvino/core/node.hpp>  // 引入 openvino/core/node.hpp 头文件
#include <openvino/core/node_output.hpp>  // 引入 openvino/core/node_output.hpp 头文件
#include <openvino/frontend/exception.hpp>  // 引入 openvino/frontend/exception.hpp 头文件
#include <openvino/op/concat.hpp>  // 引入 openvino/op/concat.hpp 头文件
#include <openvino/op/constant.hpp>  // 引入 openvino/op/constant.hpp 头文件
#include <openvino/op/convert.hpp>  // 引入 openvino/op/convert.hpp 头文件
#include <openvino/op/gather.hpp>  // 引入 openvino/op/gather.hpp 头文件
#include <openvino/op/reshape.hpp>  // 引入 openvino/op/reshape.hpp 头文件
#include <openvino/op/scatter_update.hpp>  // 引入 openvino/op/scatter_update.hpp 头文件
#include <openvino/op/shape_of.hpp>  // 引入 openvino/op/shape_of.hpp 头文件
#include <openvino/op/slice.hpp>  // 引入 openvino/op/slice.hpp 头文件
#include <openvino/op/squeeze.hpp>  // 引入 openvino/op/squeeze.hpp 头文件
#include <openvino/op/transpose.hpp>  // 引入 openvino/op/transpose.hpp 头文件
#include <vector>  // 引入 vector 头文件

namespace ov {  // 命名空间
namespace frontend {  // 命名空间
namespace ggml {  // 命名空间
namespace op {  // 命名空间

OutputVector translate_set_rows(const NodeContext & context) {
    num_inputs_check(context, 3, 3);

    auto data = context.get_input(0);
    auto indices = context.get_input(1);
    auto dst = context.get_input(2);

    data = std::make_shared<ov::op::v0::Convert>(data, context.get_output_type());

    auto dst_shape = context.get_output_shape().to_shape();

    auto ind_squeezed =
        std::make_shared<ov::op::v0::Squeeze>(indices, ov::op::v0::Constant::create(ov::element::i64, {3}, {0, 1, 2}));
    auto data_reshaped = std::make_shared<ov::op::v1::Reshape>(
        data,
        ov::op::v0::Constant::create(ov::element::i64, {4},
                                     {(int64_t) 1, (int64_t) 1, (int64_t) -1, (int64_t) dst_shape[3]}),
        false);
    auto axes = ov::op::v0::Constant::create(ov::element::i64, ov::Shape{}, {2});

    Output<Node> res;
    if (context.is_stateful()) {
        int concat_axis = 1;
        int64_t dim2 = dst.get_partial_shape()[2].get_length();
        int64_t dim3 = dst.get_partial_shape()[3].get_length();
        data = std::make_shared<ov::op::v1::Reshape>(
            data, ov::op::v0::Constant::create(ov::element::i64, {4}, {(int64_t) 1, (int64_t) -1, dim2, dim3}), false);
        res = std::make_shared<ov::op::v0::Concat>(OutputVector{dst, data}, concat_axis);
    } else {
        res = std::make_shared<ov::op::v3::ScatterUpdate>(dst, ind_squeezed, data_reshaped, axes);
    }

    if (auto dst_reshape = std::dynamic_pointer_cast<ov::op::v1::Reshape>(dst.get_node_shared_ptr())) {
        // Fix the case of multiple sequences, reshape back to original shape [1, n_seq, ctx_per_seq, emb]
        // ctx_per_seq is not fixed due to llama-bench compatibility
        auto dst_shape_partial = dst_reshape->get_input_partial_shape(0);
        std::vector<int64_t> dst_shape = {dst_shape_partial[0].get_length(), dst_shape_partial[1].get_length(),
                                          dst_shape_partial[2].is_static() ? dst_shape_partial[2].get_length() : -1,
                                          dst_shape_partial[3].get_length()};
        res = std::make_shared<ov::op::v1::Reshape>(res, ov::op::v0::Constant::create(ov::element::i64, {4}, dst_shape),
                                                    false);
    }
    return rename_outputs_with_suffix({res}, context.get_name());  // 返回
}

}  // namespace op
}  // namespace ggml
}  // namespace frontend
}  // namespace ov
