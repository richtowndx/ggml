#include "../node_context.h"  // 引入 ../node_context.h 头文件
#include "../op_table.h"  // 引入 ../op_table.h 头文件
#include "../utils.h"  // 引入 ../utils.h 头文件

#include <memory>  // 引入 memory 头文件
#include <openvino/op/add.hpp>  // 引入 openvino/op/add.hpp 头文件
#include <openvino/op/constant.hpp>  // 引入 openvino/op/constant.hpp 头文件
#include <openvino/op/divide.hpp>  // 引入 openvino/op/divide.hpp 头文件
#include <openvino/op/multiply.hpp>  // 引入 openvino/op/multiply.hpp 头文件
#include <openvino/op/power.hpp>  // 引入 openvino/op/power.hpp 头文件
#include <openvino/op/reduce_mean.hpp>  // 引入 openvino/op/reduce_mean.hpp 头文件
#include <openvino/op/sqrt.hpp>  // 引入 openvino/op/sqrt.hpp 头文件

namespace ov {  // 命名空间
namespace frontend {  // 命名空间
namespace ggml {  // 命名空间
namespace op {  // 命名空间

OutputVector translate_rms_norm(const NodeContext & context) {
    num_inputs_check(context, 1, 1);

    auto input_node = context.get_input(0);
    auto square = std::make_shared<ov::op::v1::Power>(
        input_node, ov::op::v0::Constant::create(ov::element::f32, ov::Shape{1}, {2.0f}));

    auto mean = std::make_shared<ov::op::v1::ReduceMean>(
        square, ov::op::v0::Constant::create(ov::element::i64, ov::Shape{1}, {-1}), true);

    float eps;
    memcpy(&eps, context.get_output_op_params(), sizeof(float));

    auto rms = std::make_shared<ov::op::v0::Sqrt>(
        std::make_shared<ov::op::v1::Add>(mean, ov::op::v0::Constant::create(ov::element::f32, ov::Shape{1}, {eps})));

    auto reciprocal =
        std::make_shared<ov::op::v1::Divide>(ov::op::v0::Constant::create(ov::element::f32, ov::Shape{1}, {1.0f}), rms);

    auto res = std::make_shared<ov::op::v1::Multiply>(input_node, reciprocal);

    return rename_outputs_with_suffix({res}, context.get_name());  // 返回
}

}  // namespace op
}  // namespace ggml
}  // namespace frontend
}  // namespace ov
