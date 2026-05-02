#include "../node_context.h"  // 引入 ../node_context.h 头文件
#include "../op_table.h"  // 引入 ../op_table.h 头文件
#include "../utils.h"  // 引入 ../utils.h 头文件

#include <openvino/op/add.hpp>  // 引入 openvino/op/add.hpp 头文件
#include <openvino/op/constant.hpp>  // 引入 openvino/op/constant.hpp 头文件
#include <openvino/op/multiply.hpp>  // 引入 openvino/op/multiply.hpp 头文件
#include <vector>  // 引入 vector 头文件

namespace ov {  // 命名空间
namespace frontend {  // 命名空间
namespace ggml {  // 命名空间
namespace op {  // 命名空间

OutputVector translate_scale(const NodeContext & context) {
    num_inputs_check(context, 1, 1);

    float scale;
    float bias;
    memcpy(&scale, (float *) context.get_output_op_params() + 0, sizeof(float));
    memcpy(&bias, (float *) context.get_output_op_params() + 1, sizeof(float));

    auto scale_node = std::make_shared<ov::op::v0::Constant>(ov::element::f32, ov::Shape{}, std::vector<float>{scale});
    auto scaled = std::make_shared<ov::op::v1::Multiply>(context.get_input(0), scale_node);

    std::shared_ptr<ov::Node> res;
    if (bias != 0.0f) {
        auto bias_node =
            std::make_shared<ov::op::v0::Constant>(ov::element::f32, ov::Shape{}, std::vector<float>{bias});
        res = std::make_shared<ov::op::v1::Add>(scaled, bias_node);
    } else {
        res = scaled;
    }

    return rename_outputs_with_suffix({res}, context.get_name());  // 返回
}

}  // namespace op
}  // namespace ggml
}  // namespace frontend
}  // namespace ov
