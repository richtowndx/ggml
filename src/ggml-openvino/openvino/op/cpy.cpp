#include "../node_context.h"  // 引入 ../node_context.h 头文件
#include "../op_table.h"  // 引入 ../op_table.h 头文件
#include "../utils.h"  // 引入 ../utils.h 头文件

#include <memory>  // 引入 memory 头文件
#include <openvino/op/convert.hpp>  // 引入 openvino/op/convert.hpp 头文件

namespace ov {  // 命名空间
namespace frontend {  // 命名空间
namespace ggml {  // 命名空间
namespace op {  // 命名空间

OutputVector translate_cpy(const NodeContext & context) {
    auto res = std::make_shared<ov::op::v0::Convert>(context.get_input(0), context.get_output_type());
    return rename_outputs_with_suffix({res}, context.get_name());  // 返回
}

}  // namespace op
}  // namespace ggml
}  // namespace frontend
}  // namespace ov
