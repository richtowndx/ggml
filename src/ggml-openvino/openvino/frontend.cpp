#include "frontend.h"  // 引入 frontend.h 头文件

#include "input_model.h"  // 引入 input_model.h 头文件
#include "op_table.h"  // 引入 op_table.h 头文件
#include "translate_session.h"  // 引入 translate_session.h 头文件

namespace ov {  // 命名空间
namespace frontend {  // 命名空间
namespace ggml {  // 命名空间

FrontEnd::FrontEnd() {}

std::shared_ptr<Model> FrontEnd::convert(const InputModel::Ptr & model, bool naive) {
    auto ggml_model = std::dynamic_pointer_cast<ggml::InputModel>(model);
    FRONT_END_GENERAL_CHECK(ggml_model, "Invalid input model");
    std::shared_ptr<Model> converted_model;
    const auto & supported_ops = get_supported_ops();
    {
        TranslateSession translate_session(model, supported_ops, naive);  // translate_session
        converted_model = translate_session.get_converted_model();
    }
    return converted_model;  // 返回
}

}  // namespace ggml
}  // namespace frontend
}  // namespace ov
