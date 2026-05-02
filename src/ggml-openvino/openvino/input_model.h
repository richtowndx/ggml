#pragma once  // 防止重复包含

#include <openvino/frontend/input_model.hpp>  // 引入 openvino/frontend/input_model.hpp 头文件

#include "decoder.h"  // 引入 decoder.h 头文件

namespace ov {  // 命名空间
namespace frontend {  // 命名空间
namespace ggml {  // 命名空间

class FrontEnd;  // 类定义
class GgmlDecoder;  // 类定义
using ov::frontend::ggml::GgmlDecoder;  // using 声明

class InputModel : public ov::frontend::InputModel {  // 类定义
    friend class ::ov::frontend::ggml::FrontEnd;

public:
    explicit InputModel(const std::shared_ptr<GgmlDecoder>& gdecoder);  // InputModel

    const std::shared_ptr<GgmlDecoder>& get_model_decoder() const;

private:
    std::shared_ptr<GgmlDecoder> m_decoder;
};

}  // namespace ggml
}  // namespace frontend
}  // namespace ov
