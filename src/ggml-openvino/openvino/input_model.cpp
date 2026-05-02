#include "input_model.h"  // 引入 input_model.h 头文件

#include "decoder.h"  // 引入 decoder.h 头文件

namespace ov {  // 命名空间
namespace frontend {  // 命名空间
namespace ggml {  // 命名空间

InputModel::InputModel(const std::shared_ptr<GgmlDecoder> & gdecoder) : m_decoder(gdecoder) {}

const std::shared_ptr<GgmlDecoder> & InputModel::get_model_decoder() const {
    return m_decoder;  // 返回
}

}  // namespace ggml
}  // namespace frontend
}  // namespace ov
