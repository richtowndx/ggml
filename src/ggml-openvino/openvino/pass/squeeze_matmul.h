#include "openvino/pass/matcher_pass.hpp"  // 引入 openvino/pass/matcher_pass.hpp 头文件

namespace ov {  // 命名空间
namespace frontend {  // 命名空间
namespace ggml {  // 命名空间
namespace pass {  // 命名空间

class SqueezeMatmul : public ov::pass::MatcherPass {  // 类定义
public:
    OPENVINO_MATCHER_PASS_RTTI("ov::frontend::ggml::pass::SqueezeMatmul")
    SqueezeMatmul();
};

}  // namespace pass
}  // namespace ggml
}  // namespace frontend
}  // namespace ov
