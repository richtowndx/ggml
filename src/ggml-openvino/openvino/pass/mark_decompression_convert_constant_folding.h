#pragma once  // 防止重复包含

#include "mark_decompression_convert_constant_folding.h"  // 引入 mark_decompression_convert_constant_folding.h 头文件
#include "openvino/pass/matcher_pass.hpp"  // 引入 openvino/pass/matcher_pass.hpp 头文件
#include "openvino/core/visibility.hpp"  // 引入 openvino/core/visibility.hpp 头文件

#ifdef OPENVINO_STATIC_LIBRARY  // 如果定义了 OPENVINO_STATIC_LIBRARY 则编译
#    define TRANSFORMATIONS_API  // 宏定义 TRANSFORMATIONS_API
#else  // 否则
#    ifdef IMPLEMENT_OPENVINO_API  // 如果定义了 IMPLEMENT_OPENVINO_API 则编译
#        define TRANSFORMATIONS_API OPENVINO_CORE_EXPORTS  // 宏定义 TRANSFORMATIONS_API
#    else
#        define TRANSFORMATIONS_API OPENVINO_CORE_IMPORTS  // 宏定义 TRANSFORMATIONS_API
#    endif  // IMPLEMENT_OPENVINO_API
#endif      // OPENVINO_STATIC_LIBRARY  // 条件编译结束

namespace ov {  // 命名空间
namespace pass {  // 命名空间

class TRANSFORMATIONS_API MarkCompressedFloatConstants;  // 类定义

}  // namespace pass
}  // namespace ov

class ov::pass::MarkCompressedFloatConstants : public MatcherPass {  // 类定义
public:
    OPENVINO_MATCHER_PASS_RTTI("MarkCompressedFloatConstants")
    MarkCompressedFloatConstants();
};
