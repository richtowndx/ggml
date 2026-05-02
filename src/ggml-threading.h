#pragma once  // 防止重复包含

#include "ggml.h"  // 引入 ggml.h 头文件

#ifdef __cplusplus  // 如果定义了 __cplusplus 则编译
extern "C" {  // C 链接声明
#endif  // 条件编译结束

GGML_API void ggml_critical_section_start(void);
GGML_API void ggml_critical_section_end(void);

#ifdef __cplusplus  // 如果定义了 __cplusplus 则编译
}
#endif  // 条件编译结束
