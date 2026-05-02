#include "ggml-backend.h"  // 引入 ggml-backend.h 头文件
#include "ggml-cpu-impl.h"  // 引入 ggml-cpu-impl.h 头文件

// GGML internal header

#if defined(__AMX_INT8__) && defined(__AVX512VNNI__)  // 条件编译
ggml_backend_buffer_type_t ggml_backend_amx_buffer_type(void);  // ggml_backend_amx_buffer_type
#endif  // 条件编译结束
