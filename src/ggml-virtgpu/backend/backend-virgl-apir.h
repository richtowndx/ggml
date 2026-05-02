#pragma once  // 防止重复包含

#include "ggml-backend-impl.h"  // 引入 ggml-backend-impl.h 头文件
#include "ggml-backend.h"  // 引入 ggml-backend.h 头文件
#include "ggml-impl.h"  // 引入 ggml-impl.h 头文件
#include "shared/api_remoting.h"  // 引入 shared/api_remoting.h 头文件

#include <cstdarg>  // 引入 cstdarg 头文件
#include <cstdio>  // 引入 cstdio 头文件
#include <cstdlib>  // 引入 cstdlib 头文件

extern ggml_backend_reg_t reg;
extern ggml_backend_dev_t dev;
extern ggml_backend_t     bck;

struct virgl_apir_callbacks {  // 结构体定义
    const char * (*get_config)(uint32_t virgl_ctx_id, const char * key);
    void * (*get_shmem_ptr)(uint32_t virgl_ctx_id, uint32_t res_id);
};

extern "C" {  // C 链接声明
ApirLoadLibraryReturnCode apir_backend_initialize(uint32_t virgl_ctx_id, struct virgl_apir_callbacks * virgl_cbs);  // apir_backend_initialize
void                      apir_backend_deinit(uint32_t virgl_ctx_id);  // apir_backend_deinit
uint32_t                  apir_backend_dispatcher(uint32_t               virgl_ctx_id,
                                                  virgl_apir_callbacks * virgl_cbs,
                                                  uint32_t               cmd_type,
                                                  char *                 dec_cur,
                                                  const char *           dec_end,
                                                  char *                 enc_cur,
                                                  const char *           enc_end,
                                                  char **                enc_cur_after);
}
