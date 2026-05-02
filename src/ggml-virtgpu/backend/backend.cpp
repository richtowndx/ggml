#include "backend-dispatched.h"  // 引入 backend-dispatched.h 头文件
#include "backend-virgl-apir.h"  // 引入 backend-virgl-apir.h 头文件
#include "shared/api_remoting.h"  // 引入 shared/api_remoting.h 头文件
#include "shared/apir_backend.h"  // 引入 shared/apir_backend.h 头文件
#include "shared/apir_cs.h"  // 引入 shared/apir_cs.h 头文件

#include <dlfcn.h>  // 引入 dlfcn.h 头文件
#include <ggml-backend.h>  // 引入 ggml-backend.h 头文件

#include <iostream>  // 引入 iostream 头文件

#define APIR_LLAMA_CPP_GGML_LIBRARY_PATH_ENV "APIR_LLAMA_CPP_GGML_LIBRARY_PATH"  // 宏定义 APIR_LLAMA_CPP_GGML_LIBRARY_PATH_ENV
#define APIR_LLAMA_CPP_GGML_LIBRARY_REG_ENV  "APIR_LLAMA_CPP_GGML_LIBRARY_REG"  // 宏定义 APIR_LLAMA_CPP_GGML_LIBRARY_REG_ENV
#define APIR_LLAMA_CPP_LOG_TO_FILE_ENV       "APIR_LLAMA_CPP_LOG_TO_FILE"  // 宏定义 APIR_LLAMA_CPP_LOG_TO_FILE_ENV

#define GGML_DEFAULT_BACKEND_REG "ggml_backend_init"  // 宏定义 GGML_DEFAULT_BACKEND_REG

static void * backend_library_handle = NULL;
static FILE * apir_logfile           = NULL;

static void log_to_file_callback(enum ggml_log_level level, const char * text, void * user_data) {
    FILE * logfile = (FILE *) user_data;
    fprintf(logfile, "[%d] %s", level, text);
    fflush(logfile);
}

extern "C" {  // C 链接声明
void apir_backend_deinit(uint32_t virgl_ctx_id) {
    GGML_UNUSED(virgl_ctx_id);

    auto buffers = apir_get_track_backend_buffers();
    for (const auto & buffer : buffers) {
        apir_untrack_backend_buffer(buffer);
        buffer->iface.free_buffer(buffer);
    }

    if (backend_library_handle) {
        GGML_LOG_INFO(GGML_VIRTGPU_BCK "The GGML backend library was loaded. Unloading it.\n");
        dlclose(backend_library_handle);
        backend_library_handle = NULL;
    }

    if (apir_logfile) {
        fclose(apir_logfile);
        apir_logfile = NULL;
    }
}

#define APIR_GGML_LIBRARY_PATH_KEY "ggml.library.path"  // 宏定义 APIR_GGML_LIBRARY_PATH_KEY
#define APIR_GGML_LIBRARY_REG_KEY  "ggml.library.reg"  // 宏定义 APIR_GGML_LIBRARY_REG_KEY

ApirLoadLibraryReturnCode apir_backend_initialize(uint32_t virgl_ctx_id, struct virgl_apir_callbacks * virgl_cbs) {
    const char * dlsym_error;

    const char * apir_log_to_file = getenv(APIR_LLAMA_CPP_LOG_TO_FILE_ENV);
    if (apir_log_to_file) {
        apir_logfile = fopen(apir_log_to_file, "w");
        if (apir_logfile) {
            ggml_log_set(log_to_file_callback, apir_logfile);
        } else {
            GGML_LOG_INFO(GGML_VIRTGPU_BCK "Could not open the log file at '%s'\n", apir_log_to_file);
        }
    }

    const char * library_name      = virgl_cbs->get_config(virgl_ctx_id, APIR_GGML_LIBRARY_PATH_KEY);
    const char * virgl_library_reg = virgl_cbs->get_config(virgl_ctx_id, APIR_GGML_LIBRARY_REG_KEY);
    const char * library_reg       = virgl_library_reg ? virgl_library_reg : GGML_DEFAULT_BACKEND_REG;

    if (!library_name) {
        GGML_LOG_ERROR(GGML_VIRTGPU_BCK "%s: cannot open the GGML library: env var '%s' not defined\n", __func__,  // 打印错误日志
                       APIR_LLAMA_CPP_GGML_LIBRARY_PATH_ENV);

        return APIR_LOAD_LIBRARY_ENV_VAR_MISSING;  // 返回
    }

    backend_library_handle = dlopen(library_name, RTLD_LAZY);

    if (!backend_library_handle) {
        GGML_LOG_ERROR(GGML_VIRTGPU_BCK "%s: cannot open the GGML library: %s\n", __func__, dlerror());

        return APIR_LOAD_LIBRARY_CANNOT_OPEN;  // 返回
    }

    if (!library_reg) {
        GGML_LOG_ERROR(GGML_VIRTGPU_BCK "%s: cannot register the GGML library: env var '%s' not defined\n", __func__,  // 打印错误日志
                       APIR_LLAMA_CPP_GGML_LIBRARY_REG_ENV);

        return APIR_LOAD_LIBRARY_ENV_VAR_MISSING;  // 返回
    }

    void * ggml_backend_reg_fct = dlsym(backend_library_handle, library_reg);
    dlsym_error                 = dlerror();
    if (dlsym_error) {
        GGML_LOG_ERROR(GGML_VIRTGPU_BCK "%s: cannot find the GGML backend registration symbol '%s' (from %s): %s\n",
                       __func__, library_reg, APIR_LLAMA_CPP_GGML_LIBRARY_REG_ENV, dlsym_error);

        return APIR_LOAD_LIBRARY_SYMBOL_MISSING;  // 返回
    }

    uint32_t ret = backend_dispatch_initialize(ggml_backend_reg_fct);

    return (ApirLoadLibraryReturnCode) (APIR_LOAD_LIBRARY_INIT_BASE_INDEX + ret);
}

uint32_t apir_backend_dispatcher(uint32_t               virgl_ctx_id,
                                 virgl_apir_callbacks * virgl_cbs,
                                 uint32_t               cmd_type,
                                 char *                 dec_cur,
                                 const char *           dec_end,
                                 char *                 enc_cur,
                                 const char *           enc_end,
                                 char **                enc_cur_after) {
    apir_encoder enc = {
        .cur   = enc_cur,
        .start = enc_cur,
        .end   = enc_end,
        .fatal = false,
    };

    apir_decoder dec = {
        .cur   = dec_cur,
        .end   = dec_end,
        .fatal = false,
    };

    virgl_apir_context ctx = {
        .ctx_id = virgl_ctx_id,
        .iface  = virgl_cbs,
    };

    if (cmd_type >= APIR_BACKEND_DISPATCH_TABLE_COUNT) {
        GGML_LOG_ERROR(GGML_VIRTGPU_BCK "%s: Received an invalid dispatch index (%d >= %d)\n", __func__, cmd_type,
                       APIR_BACKEND_DISPATCH_TABLE_COUNT);
        return APIR_BACKEND_FORWARD_INDEX_INVALID;  // 返回
    }

    backend_dispatch_t forward_fct = apir_backend_dispatch_table[cmd_type];
    uint32_t           ret         = forward_fct(&enc, &dec, &ctx);

    *enc_cur_after = enc.cur;

    return ret;  // 返回
}
}
