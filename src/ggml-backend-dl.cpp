#include "ggml-backend-dl.h"  // 引入 ggml-backend-dl.h 头文件

#ifdef _WIN32  // 如果定义了 _WIN32 则编译

dl_handle * dl_load_library(const fs::path & path) {
    // suppress error dialogs for missing DLLs
    DWORD old_mode = SetErrorMode(SEM_FAILCRITICALERRORS);
    SetErrorMode(old_mode | SEM_FAILCRITICALERRORS);

    HMODULE handle = LoadLibraryW(path.wstring().c_str());

    SetErrorMode(old_mode);

    return handle;  // 返回
}

void * dl_get_sym(dl_handle * handle, const char * name) {
    DWORD old_mode = SetErrorMode(SEM_FAILCRITICALERRORS);
    SetErrorMode(old_mode | SEM_FAILCRITICALERRORS);

    void * p = (void *) GetProcAddress(handle, name);

    SetErrorMode(old_mode);

    return p;  // 返回
}

const char * dl_error() {
    return "";  // 返回
}

#else  // 否则

dl_handle * dl_load_library(const fs::path & path) {
    dl_handle * handle = dlopen(path.string().c_str(), RTLD_NOW | RTLD_LOCAL);
    return handle;  // 返回
}

void * dl_get_sym(dl_handle * handle, const char * name) {
    return dlsym(handle, name);  // dlsym
}

const char * dl_error() {
    const char *rslt = dlerror();
    return rslt != nullptr ? rslt : "";  // 返回
}

#endif  // 条件编译结束
