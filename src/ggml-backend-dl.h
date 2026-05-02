#pragma once  // 防止重复包含

#ifdef _WIN32  // 如果定义了 _WIN32 则编译
#   define WIN32_LEAN_AND_MEAN  // 宏定义 WIN32_LEAN_AND_MEAN
#   ifndef NOMINMAX  // 如果未定义 NOMINMAX 则编译
#       define NOMINMAX  // 宏定义 NOMINMAX
#   endif
#   include <windows.h>  // 引入 windows.h 头文件
#   include <winevt.h>  // 引入 winevt.h 头文件
#else  // 否则
#    include <dlfcn.h>  // 引入 dlfcn.h 头文件
#    include <unistd.h>  // 引入 unistd.h 头文件
#endif  // 条件编译结束
#include <filesystem>  // 引入 filesystem 头文件

namespace fs = std::filesystem;  // 命名空间

#ifdef _WIN32  // 如果定义了 _WIN32 则编译

using dl_handle = std::remove_pointer_t<HMODULE>;  // using 声明

struct dl_handle_deleter {  // 结构体定义
    void operator()(HMODULE handle) {
        FreeLibrary(handle);
    }
};

#else  // 否则

using dl_handle = void;  // using 声明

struct dl_handle_deleter {  // 结构体定义
    void operator()(void * handle) {
        dlclose(handle);
    }
};

#endif  // 条件编译结束

using dl_handle_ptr = std::unique_ptr<dl_handle, dl_handle_deleter>;  // using 声明

dl_handle * dl_load_library(const fs::path & path);  // dl_load_library
void * dl_get_sym(dl_handle * handle, const char * name);  // dl_get_sym
const char * dl_error();  // dl_error

