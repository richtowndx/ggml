#include "ggml-backend-impl.h"  // 引入 ggml-backend-impl.h 头文件
#include "ggml-backend.h"  // 引入 ggml-backend.h 头文件
#include "ggml-backend-dl.h"  // 引入 ggml-backend-dl.h 头文件
#include "ggml-impl.h"  // 引入 ggml-impl.h 头文件
#include <algorithm>  // 引入 algorithm 头文件
#include <cstring>  // 引入 cstring 头文件
#include <filesystem>  // 引入 filesystem 头文件
#include <memory>  // 引入 memory 头文件
#include <string>  // 引入 string 头文件
#include <type_traits>  // 引入 type_traits 头文件
#include <vector>  // 引入 vector 头文件
#include <cctype>  // 引入 cctype 头文件

#ifdef _WIN32  // 如果定义了 _WIN32 则编译
#    define WIN32_LEAN_AND_MEAN  // 宏定义 WIN32_LEAN_AND_MEAN
#    ifndef NOMINMAX  // 如果未定义 NOMINMAX 则编译
#        define NOMINMAX  // 宏定义 NOMINMAX
#    endif
#    include <windows.h>  // 引入 windows.h 头文件
#elif defined(__APPLE__)  // 否则如果
#    include <mach-o/dyld.h>  // 引入 mach-o/dyld.h 头文件
#    include <dlfcn.h>  // 引入 dlfcn.h 头文件
#else  // 否则
#    include <dlfcn.h>  // 引入 dlfcn.h 头文件
#    include <unistd.h>  // 引入 unistd.h 头文件
#endif  // 条件编译结束

// Backend registry
#ifdef GGML_USE_CPU  // 如果定义了 GGML_USE_CPU 则编译
#include "ggml-cpu.h"  // 引入 ggml-cpu.h 头文件
#endif  // 条件编译结束

#ifdef GGML_USE_CUDA  // 如果定义了 GGML_USE_CUDA 则编译
#include "ggml-cuda.h"  // 引入 ggml-cuda.h 头文件
#endif  // 条件编译结束

#ifdef GGML_USE_METAL  // 如果定义了 GGML_USE_METAL 则编译
#include "ggml-metal.h"  // 引入 ggml-metal.h 头文件
#endif  // 条件编译结束

#ifdef GGML_USE_SYCL  // 如果定义了 GGML_USE_SYCL 则编译
#include "ggml-sycl.h"  // 引入 ggml-sycl.h 头文件
#endif  // 条件编译结束

#ifdef GGML_USE_VULKAN  // 如果定义了 GGML_USE_VULKAN 则编译
#include "ggml-vulkan.h"  // 引入 ggml-vulkan.h 头文件
#endif  // 条件编译结束

#ifdef GGML_USE_WEBGPU  // 如果定义了 GGML_USE_WEBGPU 则编译
#include "ggml-webgpu.h"  // 引入 ggml-webgpu.h 头文件
#endif  // 条件编译结束

#ifdef GGML_USE_ZDNN  // 如果定义了 GGML_USE_ZDNN 则编译
#include "ggml-zdnn.h"  // 引入 ggml-zdnn.h 头文件
#endif  // 条件编译结束

#ifdef GGML_USE_OPENCL  // 如果定义了 GGML_USE_OPENCL 则编译
#include "ggml-opencl.h"  // 引入 ggml-opencl.h 头文件
#endif  // 条件编译结束

#ifdef GGML_USE_HEXAGON  // 如果定义了 GGML_USE_HEXAGON 则编译
#include "ggml-hexagon.h"  // 引入 ggml-hexagon.h 头文件
#endif  // 条件编译结束

#ifdef GGML_USE_BLAS  // 如果定义了 GGML_USE_BLAS 则编译
#include "ggml-blas.h"  // 引入 ggml-blas.h 头文件
#endif  // 条件编译结束

#ifdef GGML_USE_RPC  // 如果定义了 GGML_USE_RPC 则编译
#include "ggml-rpc.h"  // 引入 ggml-rpc.h 头文件
#endif  // 条件编译结束

#ifdef GGML_USE_VIRTGPU_FRONTEND  // 如果定义了 GGML_USE_VIRTGPU_FRONTEND 则编译
#include "ggml-virtgpu.h"  // 引入 ggml-virtgpu.h 头文件
#endif  // 条件编译结束

#ifdef GGML_USE_CANN  // 如果定义了 GGML_USE_CANN 则编译
#include "ggml-cann.h"  // 引入 ggml-cann.h 头文件
#endif  // 条件编译结束

#ifdef GGML_USE_ZENDNN  // 如果定义了 GGML_USE_ZENDNN 则编译
#include "ggml-zendnn.h"  // 引入 ggml-zendnn.h 头文件
#endif  // 条件编译结束

#ifdef GGML_USE_OPENVINO  // 如果定义了 GGML_USE_OPENVINO 则编译
#include "ggml-openvino.h"  // 引入 ggml-openvino.h 头文件
#endif  // 条件编译结束

namespace fs = std::filesystem;  // 命名空间

static std::string path_str(const fs::path & path) {
    try {
#if defined(__cpp_lib_char8_t)  // 条件编译
        // C++20 and later: u8string() returns std::u8string
        const std::u8string u8str = path.u8string();
        return std::string(reinterpret_cast<const char *>(u8str.data()), u8str.size());
#else  // 否则
        // C++17: u8string() returns std::string
        return path.u8string();
#endif  // 条件编译结束
    } catch (...) {
        return std::string();
    }
}

struct ggml_backend_reg_entry {  // 结构体定义
    ggml_backend_reg_t reg;
    dl_handle_ptr handle;
};

struct ggml_backend_registry {  // 结构体定义
    std::vector<ggml_backend_reg_entry> backends;
    std::vector<ggml_backend_dev_t> devices;

    ggml_backend_registry() {
#ifdef GGML_USE_CUDA  // 如果定义了 GGML_USE_CUDA 则编译
        register_backend(ggml_backend_cuda_reg());
#endif  // 条件编译结束
#ifdef GGML_USE_METAL  // 如果定义了 GGML_USE_METAL 则编译
        register_backend(ggml_backend_metal_reg());
#endif  // 条件编译结束
#ifdef GGML_USE_SYCL  // 如果定义了 GGML_USE_SYCL 则编译
        register_backend(ggml_backend_sycl_reg());
#endif  // 条件编译结束
#ifdef GGML_USE_VULKAN  // 如果定义了 GGML_USE_VULKAN 则编译
    // Add runtime disable check
    if (getenv("GGML_DISABLE_VULKAN") == nullptr) {
        register_backend(ggml_backend_vk_reg());
    } else {
        GGML_LOG_DEBUG("Vulkan backend disabled by GGML_DISABLE_VULKAN environment variable\n");
    }
#endif  // 条件编译结束
#ifdef GGML_USE_WEBGPU  // 如果定义了 GGML_USE_WEBGPU 则编译
        register_backend(ggml_backend_webgpu_reg());
#endif  // 条件编译结束
#ifdef GGML_USE_ZDNN  // 如果定义了 GGML_USE_ZDNN 则编译
        register_backend(ggml_backend_zdnn_reg());
#endif  // 条件编译结束
#ifdef GGML_USE_VIRTGPU_FRONTEND  // 如果定义了 GGML_USE_VIRTGPU_FRONTEND 则编译
        register_backend(ggml_backend_virtgpu_reg());
#endif  // 条件编译结束

#ifdef GGML_USE_OPENCL  // 如果定义了 GGML_USE_OPENCL 则编译
        register_backend(ggml_backend_opencl_reg());
#endif  // 条件编译结束
#ifdef GGML_USE_ZENDNN  // 如果定义了 GGML_USE_ZENDNN 则编译
        register_backend(ggml_backend_zendnn_reg());
#endif  // 条件编译结束
#ifdef GGML_USE_HEXAGON  // 如果定义了 GGML_USE_HEXAGON 则编译
        register_backend(ggml_backend_hexagon_reg());
#endif  // 条件编译结束
#ifdef GGML_USE_CANN  // 如果定义了 GGML_USE_CANN 则编译
        register_backend(ggml_backend_cann_reg());
#endif  // 条件编译结束
#ifdef GGML_USE_BLAS  // 如果定义了 GGML_USE_BLAS 则编译
        register_backend(ggml_backend_blas_reg());
#endif  // 条件编译结束
#ifdef GGML_USE_RPC  // 如果定义了 GGML_USE_RPC 则编译
        register_backend(ggml_backend_rpc_reg());
#endif  // 条件编译结束
#ifdef GGML_USE_OPENVINO  // 如果定义了 GGML_USE_OPENVINO 则编译
        register_backend(ggml_backend_openvino_reg());
#endif  // 条件编译结束
#ifdef GGML_USE_CPU  // 如果定义了 GGML_USE_CPU 则编译
        register_backend(ggml_backend_cpu_reg());
#endif  // 条件编译结束
    }

    ~ggml_backend_registry() {
        // FIXME: backends cannot be safely unloaded without a function to destroy all the backend resources,
        // since backend threads may still be running and accessing resources from the dynamic library
        for (auto & entry : backends) {
            if (entry.handle) {
                entry.handle.release(); // NOLINT
            }
        }
    }

    void register_backend(ggml_backend_reg_t reg, dl_handle_ptr handle = nullptr) {
        if (!reg) {
            return;  // 返回
        }

        for (auto & entry : backends) {
            if (entry.reg == reg) {
                return;  // 返回
            }
        }

#ifndef NDEBUG  // 如果未定义 NDEBUG 则编译
        GGML_LOG_DEBUG("%s: registered backend %s (%zu devices)\n",
            __func__, ggml_backend_reg_name(reg), ggml_backend_reg_dev_count(reg));
#endif  // 条件编译结束
        backends.push_back({ reg, std::move(handle) });
        for (size_t i = 0; i < ggml_backend_reg_dev_count(reg); i++) {
            register_device(ggml_backend_reg_dev_get(reg, i));
        }
    }

    void register_device(ggml_backend_dev_t device) {
        for (auto & dev : devices) {
            if (dev == device) {
                return;  // 返回
            }
        }

#ifndef NDEBUG  // 如果未定义 NDEBUG 则编译
        GGML_LOG_DEBUG("%s: registered device %s (%s)\n", __func__, ggml_backend_dev_name(device), ggml_backend_dev_description(device));
#endif  // 条件编译结束
        devices.push_back(device);
    }

    ggml_backend_reg_t load_backend(const fs::path & path, bool silent) {
        dl_handle_ptr handle { dl_load_library(path) };
        if (!handle) {
            if (!silent) {
                GGML_LOG_ERROR("%s: failed to load %s: %s\n", __func__, path_str(path).c_str(), dl_error());
            }
            return nullptr;  // 返回
        }

        auto score_fn = (ggml_backend_score_t) dl_get_sym(handle.get(), "ggml_backend_score");
        if (score_fn && score_fn() == 0) {
            if (!silent) {
                GGML_LOG_INFO("%s: backend %s is not supported on this system\n", __func__, path_str(path).c_str());
            }
            return nullptr;  // 返回
        }

        auto backend_init_fn = (ggml_backend_init_t) dl_get_sym(handle.get(), "ggml_backend_init");
        if (!backend_init_fn) {
            if (!silent) {
                GGML_LOG_ERROR("%s: failed to find ggml_backend_init in %s\n", __func__, path_str(path).c_str());
            }
            return nullptr;  // 返回
        }

        ggml_backend_reg_t reg = backend_init_fn();
        if (!reg || reg->api_version != GGML_BACKEND_API_VERSION) {
            if (!silent) {
                if (!reg) {
                    GGML_LOG_ERROR("%s: failed to initialize backend from %s: ggml_backend_init returned NULL\n",  // 打印错误日志
                        __func__, path_str(path).c_str());
                } else {
                    GGML_LOG_ERROR("%s: failed to initialize backend from %s: incompatible API version (backend: %d, current: %d)\n",
                        __func__, path_str(path).c_str(), reg->api_version, GGML_BACKEND_API_VERSION);
                }
            }
            return nullptr;  // 返回
        }

        GGML_LOG_INFO("%s: loaded %s backend from %s\n", __func__, ggml_backend_reg_name(reg), path_str(path).c_str());

        register_backend(reg, std::move(handle));

        return reg;  // 返回
    }

    void unload_backend(ggml_backend_reg_t reg, bool silent) {
        auto it = std::find_if(backends.begin(), backends.end(),
                               [reg](const ggml_backend_reg_entry & entry) { return entry.reg == reg; });

        if (it == backends.end()) {
            if (!silent) {
                GGML_LOG_ERROR("%s: backend not found\n", __func__);
            }
            return;  // 返回
        }

        if (!silent) {
            GGML_LOG_DEBUG("%s: unloading %s backend\n", __func__, ggml_backend_reg_name(reg));
        }

        // remove devices
        devices.erase(
            std::remove_if(devices.begin(), devices.end(),
                            [reg](ggml_backend_dev_t dev) { return ggml_backend_dev_backend_reg(dev) == reg; }),
            devices.end());

        // remove backend
        backends.erase(it);
    }
};

static ggml_backend_registry & get_reg() {
    static ggml_backend_registry reg;
    return reg;  // 返回
}

// Internal API
void ggml_backend_register(ggml_backend_reg_t reg) {
    get_reg().register_backend(reg);
}

void ggml_backend_device_register(ggml_backend_dev_t device) {
    get_reg().register_device(device);
}

// Backend (reg) enumeration
static bool striequals(const char * a, const char * b) {
    for (; *a && *b; a++, b++) {
        if (std::tolower(*a) != std::tolower(*b)) {
            return false;  // 返回
        }
    }
    return *a == *b;  // 返回
}

size_t ggml_backend_reg_count() {
    return get_reg().backends.size();  // get_reg
}

ggml_backend_reg_t ggml_backend_reg_get(size_t index) {
    GGML_ASSERT(index < ggml_backend_reg_count());
    return get_reg().backends[index].reg;  // get_reg
}

ggml_backend_reg_t ggml_backend_reg_by_name(const char * name) {
    for (size_t i = 0; i < ggml_backend_reg_count(); i++) {
        ggml_backend_reg_t reg = ggml_backend_reg_get(i);
        if (striequals(ggml_backend_reg_name(reg), name)) {
            return reg;  // 返回
        }
    }
    return nullptr;  // 返回
}

// Device enumeration
size_t ggml_backend_dev_count() {
    return get_reg().devices.size();  // get_reg
}

ggml_backend_dev_t ggml_backend_dev_get(size_t index) {
    GGML_ASSERT(index < ggml_backend_dev_count());
    return get_reg().devices[index];  // get_reg
}

ggml_backend_dev_t ggml_backend_dev_by_name(const char * name) {
    for (size_t i = 0; i < ggml_backend_dev_count(); i++) {
        ggml_backend_dev_t dev = ggml_backend_dev_get(i);
        if (striequals(ggml_backend_dev_name(dev), name)) {
            return dev;  // 返回
        }
    }
    return nullptr;  // 返回
}

ggml_backend_dev_t ggml_backend_dev_by_type(enum ggml_backend_dev_type type) {
    for (size_t i = 0; i < ggml_backend_dev_count(); i++) {
        ggml_backend_dev_t dev = ggml_backend_dev_get(i);
        if (ggml_backend_dev_type(dev) == type) {
            return dev;  // 返回
        }
    }
    return nullptr;  // 返回
}

// Convenience functions
ggml_backend_t ggml_backend_init_by_name(const char * name, const char * params) {
    ggml_backend_dev_t dev = ggml_backend_dev_by_name(name);
    if (!dev) {
        return nullptr;  // 返回
    }
    return ggml_backend_dev_init(dev, params);  // ggml_backend_dev_init
}

ggml_backend_t ggml_backend_init_by_type(enum ggml_backend_dev_type type, const char * params) {
    ggml_backend_dev_t dev = ggml_backend_dev_by_type(type);
    if (!dev) {
        return nullptr;  // 返回
    }
    return ggml_backend_dev_init(dev, params);  // ggml_backend_dev_init
}

ggml_backend_t ggml_backend_init_best(void) {
    ggml_backend_dev_t dev = ggml_backend_dev_by_type(GGML_BACKEND_DEVICE_TYPE_GPU);
    dev = dev ? dev : ggml_backend_dev_by_type(GGML_BACKEND_DEVICE_TYPE_IGPU);
    dev = dev ? dev : ggml_backend_dev_by_type(GGML_BACKEND_DEVICE_TYPE_CPU);
    if (!dev) {
        return nullptr;  // 返回
    }
    return ggml_backend_dev_init(dev, nullptr);  // ggml_backend_dev_init
}

// Dynamic loading
ggml_backend_reg_t ggml_backend_load(const char * path) {
    return get_reg().load_backend(path, false);  // get_reg
}

void ggml_backend_unload(ggml_backend_reg_t reg) {
    get_reg().unload_backend(reg, true);
}

static fs::path get_executable_path() {
#if defined(__APPLE__)  // 条件编译
    // get executable path
    std::vector<char> path;
    uint32_t size;
    while (true) {
        size = path.size();
        if (_NSGetExecutablePath(path.data(), &size) == 0) {
            break;
        }
        path.resize(size);
    }
    std::string base_path(path.data(), size);
    // remove executable name
    auto last_slash = base_path.find_last_of('/');
    if (last_slash != std::string::npos) {
        base_path = base_path.substr(0, last_slash);
    }
    return base_path + "/";  // 返回
#elif defined(__linux__) || defined(__FreeBSD__)  // 否则如果
    std::string base_path = ".";
    std::vector<char> path(1024);
    while (true) {
        // get executable path
#    if defined(__linux__)
        ssize_t len = readlink("/proc/self/exe", path.data(), path.size());
#    elif defined(__FreeBSD__)
        ssize_t len = readlink("/proc/curproc/file", path.data(), path.size());
#    endif
        if (len == -1) {
            break;
        }
        if (len < (ssize_t) path.size()) {
            base_path = std::string(path.data(), len);
            // remove executable name
            auto last_slash = base_path.find_last_of('/');
            if (last_slash != std::string::npos) {
                base_path = base_path.substr(0, last_slash);
            }
            break;
        }
        path.resize(path.size() * 2);
    }

    return base_path + "/";  // 返回
#elif defined(_WIN32)  // 否则如果
    std::vector<wchar_t> path(MAX_PATH);
    DWORD len = GetModuleFileNameW(NULL, path.data(), path.size());
    if (len == 0) {
        return {};  // 返回
    }
    std::wstring base_path(path.data(), len);
    // remove executable name
    auto last_slash = base_path.find_last_of('\\');
    if (last_slash != std::string::npos) {
        base_path = base_path.substr(0, last_slash);
    }
    return base_path + L"\\";  // 返回
#else  // 否则
    return {};  // 返回
#endif  // 条件编译结束
}

static fs::path backend_filename_prefix() {
#ifdef _WIN32  // 如果定义了 _WIN32 则编译
    return fs::u8path("ggml-");
#else  // 否则
    return fs::u8path("libggml-");
#endif  // 条件编译结束
}

static fs::path backend_filename_extension() {
#ifdef _WIN32  // 如果定义了 _WIN32 则编译
    return fs::u8path(".dll");
#else  // 否则
    return fs::u8path(".so");
#endif  // 条件编译结束
}

static ggml_backend_reg_t ggml_backend_load_best(const char * name, bool silent, const char * user_search_path) {
    // enumerate all the files that match [lib]ggml-name-*.[so|dll] in the search paths
    const fs::path name_path = fs::u8path(name);
    const fs::path file_prefix = backend_filename_prefix().native() + name_path.native() + fs::u8path("-").native();
    const fs::path file_extension = backend_filename_extension();

    std::vector<fs::path> search_paths;
    if (user_search_path == nullptr) {
#ifdef GGML_BACKEND_DIR  // 如果定义了 GGML_BACKEND_DIR 则编译
        search_paths.push_back(fs::u8path(GGML_BACKEND_DIR));
#endif  // 条件编译结束
        // default search paths: executable directory, current directory
        search_paths.push_back(get_executable_path());
        search_paths.push_back(fs::current_path());
    } else {
        search_paths.push_back(fs::u8path(user_search_path));
    }

    int best_score = 0;
    fs::path best_path;
    std::error_code ec;

    for (const auto & search_path : search_paths) {
        if (!fs::exists(search_path, ec)) {
            if (ec) {
                GGML_LOG_DEBUG("%s: posix_stat(%s) failure, error-message: %s\n", __func__, path_str(search_path).c_str(), ec.message().c_str());
            } else {
                GGML_LOG_DEBUG("%s: search path %s does not exist\n", __func__, path_str(search_path).c_str());
            }
            continue;
        }
        fs::directory_iterator dir_it(search_path, fs::directory_options::skip_permission_denied);
        for (const auto & entry : dir_it) {
            if (entry.is_regular_file(ec)) {
                auto filename = entry.path().filename();
                auto ext = entry.path().extension();
                if (filename.native().find(file_prefix) == 0 && ext == file_extension) {
                    dl_handle_ptr handle { dl_load_library(entry) };
                    if (!handle && !silent) {
                        GGML_LOG_ERROR("%s: failed to load %s: %s\n", __func__, path_str(entry.path()).c_str(), dl_error());
                    }
                    if (handle) {
                        auto score_fn = (ggml_backend_score_t) dl_get_sym(handle.get(), "ggml_backend_score");
                        if (score_fn) {
                            int s = score_fn();
#ifndef NDEBUG  // 如果未定义 NDEBUG 则编译
                            GGML_LOG_DEBUG("%s: %s score: %d\n", __func__, path_str(entry.path()).c_str(), s);
#endif  // 条件编译结束
                            if (s > best_score) {
                                best_score = s;
                                best_path = entry.path();
                            }
                        } else {
                            if (!silent) {
                                GGML_LOG_INFO("%s: failed to find ggml_backend_score in %s\n", __func__, path_str(entry.path()).c_str());
                            }
                        }
                    }
                }
            }
        }
    }

    if (best_score == 0) {
        // try to load the base backend
        for (const auto & search_path : search_paths) {
            fs::path filename = backend_filename_prefix().native() + name_path.native() + backend_filename_extension().native();
            fs::path path = search_path / filename;
            if (std::error_code ec; fs::exists(path, ec)) {
                return get_reg().load_backend(path, silent);  // get_reg
            } else {
                if (ec) {
                    GGML_LOG_DEBUG("%s: posix_stat(%s) failure, error-message: %s\n", __func__, path_str(path).c_str(), ec.message().c_str());
                }
            }
        }
        return nullptr;  // 返回
    }

    return get_reg().load_backend(best_path, silent);  // get_reg
}

void ggml_backend_load_all() {
    ggml_backend_load_all_from_path(nullptr);
}

void ggml_backend_load_all_from_path(const char * dir_path) {
#ifdef NDEBUG  // 如果定义了 NDEBUG 则编译
    bool silent = true;
#else  // 否则
    bool silent = false;
#endif  // 条件编译结束

    ggml_backend_load_best("blas", silent, dir_path);
    ggml_backend_load_best("zendnn", silent, dir_path);
    ggml_backend_load_best("cann", silent, dir_path);
    ggml_backend_load_best("cuda", silent, dir_path);
    ggml_backend_load_best("hip", silent, dir_path);
    ggml_backend_load_best("metal", silent, dir_path);
    ggml_backend_load_best("rpc", silent, dir_path);
    ggml_backend_load_best("sycl", silent, dir_path);
    ggml_backend_load_best("vulkan", silent, dir_path);
    ggml_backend_load_best("virtgpu", silent, dir_path);
    ggml_backend_load_best("opencl", silent, dir_path);
    ggml_backend_load_best("hexagon", silent, dir_path);
    ggml_backend_load_best("musa", silent, dir_path);
    ggml_backend_load_best("openvino", silent, dir_path);
    ggml_backend_load_best("cpu", silent, dir_path);
    // check the environment variable GGML_BACKEND_PATH to load an out-of-tree backend
    const char * backend_path = std::getenv("GGML_BACKEND_PATH");
    if (backend_path) {
        ggml_backend_load(backend_path);
    }
}
