#pragma once  // 防止重复包含

#ifdef __cplusplus  // 如果定义了 __cplusplus 则编译
extern "C" {  // C 链接声明
#endif  // 条件编译结束

#ifdef _WIN32  // 如果定义了 _WIN32 则编译
#    pragma clang diagnostic ignored "-Wignored-attributes"
#endif  // 条件编译结束

#include <AEEStdErr.h>  // 引入 AEEStdErr.h 头文件
#include <rpcmem.h>  // 引入 rpcmem.h 头文件
#include <remote.h>  // 引入 remote.h 头文件
#include <dspqueue.h>  // 引入 dspqueue.h 头文件

#if defined(_WIN32) && !defined(__MINGW32__)  // 条件编译
#    ifdef GGML_BACKEND_BUILD  // 如果定义了 GGML_BACKEND_BUILD 则编译
#        define HTPDRV_API __declspec(dllexport) extern  // 宏定义 HTPDRV_API
#    else
#        define HTPDRV_API __declspec(dllimport) extern  // 宏定义 HTPDRV_API
#    endif
#else  // 否则
#    define HTPDRV_API __attribute__ ((visibility ("default"))) extern  // 宏定义 HTPDRV_API
#endif  // 条件编译结束

/* Offset to differentiate HLOS and Hexagon error codes.
   Stores the value of AEE_EOFFSET for Hexagon. */
#ifndef DSP_OFFSET  // 如果未定义 DSP_OFFSET 则编译
#    define DSP_OFFSET 0x80000400  // 宏定义 DSP_OFFSET
#endif  // 条件编译结束

/* Errno for connection reset by peer. */
#ifndef ECONNRESET  // 如果未定义 ECONNRESET 则编译
#    ifdef __hexagon__  // 如果定义了 __hexagon__ 则编译
#        define ECONNRESET 104  // 宏定义 ECONNRESET
#    endif
#endif  // 条件编译结束

/* Abstraction of different OS specific sleep APIs.
   SLEEP accepts input in seconds. */
#ifndef SLEEP  // 如果未定义 SLEEP 则编译
#    ifdef __hexagon__  // 如果定义了 __hexagon__ 则编译
#        define SLEEP(x)                      \
            { /* Do nothing for simulator. */ \
            }
#    else
#        ifdef _WIN32  // 如果定义了 _WIN32 则编译
#            define SLEEP(x) Sleep(1000 * x) /* Sleep accepts input in milliseconds. */  // 宏定义 SLEEP
#        else
#            define SLEEP(x) sleep(x)        /* sleep accepts input in seconds. */  // 宏定义 SLEEP
#        endif
#    endif
#endif  // 条件编译结束

/* Include windows specific header files. */
#ifdef _WIN32  // 如果定义了 _WIN32 则编译
#    include <windows.h>  // 引入 windows.h 头文件
#    include <sysinfoapi.h>  // 引入 sysinfoapi.h 头文件
#    define _CRT_SECURE_NO_WARNINGS         1  // 宏定义 _CRT_SECURE_NO_WARNINGS
#    define _WINSOCK_DEPRECATED_NO_WARNINGS 1  // 宏定义 _WINSOCK_DEPRECATED_NO_WARNINGS
#endif  // 条件编译结束

/* Includes and defines for all HLOS except windows */
#if !defined(__hexagon__) && !defined(_WIN32)  // 条件编译
#    include "unistd.h"  // 引入 unistd.h 头文件

#    include <sys/time.h>  // 引入 sys/time.h 头文件
#endif  // 条件编译结束

/* Includes and defines for Hexagon and all HLOS except Windows. */
#if !defined(_WIN32)  // 条件编译
/* Weak reference to remote symbol for compilation. */
#    pragma weak remote_session_control
#    pragma weak remote_handle_control
#    pragma weak remote_handle64_control
#    pragma weak fastrpc_mmap
#    pragma weak fastrpc_munmap
#    pragma weak rpcmem_alloc2
#endif  // 条件编译结束

#if !defined(_WIN32)  // 条件编译
#    pragma weak remote_system_request
#endif  // 条件编译结束

#ifdef _WIN32  // 如果定义了 _WIN32 则编译
#     define DSPQUEUE_TIMEOUT DSPQUEUE_TIMEOUT_NONE  // 宏定义 DSPQUEUE_TIMEOUT
#else  // 否则
#     define DSPQUEUE_TIMEOUT 1000000  // 宏定义 DSPQUEUE_TIMEOUT
#endif  // 条件编译结束

/**
 * htpdrv_init API: driver interface entry point
 *
 * @return      Return AEE error codes as defined in Hexagon SDK.
 */
HTPDRV_API int htpdrv_init(void);

/**
 * get_domain API: get domain struct from domain value.
 *
 * @param[in]  domain value of a domain
 * @return     Returns domain struct of the domain if it is supported or else
 *             returns NULL.
 *
 */
HTPDRV_API domain * get_domain(int domain_id);

/**
 * get_hex_arch_ver API: query the Hexagon processor architecture version information
 *
 * @param[in]   domain_id value of a domain
 * @param[out]  Arch version (73, 75, ...)
 * @return      0 if query is successful.
 *              non-zero if error, return value points to the error.
 *
 */
HTPDRV_API int get_hex_arch_ver(int domain, int * arch);

#ifdef __cplusplus  // 如果定义了 __cplusplus 则编译
}
#endif  // 条件编译结束
