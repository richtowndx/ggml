#include "ggml-backend-impl.h"  // 引入 ggml-backend-impl.h 头文件

#if defined(__aarch64__)  // 条件编译

#if defined(__linux__)  // 条件编译
#include <sys/auxv.h>  // 引入 sys/auxv.h 头文件
#elif defined(__APPLE__)  // 否则如果
#include <sys/sysctl.h>  // 引入 sys/sysctl.h 头文件
#endif  // 条件编译结束

#if !defined(HWCAP2_SVE2)  // 条件编译
#define HWCAP2_SVE2 (1 << 1)  // 宏定义 HWCAP2_SVE2
#endif  // 条件编译结束

#if !defined(HWCAP2_I8MM)  // 条件编译
#define HWCAP2_I8MM (1 << 13)  // 宏定义 HWCAP2_I8MM
#endif  // 条件编译结束

#if !defined(HWCAP2_SME)  // 条件编译
#define HWCAP2_SME (1 << 23)  // 宏定义 HWCAP2_SME
#endif  // 条件编译结束

struct aarch64_features {  // 结构体定义
    // has_neon not needed, aarch64 has NEON guaranteed
    bool has_dotprod     = false;
    bool has_fp16_va     = false;
    bool has_sve         = false;
    bool has_sve2        = false;
    bool has_i8mm        = false;
    bool has_sme         = false;

    aarch64_features() {
#if defined(__linux__)  // 条件编译
        uint32_t hwcap = getauxval(AT_HWCAP);
        uint32_t hwcap2 = getauxval(AT_HWCAP2);

        has_dotprod = !!(hwcap & HWCAP_ASIMDDP);
        has_fp16_va = !!(hwcap & HWCAP_FPHP);
        has_sve     = !!(hwcap & HWCAP_SVE);
        has_sve2    = !!(hwcap2 & HWCAP2_SVE2);
        has_i8mm    = !!(hwcap2 & HWCAP2_I8MM);
        has_sme     = !!(hwcap2 & HWCAP2_SME);
#elif defined(__APPLE__)  // 否则如果
        int oldp = 0;
        size_t size = sizeof(oldp);

        if (sysctlbyname("hw.optional.arm.FEAT_DotProd", &oldp, &size, NULL, 0) == 0) {
            has_dotprod = static_cast<bool>(oldp);
        }

        if (sysctlbyname("hw.optional.arm.FEAT_I8MM", &oldp, &size, NULL, 0) == 0) {
            has_i8mm = static_cast<bool>(oldp);
        }

        if (sysctlbyname("hw.optional.arm.FEAT_SME", &oldp, &size, NULL, 0) == 0) {
            has_sme = static_cast<bool>(oldp);
        }

        // Apple apparently does not implement SVE yet
#endif  // 条件编译结束
    }
};

static int ggml_backend_cpu_aarch64_score() {
    int score = 1;
    aarch64_features af;

#ifdef GGML_USE_DOTPROD  // 如果定义了 GGML_USE_DOTPROD 则编译
    if (!af.has_dotprod) { return 0; }
    score += 1<<1;
#endif  // 条件编译结束
#ifdef GGML_USE_FP16_VECTOR_ARITHMETIC  // 如果定义了 GGML_USE_FP16_VECTOR_ARITHMETIC 则编译
    if (!af.has_fp16_va) { return 0; }
    score += 1<<2;
#endif  // 条件编译结束
#ifdef GGML_USE_SVE  // 如果定义了 GGML_USE_SVE 则编译
    if (!af.has_sve) { return 0; }
    score += 1<<3;
#endif  // 条件编译结束
#ifdef GGML_USE_MATMUL_INT8  // 如果定义了 GGML_USE_MATMUL_INT8 则编译
    if (!af.has_i8mm) { return 0; }
    score += 1<<4;
#endif  // 条件编译结束
#ifdef GGML_USE_SVE2  // 如果定义了 GGML_USE_SVE2 则编译
    if (!af.has_sve2) { return 0; }
    score += 1<<5;
#endif  // 条件编译结束
#ifdef GGML_USE_SME  // 如果定义了 GGML_USE_SME 则编译
    if (!af.has_sme) { return 0; }
    score += 1<<6;
#endif  // 条件编译结束

    return score;  // 返回
}

GGML_BACKEND_DL_SCORE_IMPL(ggml_backend_cpu_aarch64_score)

# endif // defined(__aarch64__)
