#include "ggml-backend-impl.h"  // 引入 ggml-backend-impl.h 头文件

#if defined(__s390x__)  // 条件编译
#include <sys/auxv.h>  // 引入 sys/auxv.h 头文件

// find hwcap bits in asm/elf.h
#ifndef HWCAP_VXRS_EXT2  // 如果未定义 HWCAP_VXRS_EXT2 则编译
#define HWCAP_VXRS_EXT2 (1 << 15)  // 宏定义 HWCAP_VXRS_EXT2
#endif  // 条件编译结束

#ifndef HWCAP_NNPA  // 如果未定义 HWCAP_NNPA 则编译
#define HWCAP_NNPA (1 << 20)  // 宏定义 HWCAP_NNPA
#endif  // 条件编译结束

struct s390x_features {  // 结构体定义
    bool has_vxe2 = false;
    bool has_nnpa = false;

    s390x_features() {
        uint32_t hwcap = getauxval(AT_HWCAP);
        // NOTE: use hwcap2 with DFLT for z17 and later
        // uint32_t hwcap2 = getauxval(AT_HWCAP2);

        has_vxe2 = !!(hwcap & HWCAP_VXRS_EXT2);
        has_nnpa = !!(hwcap & HWCAP_NNPA);
    }
};

static int ggml_backend_cpu_s390x_score() {
    int score = 1;
    s390x_features sf;

// IBM z15 / LinuxONE 3
#ifdef GGML_USE_VXE2  // 如果定义了 GGML_USE_VXE2 则编译
    if (!sf.has_vxe2) { return 0; }
    score += 1 << 1;
#endif  // 条件编译结束

// IBM z16 / LinuxONE 4 and z17 / LinuxONE 5
#ifdef GGML_USE_NNPA  // 如果定义了 GGML_USE_NNPA 则编译
    if (!sf.has_nnpa) { return 0; }
    score += 1 << 2;
#endif  // 条件编译结束

    return score;  // 返回
}

GGML_BACKEND_DL_SCORE_IMPL(ggml_backend_cpu_s390x_score)

#endif  // __s390x__  // 条件编译结束
