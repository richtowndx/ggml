#include "ggml-backend-impl.h"  // 引入 ggml-backend-impl.h 头文件

#if defined(__riscv) && __riscv_xlen == 64  // 条件编译
#include <asm/hwprobe.h>  // 引入 asm/hwprobe.h 头文件
#include <asm/unistd.h>  // 引入 asm/unistd.h 头文件
#include <unistd.h>  // 引入 unistd.h 头文件

struct riscv64_features {  // 结构体定义
    bool has_rvv = false;

    riscv64_features() {
        struct riscv_hwprobe probe;
        probe.key = RISCV_HWPROBE_KEY_IMA_EXT_0;
        probe.value = 0;

        int ret = syscall(__NR_riscv_hwprobe, &probe, 1, 0, NULL, 0);

        if (0 == ret) {
            has_rvv = !!(probe.value & RISCV_HWPROBE_IMA_V);
        }
    }
};

static int ggml_backend_cpu_riscv64_score() {
    int score = 1;
    riscv64_features rf;

#ifdef GGML_USE_RVV  // 如果定义了 GGML_USE_RVV 则编译
    if (!rf.has_rvv) { return 0; }
    score += 1 << 1;
#endif  // 条件编译结束

    return score;  // 返回
}

GGML_BACKEND_DL_SCORE_IMPL(ggml_backend_cpu_riscv64_score)

#endif  // __riscv && __riscv_xlen == 64  // 条件编译结束
