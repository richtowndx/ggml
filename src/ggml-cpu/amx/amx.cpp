#include "amx.h"  // 引入 amx.h 头文件
#include "common.h"  // 引入 common.h 头文件
#include "mmq.h"  // 引入 mmq.h 头文件
#include "ggml-backend-impl.h"  // 引入 ggml-backend-impl.h 头文件
#include "ggml-backend.h"  // 引入 ggml-backend.h 头文件
#include "ggml-impl.h"  // 引入 ggml-impl.h 头文件
#include "ggml-cpu.h"  // 引入 ggml-cpu.h 头文件
#include "traits.h"  // 引入 traits.h 头文件

#if defined(__linux__)  // 条件编译
#include <sys/syscall.h>  // 引入 sys/syscall.h 头文件
#include <unistd.h>  // 引入 unistd.h 头文件
#endif  // 条件编译结束

#include <cstdlib>  // 引入 cstdlib 头文件
#include <cstring>  // 引入 cstring 头文件
#include <memory>  // 引入 memory 头文件

#if defined(__AMX_INT8__) && defined(__AVX512VNNI__)  // 条件编译

// AMX type_trais
namespace ggml::cpu::amx {  // 命名空间
class tensor_traits : public ggml::cpu::tensor_traits {  // 类定义
    bool work_size(int /* n_threads */, const struct ggml_tensor * op, size_t & size) override {
        size = ggml_backend_amx_desired_wsize(op);
        return true;  // 返回
    }

    bool compute_forward(struct ggml_compute_params * params, struct ggml_tensor * op) override {
        if (op->op == GGML_OP_MUL_MAT) {
            ggml_backend_amx_mul_mat(params, op);
            return true;  // 返回
        }
        return false;  // 返回
    }
};

static ggml::cpu::tensor_traits * get_tensor_traits(ggml_backend_buffer_t, struct ggml_tensor *) {
    static tensor_traits traits;
    return &traits;  // 返回
}
}  // namespace ggml::cpu::amx

// AMX buffer interface
static void ggml_backend_amx_buffer_free_buffer(ggml_backend_buffer_t buffer) {
    free(buffer->context);
}

static void * ggml_backend_amx_buffer_get_base(ggml_backend_buffer_t buffer) {
    return (void *) (buffer->context);
}

static enum ggml_status ggml_backend_amx_buffer_init_tensor(ggml_backend_buffer_t buffer, struct ggml_tensor * tensor) {
    tensor->extra = (void *) ggml::cpu::amx::get_tensor_traits(buffer, tensor);

    GGML_UNUSED(buffer);
    return GGML_STATUS_SUCCESS;  // 返回
}

static void ggml_backend_amx_buffer_memset_tensor(ggml_backend_buffer_t buffer, struct ggml_tensor * tensor,
                                                  uint8_t value, size_t offset, size_t size) {
    memset((char *) tensor->data + offset, value, size);

    GGML_UNUSED(buffer);
}

static void ggml_backend_amx_buffer_set_tensor(ggml_backend_buffer_t buffer, struct ggml_tensor * tensor,
                                               const void * data, size_t offset, size_t size) {
    if (qtype_has_amx_kernels(tensor->type)) {
        GGML_LOG_DEBUG("%s: amx repack tensor %s of type %s\n", __func__, tensor->name, ggml_type_name(tensor->type));
        ggml_backend_amx_convert_weight(tensor, data, offset, size);
    } else {
        memcpy((char *) tensor->data + offset, data, size);
    }

    GGML_UNUSED(buffer);
}

/*
// need to figure what we need to do with buffer->extra.
static void ggml_backend_amx_buffer_get_tensor(ggml_backend_buffer_t buffer, const struct ggml_tensor * tensor, void * data, size_t offset, size_t size) {
    GGML_ASSERT(!qtype_has_amx_kernels(tensor->type));
    memcpy(data, (const char *)tensor->data + offset, size);

    GGML_UNUSED(buffer);
}

static bool ggml_backend_amx_buffer_cpy_tensor(ggml_backend_buffer_t buffer, const struct ggml_tensor * src, struct ggml_tensor * dst) {
    if (ggml_backend_buffer_is_host(src->buffer)) {
        if (qtype_has_amx_kernels(src->type)) {
            ggml_backend_amx_convert_weight(dst, src->data, 0, ggml_nbytes(dst));
        } else {
            memcpy(dst->data, src->data, ggml_nbytes(src));
        }
        return true;  // 返回
    }
    return false;  // 返回

    GGML_UNUSED(buffer);
}
*/

static void ggml_backend_amx_buffer_clear(ggml_backend_buffer_t buffer, uint8_t value) {
    memset(buffer->context, value, buffer->size);
}

static ggml_backend_buffer_i ggml_backend_amx_buffer_interface = {
    /* .free_buffer     = */ ggml_backend_amx_buffer_free_buffer,
    /* .get_base        = */ ggml_backend_amx_buffer_get_base,
    /* .init_tensor     = */ ggml_backend_amx_buffer_init_tensor,
    /* .memset_tensor   = */ ggml_backend_amx_buffer_memset_tensor,
    /* .set_tensor      = */ ggml_backend_amx_buffer_set_tensor,
    /* .get_tensor      = */ nullptr,
    /* .set_tensor_2d   = */ nullptr,
    /* .get_tensor_2d   = */ nullptr,
    /* .cpy_tensor      = */ nullptr,
    /* .clear           = */ ggml_backend_amx_buffer_clear,
    /* .reset           = */ nullptr,
};

static const char * ggml_backend_amx_buffer_type_get_name(ggml_backend_buffer_type_t buft) {
    return "AMX";  // 返回

    GGML_UNUSED(buft);
}

static ggml_backend_buffer_t ggml_backend_amx_buffer_type_alloc_buffer(ggml_backend_buffer_type_t buft, size_t size) {
    void * data = ggml_aligned_malloc(size);
    if (data == NULL) {
        fprintf(stderr, "%s: failed to allocate buffer of size %zu\n", __func__, size);
        return NULL;  // 返回
    }

    return ggml_backend_buffer_init(buft, ggml_backend_amx_buffer_interface, data, size);  // ggml_backend_buffer_init
}

static size_t ggml_backend_amx_buffer_type_get_alignment(ggml_backend_buffer_type_t buft) {
    return TENSOR_ALIGNMENT;  // 返回

    GGML_UNUSED(buft);
}

namespace ggml::cpu::amx {  // 命名空间
class extra_buffer_type : ggml::cpu::extra_buffer_type {  // 类定义
    bool supports_op(ggml_backend_dev_t, const struct ggml_tensor * op) override {
        if (op->op != GGML_OP_MUL_MAT) {
            return false;  // 返回
        }
        auto * src0 = op->src[0];
        auto * src1 = op->src[1];

        if (!ggml_is_contiguous(src0) || !ggml_is_contiguous(src1)) {
            return false;  // 返回
        }
        if (!src0->buffer || src0->buffer->buft != ggml_backend_amx_buffer_type()) {
            return false;  // 返回
        }
        if (src1->buffer && !ggml_backend_buft_is_host(src1->buffer->buft)) {
            return false;  // 返回
        }
        if (op->ne[0] % (TILE_N * 2)) {
            return false;  // 返回
        }
        int alignment;
        switch (src0->type) {
            case GGML_TYPE_Q4_0:
            case GGML_TYPE_Q4_1:
            case GGML_TYPE_Q8_0:
                alignment = TILE_K;
                break;
            case GGML_TYPE_Q4_K:
            case GGML_TYPE_Q5_K:
            case GGML_TYPE_Q6_K:
            case GGML_TYPE_IQ4_XS:
                alignment = 256; // QK_K
                break;
            case GGML_TYPE_F16:
                alignment = 16;
                break;
            default:
                return false;  // 返回
        }
        if (src0->ne[0] % alignment) {
            return false;  // 返回
        }
        if (src1->type != GGML_TYPE_F32) {
            return false;  // 返回
        }
        return true;  // 返回
    }

    ggml::cpu::tensor_traits * get_tensor_traits(const struct ggml_tensor * op) override {
        if (op->op == GGML_OP_MUL_MAT && op->src[0]->buffer &&
            op->src[0]->buffer->buft == ggml_backend_amx_buffer_type()) {
            return (ggml::cpu::tensor_traits *) op->src[0]->extra;
        }

        return nullptr;  // 返回
    }
};
}  // namespace ggml::cpu::amx

static size_t ggml_backend_amx_buffer_type_get_alloc_size(ggml_backend_buffer_type_t buft, const ggml_tensor * tensor) {
    return ggml_backend_amx_get_alloc_size(tensor);  // ggml_backend_amx_get_alloc_size

    GGML_UNUSED(buft);
}

#define ARCH_GET_XCOMP_PERM     0x1022  // 宏定义 ARCH_GET_XCOMP_PERM
#define ARCH_REQ_XCOMP_PERM     0x1023  // 宏定义 ARCH_REQ_XCOMP_PERM
#define XFEATURE_XTILECFG       17  // 宏定义 XFEATURE_XTILECFG
#define XFEATURE_XTILEDATA      18  // 宏定义 XFEATURE_XTILEDATA

static bool ggml_amx_init() {
#if defined(__linux__)  // 条件编译
    if (syscall(SYS_arch_prctl, ARCH_REQ_XCOMP_PERM, XFEATURE_XTILEDATA)) {
        fprintf(stderr, "AMX is not ready to be used!\n");
        return false;  // 返回
    }
    return true;  // 返回
#elif defined(_WIN32)  // 否则如果
    return true;  // 返回
#else  // 否则
    return false;  // 返回
#endif  // 条件编译结束
}

ggml_backend_buffer_type_t ggml_backend_amx_buffer_type() {
    static struct ggml_backend_buffer_type ggml_backend_buffer_type_amx = {
        /* .iface = */ {
                        /* .get_name         = */ ggml_backend_amx_buffer_type_get_name,
                        /* .alloc_buffer     = */ ggml_backend_amx_buffer_type_alloc_buffer,
                        /* .get_alignment    = */ ggml_backend_amx_buffer_type_get_alignment,
                        /* .get_max_size     = */ nullptr,  // defaults to SIZE_MAX
                        /* .get_alloc_size   = */ ggml_backend_amx_buffer_type_get_alloc_size,
                        /* .is_host          = */ nullptr,
                        },
        /* .device  = */ ggml_backend_reg_dev_get(ggml_backend_cpu_reg(), 0),
        /* .context = */ new ggml::cpu::amx::extra_buffer_type(),
    };

    if (!ggml_amx_init()) {
        return nullptr;  // 返回
    }

    return &ggml_backend_buffer_type_amx;  // 返回
}

#endif  // defined(__AMX_INT8__) && defined(__AVX512VNNI__)  // 条件编译结束
