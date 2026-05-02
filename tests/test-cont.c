#include "ggml-backend.h"  // 引入 ggml-backend.h 头文件
#include "ggml-cpu.h"  // 引入 ggml-cpu.h 头文件
#include "ggml.h"  // 引入 ggml.h 头文件

#ifdef GGML_USE_CUDA  // 如果定义了 GGML_USE_CUDA 则编译
#include "ggml-cuda.h"  // 引入 ggml-cuda.h 头文件
#endif  // 条件编译结束

#include <stdlib.h>  // 引入 stdlib.h 头文件
#include <string.h>  // 引入 string.h 头文件

struct model {  // 结构体定义
    struct ggml_context* ctx;
    struct ggml_context* ctx0;
    ggml_backend_t backend;
    ggml_backend_buffer_t buffer;
    struct ggml_cgraph* gf;
    ggml_gallocr_t allocr;
    uint8_t* buf;
};

struct ggml_context* make_ctx(void) {  // 结构体定义
    struct ggml_init_params params = {  // 结构体定义
        .mem_size = ggml_tensor_overhead() * 3,
        .mem_buffer = NULL,
        .no_alloc = true,
    };
    return ggml_init(params);  // ggml_init
}

ggml_backend_t make_backend(void) {
    ggml_backend_t backend = NULL;

#ifdef GGML_USE_CUDA  // 如果定义了 GGML_USE_CUDA 则编译
    backend = ggml_backend_cuda_init(0);
    GGML_ASSERT(backend != NULL);
#endif  // 条件编译结束

    if (!backend) {
        backend = ggml_backend_cpu_init();
    }

    return backend;  // 返回
}

void model_init(struct model* m) {
    m->ctx = make_ctx();
    m->backend = make_backend();

    size_t buf_size = ggml_tensor_overhead() * GGML_DEFAULT_GRAPH_SIZE + ggml_graph_overhead();
    m->buf = calloc(buf_size, sizeof(uint8_t));
    struct ggml_init_params params0 = {  // 结构体定义
        .mem_size = buf_size,
        .mem_buffer = m->buf,
        .no_alloc = true,
    };
    m->ctx0 = ggml_init(params0);
    m->gf = ggml_new_graph(m->ctx0);
}

void model_alloc(struct model* m) {
    m->buffer = ggml_backend_alloc_ctx_tensors(m->ctx, m->backend);
    m->allocr = ggml_gallocr_new(ggml_backend_get_default_buffer_type(m->backend));
}

void model_compute(struct model* m) {
    ggml_gallocr_alloc_graph(m->allocr, m->gf);
    ggml_backend_graph_compute(m->backend, m->gf);
}

void model_free(struct model* m) {
    ggml_free(m->ctx0);
    free(m->buf);
    ggml_gallocr_free(m->allocr);
    ggml_free(m->ctx);
    ggml_backend_buffer_free(m->buffer);
    ggml_backend_free(m->backend);
}

void check_tensor(struct ggml_tensor* t,
                  const float* expected_t_d,
                  const int ne0,
                  const int ne1,
                  const int ne2) {
    GGML_ASSERT(t->ne[0] == ne0);
    GGML_ASSERT(t->ne[1] == ne1);
    GGML_ASSERT(t->ne[2] == ne2);
    const size_t bsize = ggml_nbytes(t);
    if (t->type == GGML_TYPE_F32) {
        float* buffer = malloc(bsize);
        ggml_backend_tensor_get(t, buffer, 0, bsize);
        for (int i = 0; i < bsize / sizeof(float); ++i) {
            float expected = expected_t_d[i];
            float actual = buffer[i];
            if (expected != actual) {
                printf("expected %.1f, got %.1f\n", expected, actual);
            }
            GGML_ASSERT(expected == actual);
        }
        free(buffer);
    } else if (t->type == GGML_TYPE_F16) {
        ggml_fp16_t* buffer = malloc(bsize);
        ggml_backend_tensor_get(t, buffer, 0, bsize);
        for (int i = 0; i < bsize / sizeof(ggml_fp16_t); ++i) {
            float expected = expected_t_d[i];
            float actual = ggml_fp16_to_fp32(buffer[i]);
            if (expected != actual) {
                printf("expected %.1f, got %.1f\n", expected, actual);
            }
            GGML_ASSERT(expected == actual);
        }
        free(buffer);
    //} else if (t->type == GGML_TYPE_BF16) {
    //    ggml_bf16_t* buffer = malloc(bsize);
    //    ggml_backend_tensor_get(t, buffer, 0, bsize);
    //    for (int i = 0; i < bsize / sizeof(ggml_bf16_t); ++i) {
    //        float expected = expected_t_d[i];
    //        float actual = ggml_bf16_to_fp32(buffer[i]);
    //        if (expected != actual) {
    //            printf("expected %.1f, got %.1f\n", expected, actual);
    //        }
    //        GGML_ASSERT(expected == actual);
    //    }
    //    free(buffer);
    } else {
        GGML_ABORT("unknown type");
    }
}

void test_cont(void) {
    float buf_f32[] = {1.0, 2.0};
    ggml_fp16_t buf_f16[] = {ggml_fp32_to_fp16(buf_f32[0]), ggml_fp32_to_fp16(buf_f32[1])};
    ggml_bf16_t buf_bf16[] = {ggml_fp32_to_bf16(buf_f32[0]), ggml_fp32_to_bf16(buf_f32[1])};

    float expected_out[] = {1.0, 2.0};

    struct model m;
    model_init(&m);

    struct ggml_tensor* in_1 = ggml_new_tensor_1d(m.ctx, GGML_TYPE_F32, 2);
    struct ggml_tensor* in_2 = ggml_new_tensor_1d(m.ctx, GGML_TYPE_F16, 2);
    //struct ggml_tensor* in_3 = ggml_new_tensor_1d(m.ctx, GGML_TYPE_BF16, 2);

    model_alloc(&m);

    ggml_backend_tensor_set(in_1, buf_f32, 0, ggml_nbytes(in_1));
    ggml_backend_tensor_set(in_2, buf_f16, 0, ggml_nbytes(in_2));
    //ggml_backend_tensor_set(in_3, buf_bf16, 0, ggml_nbytes(in_3));

    struct ggml_tensor* out_1 = ggml_cont(m.ctx0, ggml_transpose(m.ctx0, in_1));
    struct ggml_tensor* out_2 = ggml_cont(m.ctx0, ggml_transpose(m.ctx0, in_2));
    //struct ggml_tensor* out_3 = ggml_cont(m.ctx0, ggml_transpose(m.ctx0, in_3));

    ggml_build_forward_expand(m.gf, out_1);
    ggml_build_forward_expand(m.gf, out_2);
    //ggml_build_forward_expand(m.gf, out_3);

    model_compute(&m);

    check_tensor(out_1, expected_out, 1, 2, 1);
    check_tensor(out_2, expected_out, 1, 2, 1);
    //check_tensor(out_3, expected_out, 1, 2, 1);

    model_free(&m);
}

int main(int argc, const char* argv[]) {
    test_cont();
    return 0;  // 返回
}
