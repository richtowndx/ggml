#include "ggml.h"  // 引入 ggml.h 头文件
#include "ggml-cpu.h"  // 引入 ggml-cpu.h 头文件
#include "ggml-alloc.h"  // 引入 ggml-alloc.h 头文件
#include "ggml-backend.h"  // 引入 ggml-backend.h 头文件

#ifdef GGML_USE_CUDA  // 如果定义了 GGML_USE_CUDA 则编译
#include "ggml-cuda.h"  // 引入 ggml-cuda.h 头文件
#endif  // 条件编译结束

#ifdef GGML_USE_METAL  // 如果定义了 GGML_USE_METAL 则编译
#include "ggml-metal.h"  // 引入 ggml-metal.h 头文件
#endif  // 条件编译结束

#include <string.h>  // 引入 string.h 头文件
#include <stdio.h>  // 引入 stdio.h 头文件
#include <stdlib.h>  // 引入 stdlib.h 头文件

int main(int /*argc*/, const char** /*argv*/) {
    {
        bool use_gpu = true;
        GGML_UNUSED(use_gpu);

        ggml_backend_t backend = NULL;
        //ggml_backend_buffer_t buffer;

        #ifdef GGML_USE_CUDA  // 如果定义了 GGML_USE_CUDA 则编译
        if (use_gpu) {
            fprintf(stderr, "%s: using CUDA backend\n", __func__);
            backend = ggml_backend_cuda_init(0);
            if (!backend) {
                fprintf(stderr, "%s: ggml_backend_cuda_init() failed\n", __func__);
            }
        }
        #endif  // 条件编译结束

        #ifdef GGML_USE_METAL  // 如果定义了 GGML_USE_METAL 则编译
        if (!backend) {
            fprintf(stderr, "%s: using Metal backend\n", __func__);
            backend = ggml_backend_metal_init();
            if (!backend) {
                fprintf(stderr, "%s: ggml_backend_metal_init() failed\n", __func__);
            }
        }
        #endif  // 条件编译结束

        const int num_tensors = 2;

        struct ggml_init_params params = {  // 结构体定义
                /*.mem_size   =*/ ggml_tensor_overhead() * num_tensors + 2 * 1024 * 1024,
                /*.mem_size   =*/ NULL,
                /*.mem_size   =*/ true,
        };

        if (!backend) {
            // fallback to CPU backend
            backend = ggml_backend_cpu_init();
        }

        // create context
        struct ggml_context* ctx = ggml_init(params);
        struct ggml_tensor * t = ggml_arange(ctx, 0, 3, 1);

        GGML_ASSERT(t->ne[0] == 3);

        ggml_gallocr_t galloc = ggml_gallocr_new(ggml_backend_get_default_buffer_type(backend));

        struct ggml_cgraph * graph = ggml_new_graph(ctx);
        ggml_build_forward_expand(graph, t);

        // allocate tensors
        ggml_gallocr_alloc_graph(galloc, graph);

        int n_threads = 4;

        if (ggml_backend_is_cpu(backend)) {
            ggml_backend_cpu_set_n_threads(backend, n_threads);
        }

        ggml_backend_graph_compute(backend, graph);

        float * output = new float[ggml_nelements(t)];
        ggml_backend_tensor_get(t, output, 0, ggml_nbytes(t));

        for (int i = 0; i < t->ne[0]; i++) {
            printf("%.2f ", output[i]);
        }
        printf("\n");

        GGML_ASSERT(output[0] == 0);
        GGML_ASSERT(output[1] == 1);
        GGML_ASSERT(output[2] == 2);

        delete[] output;
        ggml_free(ctx);
        ggml_gallocr_free(galloc);
        ggml_backend_free(backend);
    }

    return 0;  // 返回
}
