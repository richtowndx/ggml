#include <algorithm>  // 引入 algorithm 头文件
#include <cstdint>  // 引入 cstdint 头文件
#include <random>  // 引入 random 头文件
#include <string>  // 引入 string 头文件
#include <thread>  // 引入 thread 头文件
#include <vector>  // 引入 vector 头文件

#include "ggml-alloc.h"  // 引入 ggml-alloc.h 头文件
#include "ggml-backend.h"  // 引入 ggml-backend.h 头文件
#include "ggml.h"  // 引入 ggml.h 头文件
#include "gguf.h"  // 引入 gguf.h 头文件
#include "ggml-cpu.h"  // 引入 ggml-cpu.h 头文件
#include "ggml-opt.h"  // 引入 ggml-opt.h 头文件

#define MNIST_NTRAIN 60000  // 宏定义 MNIST_NTRAIN
#define MNIST_NTEST  10000  // 宏定义 MNIST_NTEST

// Gradient accumulation can be achieved by setting the logical batch size to a multiple of the physical one.
// The logical batch size determines how many datapoints are used for a gradient update.
// The physical batch size determines how many datapoints are processed in parallel, larger values utilize compute better but need more memory.
#define MNIST_NBATCH_LOGICAL  1000  // 宏定义 MNIST_NBATCH_LOGICAL
#define MNIST_NBATCH_PHYSICAL  500  // 宏定义 MNIST_NBATCH_PHYSICAL

static_assert(MNIST_NBATCH_LOGICAL % MNIST_NBATCH_PHYSICAL == 0, "MNIST_NBATCH_LOGICAL % MNIST_NBATCH_PHYSICAL != 0");
static_assert(MNIST_NTRAIN % MNIST_NBATCH_LOGICAL == 0, "MNIST_NTRAIN % MNIST_NBATCH_LOGICAL != 0");
static_assert(MNIST_NTEST  % MNIST_NBATCH_LOGICAL == 0, "MNIST_NTRAIN % MNIST_NBATCH_LOGICAL != 0");

#define MNIST_HW       28  // 宏定义 MNIST_HW
#define MNIST_NINPUT   (MNIST_HW*MNIST_HW)  // 宏定义 MNIST_NINPUT
#define MNIST_NCLASSES 10  // 宏定义 MNIST_NCLASSES

#define MNIST_NHIDDEN  500  // 宏定义 MNIST_NHIDDEN

// NCB = number of channels base
#define MNIST_CNN_NCB 8  // 宏定义 MNIST_CNN_NCB

struct mnist_model {  // 结构体定义
    std::string arch;
    ggml_backend_sched_t backend_sched;
    std::vector<ggml_backend_t> backends;
    const int nbatch_logical;
    const int nbatch_physical;

    struct ggml_tensor * images     = nullptr;
    struct ggml_tensor * logits     = nullptr;

    struct ggml_tensor * fc1_weight = nullptr;
    struct ggml_tensor * fc1_bias   = nullptr;
    struct ggml_tensor * fc2_weight = nullptr;
    struct ggml_tensor * fc2_bias   = nullptr;

    struct ggml_tensor * conv1_kernel = nullptr;
    struct ggml_tensor * conv1_bias   = nullptr;
    struct ggml_tensor * conv2_kernel = nullptr;
    struct ggml_tensor * conv2_bias   = nullptr;
    struct ggml_tensor * dense_weight = nullptr;
    struct ggml_tensor * dense_bias   = nullptr;

    struct ggml_context * ctx_gguf    = nullptr;
    struct ggml_context * ctx_static  = nullptr;
    struct ggml_context * ctx_compute = nullptr;
    ggml_backend_buffer_t buf_gguf    = nullptr;
    ggml_backend_buffer_t buf_static  = nullptr;

    mnist_model(const std::string & backend_name, const int nbatch_logical, const int nbatch_physical)
            : nbatch_logical(nbatch_logical), nbatch_physical(nbatch_physical) {
        std::vector<ggml_backend_dev_t> devices;
        const int ncores_logical = std::thread::hardware_concurrency();
        const int nthreads = std::min(ncores_logical, (ncores_logical + 4) / 2);

        // Add primary backend:
        if (!backend_name.empty()) {
            ggml_backend_dev_t dev = ggml_backend_dev_by_name(backend_name.c_str());
            if (dev == nullptr) {
                fprintf(stderr, "%s: ERROR: backend %s not found, available:\n", __func__, backend_name.c_str());
                for (size_t i = 0; i < ggml_backend_dev_count(); ++i) {
                    ggml_backend_dev_t dev_i = ggml_backend_dev_get(i);
                    fprintf(stderr, "  - %s (%s)\n", ggml_backend_dev_name(dev_i), ggml_backend_dev_description(dev_i));
                }
                exit(1);
            }

            ggml_backend_t backend = ggml_backend_dev_init(dev, nullptr);
            GGML_ASSERT(backend);

            if (ggml_backend_is_cpu(backend)) {
                ggml_backend_cpu_set_n_threads(backend, nthreads);
            }

            backends.push_back(backend);
            devices.push_back(dev);
        }

        // Add all available backends as fallback.
        // A "backend" is a stream on a physical device so there is no problem with adding multiple backends for the same device.
        for (size_t i = 0; i < ggml_backend_dev_count(); ++i) {
            ggml_backend_dev_t dev = ggml_backend_dev_get(i);

            ggml_backend_t backend = ggml_backend_dev_init(dev, nullptr);
            GGML_ASSERT(backend);

            if (ggml_backend_is_cpu(backend)) {
                ggml_backend_cpu_set_n_threads(backend, nthreads);
            }

            backends.push_back(backend);
            devices.push_back(dev);
        }

        // The order of the backends passed to ggml_backend_sched_new determines which backend is given priority.
        backend_sched = ggml_backend_sched_new(backends.data(), nullptr, backends.size(), GGML_DEFAULT_GRAPH_SIZE, false, true);
        fprintf(stderr, "%s: using %s (%s) as primary backend\n",
                __func__, ggml_backend_name(backends[0]), ggml_backend_dev_description(devices[0]));
        if (backends.size() >= 2) {
            fprintf(stderr, "%s: unsupported operations will be executed on the following fallback backends (in order of priority):\n", __func__);
            for (size_t i = 1; i < backends.size(); ++i) {
                fprintf(stderr, "%s:  - %s (%s)\n", __func__, ggml_backend_name(backends[i]), ggml_backend_dev_description(devices[i]));
            }
        }

        {
            const size_t size_meta = 1024*ggml_tensor_overhead();
            struct ggml_init_params params = {  // 结构体定义
                /*.mem_size   =*/ size_meta,
                /*.mem_buffer =*/ nullptr,
                /*.no_alloc   =*/ true,
            };
            ctx_static = ggml_init(params);
        }

        {
            // The compute context needs a total of 3 compute graphs: forward pass + backwards pass (with/without optimizer step).
            const size_t size_meta = GGML_DEFAULT_GRAPH_SIZE*ggml_tensor_overhead() + 3*ggml_graph_overhead();
            struct ggml_init_params params = {  // 结构体定义
                /*.mem_size   =*/ size_meta,
                /*.mem_buffer =*/ nullptr,
                /*.no_alloc   =*/ true,
            };
            ctx_compute = ggml_init(params);
        }
    }

    ~mnist_model() {
        ggml_free(ctx_gguf);
        ggml_free(ctx_static);
        ggml_free(ctx_compute);

        ggml_backend_buffer_free(buf_gguf);
        ggml_backend_buffer_free(buf_static);
        ggml_backend_sched_free(backend_sched);
        for (ggml_backend_t backend : backends) {
            ggml_backend_free(backend);
        }
    }
};

bool mnist_image_load(const std::string & fname, ggml_opt_dataset_t dataset);  // mnist_image_load
void mnist_image_print(FILE * f, ggml_opt_dataset_t dataset, const int iex);  // mnist_image_print
bool mnist_label_load(const std::string & fname, ggml_opt_dataset_t dataset);  // mnist_label_load

mnist_model       mnist_model_init_from_file(const std::string & fname, const std::string & backend, const int nbatch_logical, const int nbatch_physical);  // mnist_model_init_from_file
mnist_model       mnist_model_init_random(const std::string & arch, const std::string & backend, const int nbatch_logical, const int nbatch_physical);  // mnist_model_init_random
void              mnist_model_build(mnist_model & model);  // mnist_model_build
ggml_opt_result_t mnist_model_eval(mnist_model & model, ggml_opt_dataset_t dataset);  // mnist_model_eval
void              mnist_model_train(mnist_model & model, ggml_opt_dataset_t dataset, const int nepoch, const float val_split);  // mnist_model_train
void              mnist_model_save(mnist_model & model, const std::string & fname);  // mnist_model_save
