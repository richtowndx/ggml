#include "ggml-backend-impl.h"  // 引入 ggml-backend-impl.h 头文件
#include "ggml-decoder.h"  // 引入 ggml-decoder.h 头文件
#include "ggml-impl.h"  // 引入 ggml-impl.h 头文件

#include <algorithm>  // 引入 algorithm 头文件
#include <atomic>  // 引入 atomic 头文件
#include <cstddef>  // 引入 cstddef 头文件
#include <memory>  // 引入 memory 头文件
#include <mutex>  // 引入 mutex 头文件
#include <openvino/runtime/core.hpp>  // 引入 openvino/runtime/core.hpp 头文件
#include <openvino/runtime/infer_request.hpp>  // 引入 openvino/runtime/infer_request.hpp 头文件
#include <string>  // 引入 string 头文件
#include <unordered_map>  // 引入 unordered_map 头文件
#include <utility>  // 引入 utility 头文件
#include <vector>  // 引入 vector 头文件

struct graph_key {  // 结构体定义
    int n_nodes;
    std::string first_node_name;
    std::string last_node_name;

    graph_key(const ggml_cgraph * cgraph) : n_nodes(cgraph->n_nodes) {
        if (n_nodes > 0) {
            first_node_name = cgraph->nodes[0]->name;
            last_node_name = cgraph->nodes[n_nodes - 1]->name;
        }
    }

    bool operator==(const graph_key & other) const {
        return n_nodes == other.n_nodes && first_node_name == other.first_node_name &&  // 返回
               last_node_name == other.last_node_name;
    }
};

struct graph_key_hash {  // 结构体定义
    size_t operator()(const graph_key & key) const {
        size_t h = std::hash<int>{}(key.n_nodes);
        if (key.n_nodes > 0) {
            h ^= std::hash<std::string>{}(key.first_node_name) + 0x9e3779b9 + (h << 6) + (h >> 2);
            h ^= std::hash<std::string>{}(key.last_node_name) + 0x9e3779b9 + (h << 6) + (h >> 2);
        }
        return h;  // 返回
    }
};

struct decoder_runtime_ctx {  // 结构体定义
    decoder_runtime_ctx(std::shared_ptr<std::mutex> mutex) : mutex(std::move(mutex)) {}
    std::shared_ptr<std::mutex> mutex;
    std::shared_ptr<GgmlOvDecoder> ptr;
};

struct ov_runtime_context {  // 结构体定义
    mutable std::mutex ctx_mutex;
    std::string device;
    bool stateful;
    std::unordered_map<graph_key, std::shared_ptr<decoder_runtime_ctx>, graph_key_hash> decoder_cache;
    std::unordered_map<graph_key, std::shared_ptr<ov::InferRequest>, graph_key_hash> infer_request_cache;
    std::unordered_map<graph_key, std::shared_ptr<ov::InferRequest>, graph_key_hash> infer_request_cache_prefill;
    std::unordered_map<graph_key, std::vector<std::string>, graph_key_hash> ov_input_names_cache;
    std::unordered_map<graph_key, std::vector<std::string>, graph_key_hash> ov_output_names_cache;
    //TODO: Stateful is only supported for single request at a time.
    //      Simultanous stateful inference request support to be added.
    size_t stateful_kv_size;
    std::map<std::string, std::string> kv_state_input_name_map;
    std::atomic<int> backend_count;

    ov_runtime_context() :
        device("CPU"),
        stateful(false),
        stateful_kv_size(0),
        backend_count(0) {}

    void clear_caches() {
        std::lock_guard<std::mutex> lock(ctx_mutex);
        decoder_cache.clear();
        infer_request_cache.clear();
        infer_request_cache_prefill.clear();
        ov_input_names_cache.clear();
        ov_output_names_cache.clear();
    }
};

enum ggml_status ov_graph_compute(struct ggml_cgraph * cgraph, ggml_backend_t backend);  // ov_graph_compute

enum ggml_status ov_graph_compute_dynamic(struct ggml_cgraph * cgraph, std::shared_ptr<ov_runtime_context> r_ctx);  // ov_graph_compute_dynamic
enum ggml_status ov_graph_compute_static(struct ggml_cgraph * cgraph, std::shared_ptr<ov_runtime_context> r_ctx);  // ov_graph_compute_static

size_t checksum(const void * data, size_t size);  // checksum

void print_input_tensor_info(const std::string & name, const ov::Tensor & tensor);  // print_input_tensor_info

void print_output_tensor_info(const std::string & name, const ov::Tensor & tensor, const void * output_dst);  // print_output_tensor_info

template <typename T>  // 模板
std::vector<T> pad_input(const T * data,
                         size_t rows,
                         size_t cols,
                         size_t padded_rows,
                         size_t padded_cols,
                         T pad_value) {
    std::vector<T> padded(padded_rows * padded_cols, pad_value);

    for (size_t i = 0; i < std::min(rows, padded_rows); ++i) {
        for (size_t j = 0; j < std::min(cols, padded_cols); ++j) {
            padded[i * padded_cols + j] = data[i * cols + j];
        }
    }

    return padded;  // 返回
}

template <typename T>  // 模板
std::vector<T> pad_input(const ggml_tensor * tensor, size_t padded_rows, size_t padded_cols, T pad_value) {
    return pad_input<T>(reinterpret_cast<const T *>(tensor->data),
                        static_cast<size_t>(tensor->ne[1]),  // rows
                        static_cast<size_t>(tensor->ne[0]),  // cols
                        padded_rows, padded_cols, pad_value);
}

void set_zero_diagonal(std::vector<float> & matrix, size_t rows, size_t cols);  // set_zero_diagonal

const ggml_tensor * get_inp_pos_tensor(struct ggml_cgraph * cgraph);  // get_inp_pos_tensor

bool get_is_prefill(const ggml_tensor * inp_pos);  // get_is_prefill

ov::Tensor get_ov_input_tensor(std::shared_ptr<GgmlOvDecoder> ggml_decoder, const std::string & param_name);
ov::Tensor get_ov_input_tensor_static_decode(std::shared_ptr<GgmlOvDecoder> ggml_decoder,
                                             const std::string & param_name);
ov::Tensor get_ov_input_tensor_static_prefill(std::shared_ptr<GgmlOvDecoder> ggml_decoder,
                                              const std::string & param_name,
                                              int chunk_index);

ov::Tensor create_ov_output_tensor(std::shared_ptr<GgmlOvDecoder> ggml_decoder,
                                   std::shared_ptr<ov::InferRequest> infer_request,
                                   int output_index,
                                   const ggml_tensor * ggml_tensor);

bool is_naive(struct ggml_cgraph * cgraph);  // is_naive

enum ggml_status naive_compute(struct ggml_cgraph * cgraph,
                               ov::Core & core,
                               const std::string & device,
                               const ov::AnyMap & config);
