#pragma once  // 防止重复包含

// clang-format off
#include "ggml.h"  // 引入 ggml.h 头文件
#include "ggml-backend-impl.h"  // 引入 ggml-backend-impl.h 头文件

#include <unordered_map>  // 引入 unordered_map 头文件
#include <unordered_set>  // 引入 unordered_set 头文件
#include <vector>  // 引入 vector 头文件
#include <cstdint>  // 引入 cstdint 头文件
// clang-format on

// ggml_tensor is serialized into apir_rpc_tensor
struct apir_rpc_tensor {  // 结构体定义
    uint64_t id;
    uint32_t type;
    uint64_t buffer;
    uint32_t ne[GGML_MAX_DIMS];
    uint32_t nb[GGML_MAX_DIMS];
    uint32_t op;
    int32_t  op_params[GGML_MAX_OP_PARAMS / sizeof(int32_t)];
    int32_t  flags;
    uint64_t src[GGML_MAX_SRC];
    uint64_t view_src;
    uint64_t view_offs;
    uint64_t data;
    char     name[GGML_MAX_NAME];

    char padding[4];
};

/* frontend */

apir_rpc_tensor apir_serialize_tensor(const ggml_tensor * tensor);  // apir_serialize_tensor

void apir_serialize_graph(const ggml_cgraph * cgraph, std::vector<uint8_t> & output);  // apir_serialize_graph

/* backend */

void                                      apir_track_backend_buffer(ggml_backend_buffer_t buffer);  // apir_track_backend_buffer
bool                                      apir_untrack_backend_buffer(ggml_backend_buffer_t buffer);  // apir_untrack_backend_buffer
std::unordered_set<ggml_backend_buffer_t> apir_get_track_backend_buffers();

void apir_add_tensor(ggml_tensor *                       tensor,
                     std::vector<apir_rpc_tensor> &      tensors,
                     std::unordered_set<ggml_tensor *> & visited);

ggml_tensor * apir_deserialize_tensor(ggml_context * ctx, const apir_rpc_tensor * tensor);  // apir_deserialize_tensor

ggml_tensor * apir_create_node(uint64_t                                                      id,
                               ggml_context *                                                ctx,
                               const std::unordered_map<uint64_t, const apir_rpc_tensor *> & tensor_ptrs,
                               std::unordered_map<uint64_t, ggml_tensor *> &                 tensor_map);

ggml_cgraph * apir_deserialize_graph(uint32_t                n_nodes,
                                     uint32_t                n_tensors,
                                     const apir_rpc_tensor * tensors,
                                     const uint64_t *        nodes);
