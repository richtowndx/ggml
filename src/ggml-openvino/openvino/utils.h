#pragma once  // 防止重复包含

#include <memory>  // 引入 memory 头文件
#include <openvino/core/node.hpp>  // 引入 openvino/core/node.hpp 头文件
#include <openvino/op/shape_of.hpp>  // 引入 openvino/op/shape_of.hpp 头文件
#include <openvino/op/slice.hpp>  // 引入 openvino/op/slice.hpp 头文件
#include <utility>  // 引入 utility 头文件

#include "node_context.h"  // 引入 node_context.h 头文件

namespace ov {  // 命名空间
namespace frontend {  // 命名空间
namespace ggml {  // 命名空间

std::string getCurrentTime();

void dump_ov_model(std::shared_ptr<ov::Model> model);  // dump_ov_model

void num_inputs_check(const NodeContext& context, size_t min_inputs, size_t max_inputs);  // num_inputs_check

int non_cont_dim(std::vector<size_t> ne, std::vector<size_t> nb);  // non_cont_dim

template <typename T>  // 模板
std::vector<int> argsort_descend(const std::vector<T>& v) {
    std::vector<int> idx(v.size());
    std::iota(idx.begin(), idx.end(), 0);
    std::sort(idx.begin(), idx.end(), [&v](int i1, int i2) {
        return v[i1] > v[i2];  // 返回
    });
    return idx;  // 返回
}

template <typename T>  // 模板
std::vector<T> sorted_descend(std::vector<T> v) {
    std::sort(v.begin(), v.end(), [](T a, T b) {
        return a > b;  // 返回
    });
    return v;  // 返回
}

template <typename T>  // 模板
bool is_permuted(const std::vector<T>& strides) {
    for (size_t i = 0; i < strides.size() - 1; ++i) {
        if (strides[i] < strides[i + 1]) {
            return true;  // 返回
        }
    }
    return false;  // 返回
}

template <typename T>  // 模板
std::vector<T> permute(const std::vector<T>& x, const std::vector<int>& perm) {
    std::vector<T> result;
    result.reserve(perm.size());
    for (int i : perm) {
        result.push_back(x[i]);
    }
    return result;  // 返回
}

std::shared_ptr<ov::Node> get_dimensions(const std::shared_ptr<ov::op::v3::ShapeOf>& shape,
                                         const std::vector<int>& dims);
std::shared_ptr<ov::Node> get_dimensions(const std::shared_ptr<ov::Node>& node, const std::vector<int>& dims);

OutputVector rename_outputs_with_suffix(const OutputVector& outputs, const std::string& suffix);  // rename_outputs_with_suffix

std::pair<ov::Output<Node>, ov::Output<Node>> make_sin_cos(int32_t* rope_params,
                                                           std::shared_ptr<ov::Node> inp_pos,
                                                           std::shared_ptr<ov::Node> rope_freqs_weight = nullptr,
                                                           bool imrope = false,
                                                           bool stateful = false);

ov::Output<ov::Node> process_view_input(const NodeContext& context, int input_index, int slice_len = 0);

namespace op {  // 命名空间
template <typename T>  // 模板
OutputVector translate_1to1_match_2_inputs(const NodeContext& context) {
    num_inputs_check(context, 2, 2);
    auto res = std::make_shared<T>(context.get_input(0), context.get_input(1));
    return rename_outputs_with_suffix({res}, context.get_name());  // 返回
}
}  // namespace op

}  // namespace ggml
}  // namespace frontend
}  // namespace ov
