// Copyright (C) 2018-2024 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once  // 防止重复包含

#include <openvino/frontend/frontend.hpp>  // 引入 openvino/frontend/frontend.hpp 头文件

namespace ov {  // 命名空间
namespace frontend {  // 命名空间
namespace ggml {  // 命名空间

class FrontEnd {  // 类定义
public:
    using Ptr = std::shared_ptr<FrontEnd>;  // using 声明
    FrontEnd();

    static std::shared_ptr<Model> convert(const InputModel::Ptr& model, bool naive = false);
};

}  // namespace ggml
}  // namespace frontend
}  // namespace ov
