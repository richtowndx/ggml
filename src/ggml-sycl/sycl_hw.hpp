#ifndef SYCL_HW_HPP  // 如果未定义 SYCL_HW_HPP 则编译
#define SYCL_HW_HPP  // 宏定义 SYCL_HW_HPP

#include <algorithm>  // 引入 algorithm 头文件
#include <stdio.h>  // 引入 stdio.h 头文件
#include <vector>  // 引入 vector 头文件
#include <map>  // 引入 map 头文件

#include <sycl/sycl.hpp>  // 引入 sycl/sycl.hpp 头文件

namespace syclex = sycl::ext::oneapi::experimental;  // 命名空间
using gpu_arch = sycl::ext::oneapi::experimental::architecture;  // using 声明

// It's used to mark the GPU computing capacity
// The value must flow the order of performance.
enum sycl_intel_gpu_family {  // 枚举定义
  GPU_FAMILY_UKNOWN = -1,
  // iGPU without Xe core, before Meteor Lake iGPU(Xe)
  GPU_FAMILY_IGPU_NON_XE = 0,
  // iGPU with Xe core, Meteor Lake iGPU or newer.
  GPU_FAMILY_IGPU_XE = 1,
  // dGPU for gaming in client/data center (DG1/FLex 140 or newer).
  GPU_FAMILY_DGPU_CLIENT_GAME = 2,
  // dGPU for AI in cloud, PVC or newer.
  GPU_FAMILY_DGPU_CLOUD = 3
};

struct sycl_hw_info {  // 结构体定义
  syclex::architecture arch;
  const char* arch_name;
  int32_t device_id;
  std::string name;
  sycl_intel_gpu_family gpu_family;
};

sycl_hw_info get_device_hw_info(sycl::device *device_ptr);  // get_device_hw_info

#endif // SYCL_HW_HPP  // 条件编译结束
