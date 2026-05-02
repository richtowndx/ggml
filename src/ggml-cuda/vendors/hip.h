#pragma once  // 防止重复包含

#define HIP_DISABLE_WARP_SYNC_BUILTINS 1  // 宏定义 HIP_DISABLE_WARP_SYNC_BUILTINS
#include <hip/hip_runtime.h>  // 引入 hip/hip_runtime.h 头文件
#include <hipblas/hipblas.h>  // 引入 hipblas/hipblas.h 头文件
#include <hip/hip_fp16.h>  // 引入 hip/hip_fp16.h 头文件
#include <hip/hip_bf16.h>  // 引入 hip/hip_bf16.h 头文件

#if defined(GGML_HIP_ROCWMMA_FATTN)  // 条件编译
#include <rocwmma/rocwmma-version.hpp>  // 引入 rocwmma/rocwmma-version.hpp 头文件
#endif // defined(GGML_HIP_ROCWMMA_FATTN)  // 条件编译结束

#ifdef GGML_USE_NCCL  // 如果定义了 GGML_USE_NCCL 则编译
#include <rccl/rccl.h>  // 引入 rccl/rccl.h 头文件
#endif // GGML_USE_NCCL  // 条件编译结束


#define CUBLAS_GEMM_DEFAULT HIPBLAS_GEMM_DEFAULT  // 宏定义 CUBLAS_GEMM_DEFAULT
#define CUBLAS_GEMM_DEFAULT_TENSOR_OP HIPBLAS_GEMM_DEFAULT  // 宏定义 CUBLAS_GEMM_DEFAULT_TENSOR_OP
#define CUBLAS_OP_N HIPBLAS_OP_N  // 宏定义 CUBLAS_OP_N
#define CUBLAS_OP_T HIPBLAS_OP_T  // 宏定义 CUBLAS_OP_T
#define CUBLAS_STATUS_SUCCESS HIPBLAS_STATUS_SUCCESS  // 宏定义 CUBLAS_STATUS_SUCCESS
#define CUBLAS_TF32_TENSOR_OP_MATH 0  // 宏定义 CUBLAS_TF32_TENSOR_OP_MATH
#define CUDA_R_16F  HIPBLAS_R_16F  // 宏定义 CUDA_R_16F
#define CUDA_R_16BF HIPBLAS_R_16B  // 宏定义 CUDA_R_16BF
#define CUDA_R_32F  HIPBLAS_R_32F  // 宏定义 CUDA_R_32F
#define CUBLAS_SIDE_RIGHT HIPBLAS_SIDE_RIGHT  // 宏定义 CUBLAS_SIDE_RIGHT
#define CUBLAS_FILL_MODE_UPPER HIPBLAS_FILL_MODE_UPPER  // 宏定义 CUBLAS_FILL_MODE_UPPER
#define CUBLAS_DIAG_NON_UNIT HIPBLAS_DIAG_NON_UNIT  // 宏定义 CUBLAS_DIAG_NON_UNIT
#define CU_DEVICE_ATTRIBUTE_VIRTUAL_MEMORY_MANAGEMENT_SUPPORTED hipDeviceAttributeVirtualMemoryManagementSupported  // 宏定义 CU_DEVICE_ATTRIBUTE_VIRTUAL_MEMORY_MANAGEMENT_SUPPORTED
#define CU_MEM_ALLOC_GRANULARITY_RECOMMENDED hipMemAllocationGranularityRecommended  // 宏定义 CU_MEM_ALLOC_GRANULARITY_RECOMMENDED
#define CU_MEM_ALLOCATION_TYPE_PINNED hipMemAllocationTypePinned  // 宏定义 CU_MEM_ALLOCATION_TYPE_PINNED
#define CU_MEM_LOCATION_TYPE_DEVICE hipMemLocationTypeDevice  // 宏定义 CU_MEM_LOCATION_TYPE_DEVICE
#define CU_MEM_ACCESS_FLAGS_PROT_READWRITE hipMemAccessFlagsProtReadWrite  // 宏定义 CU_MEM_ACCESS_FLAGS_PROT_READWRITE
#define CU_CHECK(fn) {hipError_t err = fn; if(err != hipSuccess) { GGML_ABORT("HipVMM Failure: %s\n", hipGetErrorString(err)); }}  // 宏定义 CU_CHECK
#define __shfl_sync(mask, var, laneMask, width) __shfl(var, laneMask, width)  // 宏定义 __shfl_sync
#define __shfl_up_sync(mask, var, laneMask, width) __shfl_up(var, laneMask, width)  // 宏定义 __shfl_up_sync
#define __shfl_xor_sync(mask, var, laneMask, width) __shfl_xor(var, laneMask, width)  // 宏定义 __shfl_xor_sync
#define __all_sync(mask, var) __all(var)  // 宏定义 __all_sync
#define __any_sync(mask, var) __any(var)  // 宏定义 __any_sync
#define cublasStrsmBatched hipblasStrsmBatched  // 宏定义 cublasStrsmBatched
#define cublasCreate hipblasCreate  // 宏定义 cublasCreate
#define cublasDestroy hipblasDestroy  // 宏定义 cublasDestroy
#define cublasGemmEx hipblasGemmEx  // 宏定义 cublasGemmEx
#define cublasGemmBatchedEx hipblasGemmBatchedEx  // 宏定义 cublasGemmBatchedEx
#define cublasGemmStridedBatchedEx hipblasGemmStridedBatchedEx  // 宏定义 cublasGemmStridedBatchedEx
#define cublasHandle_t hipblasHandle_t  // 宏定义 cublasHandle_t
#define cublasSetMathMode(handle, mode) CUBLAS_STATUS_SUCCESS  // 宏定义 cublasSetMathMode
#define cublasSetStream hipblasSetStream  // 宏定义 cublasSetStream
#define cublasSgemm hipblasSgemm  // 宏定义 cublasSgemm
#define cublasStatus_t hipblasStatus_t  // 宏定义 cublasStatus_t
#define cublasOperation_t hipblasOperation_t  // 宏定义 cublasOperation_t
#define cudaDevAttrCooperativeLaunch hipDeviceAttributeCooperativeLaunch  // 宏定义 cudaDevAttrCooperativeLaunch
#define cudaDeviceCanAccessPeer hipDeviceCanAccessPeer  // 宏定义 cudaDeviceCanAccessPeer
#define cudaDeviceDisablePeerAccess hipDeviceDisablePeerAccess  // 宏定义 cudaDeviceDisablePeerAccess
#define cudaDeviceEnablePeerAccess hipDeviceEnablePeerAccess  // 宏定义 cudaDeviceEnablePeerAccess
#define cudaDeviceGetAttribute hipDeviceGetAttribute  // 宏定义 cudaDeviceGetAttribute
#define cudaDeviceProp hipDeviceProp_t  // 宏定义 cudaDeviceProp
#define cudaDeviceSynchronize hipDeviceSynchronize  // 宏定义 cudaDeviceSynchronize
#define cudaError_t hipError_t  // 宏定义 cudaError_t
#define cudaErrorMemoryAllocation hipErrorOutOfMemory  // 宏定义 cudaErrorMemoryAllocation
#define cudaErrorPeerAccessAlreadyEnabled hipErrorPeerAccessAlreadyEnabled  // 宏定义 cudaErrorPeerAccessAlreadyEnabled
#define cudaErrorPeerAccessNotEnabled hipErrorPeerAccessNotEnabled  // 宏定义 cudaErrorPeerAccessNotEnabled
#define cudaEventCreateWithFlags hipEventCreateWithFlags  // 宏定义 cudaEventCreateWithFlags
#define cudaEventDisableTiming hipEventDisableTiming  // 宏定义 cudaEventDisableTiming
#define cudaEventRecord hipEventRecord  // 宏定义 cudaEventRecord
#define cudaEventSynchronize hipEventSynchronize  // 宏定义 cudaEventSynchronize
#define cudaEvent_t hipEvent_t  // 宏定义 cudaEvent_t
#define cudaEventDestroy hipEventDestroy  // 宏定义 cudaEventDestroy
#define cudaFree hipFree  // 宏定义 cudaFree
#define cudaFreeHost hipHostFree  // 宏定义 cudaFreeHost
#define cudaGetDevice hipGetDevice  // 宏定义 cudaGetDevice
#define cudaGetDeviceCount hipGetDeviceCount  // 宏定义 cudaGetDeviceCount
#define cudaGetDeviceProperties hipGetDeviceProperties  // 宏定义 cudaGetDeviceProperties
#define cudaGetErrorString hipGetErrorString  // 宏定义 cudaGetErrorString
#define cudaGetLastError hipGetLastError  // 宏定义 cudaGetLastError
#define cudaHostRegister hipHostRegister  // 宏定义 cudaHostRegister
#define cudaHostRegisterPortable hipHostRegisterPortable  // 宏定义 cudaHostRegisterPortable
#define cudaHostRegisterReadOnly hipHostRegisterReadOnly  // 宏定义 cudaHostRegisterReadOnly
#define cudaHostUnregister hipHostUnregister  // 宏定义 cudaHostUnregister
#define cudaLaunchCooperativeKernel hipLaunchCooperativeKernel  // 宏定义 cudaLaunchCooperativeKernel
#define cudaLaunchHostFunc hipLaunchHostFunc  // 宏定义 cudaLaunchHostFunc
#define cudaMalloc hipMalloc  // 宏定义 cudaMalloc
#define cudaMallocHost(ptr, size) hipHostMalloc(ptr, size, hipHostMallocDefault)  // 宏定义 cudaMallocHost
#define cudaMallocManaged hipMallocManaged  // 宏定义 cudaMallocManaged
#define cudaMemAdvise hipMemAdvise  // 宏定义 cudaMemAdvise
#define cudaMemcpy hipMemcpy  // 宏定义 cudaMemcpy
#define cudaMemcpyAsync hipMemcpyAsync  // 宏定义 cudaMemcpyAsync
#define cudaMemcpyPeerAsync hipMemcpyPeerAsync  // 宏定义 cudaMemcpyPeerAsync
#define cudaMemcpy2DAsync hipMemcpy2DAsync  // 宏定义 cudaMemcpy2DAsync
#define cudaMemcpyDeviceToDevice hipMemcpyDeviceToDevice  // 宏定义 cudaMemcpyDeviceToDevice
#define cudaMemcpyDeviceToHost hipMemcpyDeviceToHost  // 宏定义 cudaMemcpyDeviceToHost
#define cudaMemcpyHostToDevice hipMemcpyHostToDevice  // 宏定义 cudaMemcpyHostToDevice
#define cudaMemcpyKind hipMemcpyKind  // 宏定义 cudaMemcpyKind
#define cudaMemset hipMemset  // 宏定义 cudaMemset
#define cudaMemsetAsync hipMemsetAsync  // 宏定义 cudaMemsetAsync
#define cudaMemGetInfo hipMemGetInfo  // 宏定义 cudaMemGetInfo
#define cudaOccupancyMaxPotentialBlockSize hipOccupancyMaxPotentialBlockSize  // 宏定义 cudaOccupancyMaxPotentialBlockSize
#define cudaSetDevice hipSetDevice  // 宏定义 cudaSetDevice
#define cuDeviceGet hipDeviceGet  // 宏定义 cuDeviceGet
#define CUdevice hipDevice_t  // 宏定义 CUdevice
#define CUdeviceptr hipDeviceptr_t  // 宏定义 CUdeviceptr
#define cuMemUnmap hipMemUnmap  // 宏定义 cuMemUnmap
#define CUmemAccessDesc hipMemAccessDesc  // 宏定义 CUmemAccessDesc
#define cuMemAddressFree hipMemAddressFree  // 宏定义 cuMemAddressFree
#define cuMemRelease hipMemRelease  // 宏定义 cuMemRelease
#define CUmemGenericAllocationHandle hipMemGenericAllocationHandle_t  // 宏定义 CUmemGenericAllocationHandle
#define cuMemCreate hipMemCreate  // 宏定义 cuMemCreate
#define cuMemAddressReserve hipMemAddressReserve  // 宏定义 cuMemAddressReserve
#define cuMemMap hipMemMap  // 宏定义 cuMemMap
#define cuMemSetAccess hipMemSetAccess  // 宏定义 cuMemSetAccess
#define cuMemGetAllocationGranularity hipMemGetAllocationGranularity  // 宏定义 cuMemGetAllocationGranularity
#define CUmemAllocationProp hipMemAllocationProp  // 宏定义 CUmemAllocationProp
#define cuDeviceGetAttribute hipDeviceGetAttribute  // 宏定义 cuDeviceGetAttribute
#define cudaStreamCreateWithFlags hipStreamCreateWithFlags  // 宏定义 cudaStreamCreateWithFlags
#define cudaStreamDestroy hipStreamDestroy  // 宏定义 cudaStreamDestroy
#define cudaStreamFireAndForget hipStreamFireAndForget  // 宏定义 cudaStreamFireAndForget
#define cudaStreamNonBlocking hipStreamNonBlocking  // 宏定义 cudaStreamNonBlocking
#define cudaStreamPerThread hipStreamPerThread  // 宏定义 cudaStreamPerThread
#define cudaStreamSynchronize hipStreamSynchronize  // 宏定义 cudaStreamSynchronize
#define cudaStreamWaitEvent hipStreamWaitEvent  // 宏定义 cudaStreamWaitEvent
#define cudaGraphExec_t hipGraphExec_t  // 宏定义 cudaGraphExec_t
#define cudaGraphNode_t hipGraphNode_t  // 宏定义 cudaGraphNode_t
#define cudaKernelNodeParams hipKernelNodeParams  // 宏定义 cudaKernelNodeParams
#define cudaKernelNodeParams hipKernelNodeParams  // 宏定义 cudaKernelNodeParams
#define cudaGraphExecDestroy hipGraphExecDestroy  // 宏定义 cudaGraphExecDestroy
#define cudaGraphLaunch hipGraphLaunch  // 宏定义 cudaGraphLaunch
#define cudaErrorGraphExecUpdateFailure hipErrorGraphExecUpdateFailure  // 宏定义 cudaErrorGraphExecUpdateFailure
#define cudaGraphExecUpdateResult hipGraphExecUpdateResult  // 宏定义 cudaGraphExecUpdateResult
#define cudaGraphNodeType hipGraphNodeType  // 宏定义 cudaGraphNodeType
#define cudaGraphNodeTypeKernel hipGraphNodeTypeKernel  // 宏定义 cudaGraphNodeTypeKernel
#define cudaGraphInstantiate hipGraphInstantiate  // 宏定义 cudaGraphInstantiate
#define cudaStreamEndCapture hipStreamEndCapture  // 宏定义 cudaStreamEndCapture
#define cudaGraphDestroy hipGraphDestroy  // 宏定义 cudaGraphDestroy
#define cudaGraphKernelNodeSetParams hipGraphKernelNodeSetParams  // 宏定义 cudaGraphKernelNodeSetParams
#define cudaErrorInvalidDeviceFunction hipErrorInvalidDeviceFunction  // 宏定义 cudaErrorInvalidDeviceFunction
#define cudaGraphKernelNodeGetParams hipGraphKernelNodeGetParams  // 宏定义 cudaGraphKernelNodeGetParams
#define cudaGraphNodeGetType hipGraphNodeGetType  // 宏定义 cudaGraphNodeGetType
#define cudaGraphGetNodes hipGraphGetNodes  // 宏定义 cudaGraphGetNodes
#define cudaGraphExecUpdate hipGraphExecUpdate  // 宏定义 cudaGraphExecUpdate
#define cudaStreamCaptureModeRelaxed hipStreamCaptureModeRelaxed  // 宏定义 cudaStreamCaptureModeRelaxed
#define cudaStreamBeginCapture hipStreamBeginCapture  // 宏定义 cudaStreamBeginCapture
#define cudaGraph_t hipGraph_t  // 宏定义 cudaGraph_t
#define cudaStream_t hipStream_t  // 宏定义 cudaStream_t
#define cudaSuccess hipSuccess  // 宏定义 cudaSuccess
#define cudaOccupancyMaxActiveBlocksPerMultiprocessor hipOccupancyMaxActiveBlocksPerMultiprocessor  // 宏定义 cudaOccupancyMaxActiveBlocksPerMultiprocessor
#define cudaFuncSetAttribute hipFuncSetAttribute  // 宏定义 cudaFuncSetAttribute
#define cudaFuncAttributeMaxDynamicSharedMemorySize hipFuncAttributeMaxDynamicSharedMemorySize  // 宏定义 cudaFuncAttributeMaxDynamicSharedMemorySize
#define __trap() do { abort(); __builtin_unreachable(); } while(0)  // 宏定义 __trap
#define CUBLAS_STATUS_SUCCESS HIPBLAS_STATUS_SUCCESS  // 宏定义 CUBLAS_STATUS_SUCCESS
#define CUBLAS_STATUS_NOT_INITIALIZED HIPBLAS_STATUS_NOT_INITIALIZED  // 宏定义 CUBLAS_STATUS_NOT_INITIALIZED
#define CUBLAS_STATUS_ALLOC_FAILED HIPBLAS_STATUS_ALLOC_FAILED  // 宏定义 CUBLAS_STATUS_ALLOC_FAILED
#define CUBLAS_STATUS_INVALID_VALUE HIPBLAS_STATUS_INVALID_VALUE  // 宏定义 CUBLAS_STATUS_INVALID_VALUE
#define CUBLAS_STATUS_ARCH_MISMATCH HIPBLAS_STATUS_ARCH_MISMATCH  // 宏定义 CUBLAS_STATUS_ARCH_MISMATCH
#define CUBLAS_STATUS_MAPPING_ERROR HIPBLAS_STATUS_MAPPING_ERROR  // 宏定义 CUBLAS_STATUS_MAPPING_ERROR
#define CUBLAS_STATUS_EXECUTION_FAILED HIPBLAS_STATUS_EXECUTION_FAILED  // 宏定义 CUBLAS_STATUS_EXECUTION_FAILED
#define CUBLAS_STATUS_INTERNAL_ERROR HIPBLAS_STATUS_INTERNAL_ERROR  // 宏定义 CUBLAS_STATUS_INTERNAL_ERROR
#define CUBLAS_STATUS_NOT_SUPPORTED HIPBLAS_STATUS_NOT_SUPPORTED  // 宏定义 CUBLAS_STATUS_NOT_SUPPORTED

#if HIP_VERSION >= 60500000  // 条件编译
#define CUBLAS_COMPUTE_16F HIPBLAS_COMPUTE_16F  // 宏定义 CUBLAS_COMPUTE_16F
#define CUBLAS_COMPUTE_32F HIPBLAS_COMPUTE_32F  // 宏定义 CUBLAS_COMPUTE_32F
#define CUBLAS_COMPUTE_32F_FAST_16F HIPBLAS_COMPUTE_32F_FAST_16F  // 宏定义 CUBLAS_COMPUTE_32F_FAST_16F
#define cublasComputeType_t hipblasComputeType_t  // 宏定义 cublasComputeType_t
#define cudaDataType_t hipDataType  // 宏定义 cudaDataType_t
#else  // 否则
#define CUBLAS_COMPUTE_16F HIPBLAS_R_16F  // 宏定义 CUBLAS_COMPUTE_16F
#define CUBLAS_COMPUTE_32F HIPBLAS_R_32F  // 宏定义 CUBLAS_COMPUTE_32F
#define CUBLAS_COMPUTE_32F_FAST_16F HIPBLAS_R_32F  // 宏定义 CUBLAS_COMPUTE_32F_FAST_16F
#define cublasComputeType_t hipblasDatatype_t  // 宏定义 cublasComputeType_t
#define cudaDataType_t hipblasDatatype_t  // 宏定义 cudaDataType_t
#endif // HIP_VERSION >= 6050000  // 条件编译结束

#if !defined(__HIP_PLATFORM_AMD__)  // 条件编译
#error "The HIP backend supports only AMD targets"
#endif // !defined(__HIP_PLATFORM_AMD__)  // 条件编译结束

#define __CUDA_ARCH__ 1300  // 宏定义 __CUDA_ARCH__

#if defined(__gfx900__) || defined(__gfx906__)  // 条件编译
#define GCN5  // 宏定义 GCN5
#endif // defined(__gfx900__) || defined(__gfx906__)  // 条件编译结束

#if defined(__gfx803__)  // 条件编译
#define GCN4  // 宏定义 GCN4
#endif // defined(__gfx803__)  // 条件编译结束

#if defined(GCN5) || defined(GCN4)  // 条件编译
#define GCN  // 宏定义 GCN
#endif // defined(GCN5) || defined(GCN4)  // 条件编译结束

#if defined(__gfx950__)  // 条件编译
#define CDNA4  // 宏定义 CDNA4
#endif // defined(__gfx950__)  // 条件编译结束

#if defined(__gfx942__)  // 条件编译
#define CDNA3  // 宏定义 CDNA3
#endif // defined(__gfx942__)  // 条件编译结束

#if defined(__gfx90a__)  // 条件编译
#define CDNA2  // 宏定义 CDNA2
#endif // defined(__gfx90a__)  // 条件编译结束

#if defined(__gfx908__)  // 条件编译
#define CDNA1  // 宏定义 CDNA1
#endif // defined(__gfx908__)  // 条件编译结束

#if defined(CDNA4) || defined(CDNA3) || defined(CDNA2) || defined(CDNA1)  // 条件编译
#define CDNA // For the entire family  // 宏定义 CDNA
#endif // defined(CDNA4) || defined(CDNA3) || defined(CDNA2) || defined(CDNA1)  // 条件编译结束

#if defined(__GFX12__)  // 条件编译
#define RDNA4  // 宏定义 RDNA4
#endif // defined(__GFX12__)  // 条件编译结束

#if defined(__GFX11__)  // 条件编译
#define RDNA3  // 宏定义 RDNA3
#endif // defined(__GFX11__)  // 条件编译结束

#if defined(__gfx1150__) || defined(__gfx1151__)  // 条件编译
#define RDNA3_5  // 宏定义 RDNA3_5
#endif // defined(__gfx1150__) || defined(__gfx1151__)  // 条件编译结束

#if defined(RDNA3) && !defined(RDNA3_5)  // 条件编译
#define RDNA3_0  // 宏定义 RDNA3_0
#endif // defined(RDNA3) && !defined(RDNA3_5)  // 条件编译结束

#if defined(__gfx1030__) || defined(__gfx1031__) || defined(__gfx1032__) || defined(__gfx1033__) || \
    defined(__gfx1034__) || defined(__gfx1035__) || defined(__gfx1036__) || defined(__gfx1037__)
#define RDNA2  // 宏定义 RDNA2
#endif  // 条件编译结束

#if defined(__gfx1010__) || defined(__gfx1012__)  // 条件编译
#define RDNA1  // 宏定义 RDNA1
#endif // defined(__gfx1010__) || defined(__gfx1012__)  // 条件编译结束

#if defined(RDNA4) || defined(RDNA3) || defined(RDNA2) || defined(RDNA1)  // 条件编译
#define RDNA // For the entire family  // 宏定义 RDNA
#endif // defined(RDNA4) || defined(RDNA3) || defined(RDNA2) || defined(RDNA1)  // 条件编译结束

#ifndef __has_builtin  // 如果未定义 __has_builtin 则编译
    #define __has_builtin(x) 0  // 宏定义 __has_builtin
#endif  // 条件编译结束

typedef __hip_bfloat16 nv_bfloat16;  // 类型定义
typedef __hip_bfloat162 nv_bfloat162;  // 类型定义

#if HIP_VERSION >= 60200000  // 条件编译
#include <hip/hip_fp8.h>  // 引入 hip/hip_fp8.h 头文件
typedef __hip_fp8_e4m3 __nv_fp8_e4m3;  // 类型定义
#define FP8_AVAILABLE  // 宏定义 FP8_AVAILABLE
#endif // HIP_VERSION >= 60200000  // 条件编译结束

typedef int8_t int8x4_t __attribute__((ext_vector_type(4)));  // 类型定义
typedef uint8_t uint8x4_t __attribute__((ext_vector_type(4)));  // 类型定义
static __device__ __forceinline__ int __vsubss4(const int a, const int b) {
    const int8x4_t va = reinterpret_cast<const int8x4_t&>(a);
    const int8x4_t vb = reinterpret_cast<const int8x4_t&>(b);
#if __has_builtin(__builtin_elementwise_sub_sat)  // 条件编译
    const int8x4_t c = __builtin_elementwise_sub_sat(va, vb);
    return reinterpret_cast<const int &>(c);
#else  // 否则
    int8x4_t c;
    int16_t tmp;
#pragma unroll
    for (int i = 0; i < 4; i++) {
        tmp = va[i] - vb[i];
        if(tmp > std::numeric_limits<int8_t>::max()) tmp = std::numeric_limits<int8_t>::max();
        if(tmp < std::numeric_limits<int8_t>::min()) tmp = std::numeric_limits<int8_t>::min();
        c[i] = tmp;
    }
    return reinterpret_cast<int &>(c);
#endif // __has_builtin(__builtin_elementwise_sub_sat)  // 条件编译结束
}

static __device__ __forceinline__ int __vsub4(const int a, const int b) {
    return __vsubss4(a, b);  // __vsubss4
}

static __device__ __forceinline__ unsigned int __vcmpeq4(unsigned int a, unsigned int b) {
    const uint8x4_t& va = reinterpret_cast<const uint8x4_t&>(a);
    const uint8x4_t& vb = reinterpret_cast<const uint8x4_t&>(b);
    unsigned int c;
    uint8x4_t& vc = reinterpret_cast<uint8x4_t&>(c);
#pragma unroll
    for (int i = 0; i < 4; ++i) {
        vc[i] = va[i] == vb[i] ? 0xff : 0x00;
    }
    return c;  // 返回
}

static __device__ __forceinline__ unsigned int __vcmpne4(unsigned int a, unsigned int b) {
    const uint8x4_t& va = reinterpret_cast<const uint8x4_t&>(a);
    const uint8x4_t& vb = reinterpret_cast<const uint8x4_t&>(b);
    unsigned int c;
    uint8x4_t& vc = reinterpret_cast<uint8x4_t&>(c);
#pragma unroll
    for (int i = 0; i < 4; ++i) {
        vc[i] = va[i] == vb[i] ? 0x00 : 0xff;
    }
    return c;  // 返回
}
