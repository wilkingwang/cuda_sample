#pragma once

#include <iostream>
#include <cusparse.h>
#include <cublas_v2.h>
#include <driver_types.h>
#include <cuda_runtime_api.h>

static const char* _cudaGetErrorEnum(cudaError_t error)
{
    switch (error) {
    case cudaSuccess:
        return "cudaSuccess";

    case cudaErrorMissingConfiguration:
        return "cudaErrorMissingConfiguration";

    case cudaErrorMemoryAllocation:
        return "cudaErrorMemoryAllocation";

    case cudaErrorInitializationError:
        return "cudaErrorInitializationError";

    case cudaErrorLaunchFailure:
        return "cudaErrorLaunchFailure";

    case cudaErrorPriorLaunchFailure:
        return "cudaErrorPriorLaunchFailure";

    case cudaErrorLaunchTimeout:
        return "cudaErrorLaunchTimeout";

    case cudaErrorLaunchOutOfResources:
        return "cudaErrorLaunchOutOfResources";

    case cudaErrorInvalidDeviceFunction:
        return "cudaErrorInvalidDeviceFunction";

    case cudaErrorInvalidConfiguration:
        return "cudaErrorInvalidConfiguration";

    case cudaErrorInvalidDevice:
        return "cudaErrorInvalidDevice";

    case cudaErrorInvalidValue:
        return "cudaErrorInvalidValue";

    case cudaErrorInvalidPitchValue:
        return "cudaErrorInvalidPitchValue";

    case cudaErrorInvalidSymbol:
        return "cudaErrorInvalidSymbol";

    case cudaErrorMapBufferObjectFailed:
        return "cudaErrorMapBufferObjectFailed";

    case cudaErrorUnmapBufferObjectFailed:
        return "cudaErrorUnmapBufferObjectFailed";

    case cudaErrorInvalidHostPointer:
        return "cudaErrorInvalidHostPointer";

    case cudaErrorInvalidDevicePointer:
        return "cudaErrorInvalidDevicePointer";

    case cudaErrorInvalidTexture:
        return "cudaErrorInvalidTexture";

    case cudaErrorInvalidTextureBinding:
        return "cudaErrorInvalidTextureBinding";

    case cudaErrorInvalidChannelDescriptor:
        return "cudaErrorInvalidChannelDescriptor";

    case cudaErrorInvalidMemcpyDirection:
        return "cudaErrorInvalidMemcpyDirection";

    case cudaErrorAddressOfConstant:
        return "cudaErrorAddressOfConstant";

    case cudaErrorTextureFetchFailed:
        return "cudaErrorTextureFetchFailed";

    case cudaErrorTextureNotBound:
        return "cudaErrorTextureNotBound";

    case cudaErrorSynchronizationError:
        return "cudaErrorSynchronizationError";

    case cudaErrorInvalidFilterSetting:
        return "cudaErrorInvalidFilterSetting";

    case cudaErrorInvalidNormSetting:
        return "cudaErrorInvalidNormSetting";

    case cudaErrorMixedDeviceExecution:
        return "cudaErrorMixedDeviceExecution";

    case cudaErrorCudartUnloading:
        return "cudaErrorCudartUnloading";

    case cudaErrorUnknown:
        return "cudaErrorUnknown";

    case cudaErrorNotYetImplemented:
        return "cudaErrorNotYetImplemented";

    case cudaErrorMemoryValueTooLarge:
        return "cudaErrorMemoryValueTooLarge";

    case cudaErrorInvalidResourceHandle:
        return "cudaErrorInvalidResourceHandle";

    case cudaErrorNotReady:
        return "cudaErrorNotReady";

    case cudaErrorInsufficientDriver:
        return "cudaErrorInsufficientDriver";

    case cudaErrorSetOnActiveProcess:
        return "cudaErrorSetOnActiveProcess";

    case cudaErrorInvalidSurface:
        return "cudaErrorInvalidSurface";

    case cudaErrorNoDevice:
        return "cudaErrorNoDevice";

    case cudaErrorECCUncorrectable:
        return "cudaErrorECCUncorrectable";

    case cudaErrorSharedObjectSymbolNotFound:
        return "cudaErrorSharedObjectSymbolNotFound";

    case cudaErrorSharedObjectInitFailed:
        return "cudaErrorSharedObjectInitFailed";

    case cudaErrorUnsupportedLimit:
        return "cudaErrorUnsupportedLimit";

    case cudaErrorDuplicateVariableName:
        return "cudaErrorDuplicateVariableName";

    case cudaErrorDuplicateTextureName:
        return "cudaErrorDuplicateTextureName";

    case cudaErrorDuplicateSurfaceName:
        return "cudaErrorDuplicateSurfaceName";

    case cudaErrorDevicesUnavailable:
        return "cudaErrorDevicesUnavailable";

    case cudaErrorInvalidKernelImage:
        return "cudaErrorInvalidKernelImage";

    case cudaErrorNoKernelImageForDevice:
        return "cudaErrorNoKernelImageForDevice";

    case cudaErrorIncompatibleDriverContext:
        return "cudaErrorIncompatibleDriverContext";

    case cudaErrorPeerAccessAlreadyEnabled:
        return "cudaErrorPeerAccessAlreadyEnabled";

    case cudaErrorPeerAccessNotEnabled:
        return "cudaErrorPeerAccessNotEnabled";

    case cudaErrorDeviceAlreadyInUse:
        return "cudaErrorDeviceAlreadyInUse";

    case cudaErrorProfilerDisabled:
        return "cudaErrorProfilerDisabled";

    case cudaErrorProfilerNotInitialized:
        return "cudaErrorProfilerNotInitialized";

    case cudaErrorProfilerAlreadyStarted:
        return "cudaErrorProfilerAlreadyStarted";

    case cudaErrorProfilerAlreadyStopped:
        return "cudaErrorProfilerAlreadyStopped";

        /* Since CUDA 4.0*/
    case cudaErrorAssert:
        return "cudaErrorAssert";

    case cudaErrorTooManyPeers:
        return "cudaErrorTooManyPeers";

    case cudaErrorHostMemoryAlreadyRegistered:
        return "cudaErrorHostMemoryAlreadyRegistered";

    case cudaErrorHostMemoryNotRegistered:
        return "cudaErrorHostMemoryNotRegistered";

        /* Since CUDA 5.0 */
    case cudaErrorOperatingSystem:
        return "cudaErrorOperatingSystem";

    case cudaErrorPeerAccessUnsupported:
        return "cudaErrorPeerAccessUnsupported";

    case cudaErrorLaunchMaxDepthExceeded:
        return "cudaErrorLaunchMaxDepthExceeded";

    case cudaErrorLaunchFileScopedTex:
        return "cudaErrorLaunchFileScopedTex";

    case cudaErrorLaunchFileScopedSurf:
        return "cudaErrorLaunchFileScopedSurf";

    case cudaErrorSyncDepthExceeded:
        return "cudaErrorSyncDepthExceeded";

    case cudaErrorLaunchPendingCountExceeded:
        return "cudaErrorLaunchPendingCountExceeded";

    case cudaErrorNotPermitted:
        return "cudaErrorNotPermitted";

    case cudaErrorNotSupported:
        return "cudaErrorNotSupported";

        /* Since CUDA 6.0 */
    case cudaErrorHardwareStackError:
        return "cudaErrorHardwareStackError";

    case cudaErrorIllegalInstruction:
        return "cudaErrorIllegalInstruction";

    case cudaErrorMisalignedAddress:
        return "cudaErrorMisalignedAddress";

    case cudaErrorInvalidAddressSpace:
        return "cudaErrorInvalidAddressSpace";

    case cudaErrorInvalidPc:
        return "cudaErrorInvalidPc";

    case cudaErrorIllegalAddress:
        return "cudaErrorIllegalAddress";

        /* Since CUDA 6.5*/
    case cudaErrorInvalidPtx:
        return "cudaErrorInvalidPtx";

    case cudaErrorInvalidGraphicsContext:
        return "cudaErrorInvalidGraphicsContext";

    case cudaErrorStartupFailure:
        return "cudaErrorStartupFailure";

    case cudaErrorApiFailureBase:
        return "cudaErrorApiFailureBase";

        /* Since CUDA 8.0*/
    case cudaErrorNvlinkUncorrectable:
        return "cudaErrorNvlinkUncorrectable";

        /* Since CUDA 8.5*/
    case cudaErrorJitCompilerNotFound:
        return "cudaErrorJitCompilerNotFound";

        /* Since CUDA 9.0*/
    case cudaErrorCooperativeLaunchTooLarge:
        return "cudaErrorCooperativeLaunchTooLarge";
    }

    return "<unknown>";
}

static const char* _cudaGetErrorEnum(cublasStatus_t error)
{
    switch (error)
    {
    case CUBLAS_STATUS_SUCCESS:
        return "CUBLAS_STATUS_SUCCESS";
    case CUBLAS_STATUS_NOT_INITIALIZED:
        return "CUBLAS_STATUS_NOT_INITIALIZED";
    case CUBLAS_STATUS_ALLOC_FAILED:
        return "CUBLAS_STATUS_ALLOC_FAILED";
    case CUBLAS_STATUS_INVALID_VALUE:
        return "CUBLAS_STATUS_INVALID_VALUE";
    case CUBLAS_STATUS_ARCH_MISMATCH:
        return "CUBLAS_STATUS_ARCH_MISMATCH";
    case CUBLAS_STATUS_MAPPING_ERROR:
        return "CUBLAS_STATUS_MAPPING_ERROR";
    case CUBLAS_STATUS_EXECUTION_FAILED:
        return "CUBLAS_STATUS_EXECUTION_FAILED";
    case CUBLAS_STATUS_INTERNAL_ERROR:
        return "CUBLAS_STATUS_INTERNAL_ERROR";
    case CUBLAS_STATUS_NOT_SUPPORTED:
        return "CUBLAS_STATUS_NOT_SUPPORTED";
    case CUBLAS_STATUS_LICENSE_ERROR:
        return "CUBLAS_STATUS_LICENSE_ERROR";
    }

    return "<UnKnown>";
}

template <typename T>
void check(T result, char const* const func, const char* const file, int const line)
{
    if (result)
    {
        fprintf(stderr, "CUDA error at %s:%d code=%d(%s) \"%s\" \n", file, line,
            static_cast<unsigned int>(result), _cudaGetErrorEnum(result), func);
        cudaDeviceReset();
        throw 0;
    }
}

#define checkCudaErrors(val) check((val), #val, __FILE__, __LINE__)

inline void CheckCudaError(const char* file, int line)
{
#if 1
    cudaDeviceSynchronize();
    auto result = cudaGetLastError();

    if (result)
    {
        fprintf(stderr, "CUDA error at %s:%d code=%d(%s)\n", file, line,
            static_cast<unsigned int>(result), _cudaGetErrorEnum(result));
        cudaDeviceReset();
        throw 0;
    }
#endif
}

#define CCE CheckCudaError(__FILE__, __LINE__)

inline int _ConvertSMVer2Cores(int major, int minor) {
    // Defines for GPU Architecture types (using the SM version to determine
    // the # of cores per SM
    typedef struct
    {
        int SM;  // 0xMm (hexidecimal notation), M = SM Major version, and m = SM minor version
        int Cores;
    } sSMtoCores;

    sSMtoCores nGpuArchCoresPerSM[] = {
        {0x30, 192},  // Kepler Generation (SM 3.0) GK10x class
        {0x32, 192},  // Kepler Generation (SM 3.2) GK10x class
        {0x35, 192},  // Kepler Generation (SM 3.5) GK11x class
        {0x37, 192},  // Kepler Generation (SM 3.7) GK21x class
        {0x50, 128},  // Maxwell Generation (SM 5.0) GM10x class
        {0x52, 128},  // Maxwell Generation (SM 5.2) GM20x class
        {0x53, 128},  // Maxwell Generation (SM 5.3) GM20x class
        {0x60, 64},   // Pascal Generation (SM 6.0) GP100 class
        {0x61, 128},  // Pascal Generation (SM 6.1) GP10x class
        {0x62, 128},  // Pascal Generation (SM 6.2) GP10x class
        {0x70, 64},   // Volta Generation (SM 7.0) GV100 class
        {0x72, 64},   // Volta Generation (SM 7.2) GV11b class
        {0x75, 64},
        {0x80, 64},
        {0x86, 128},
        {0x87, 128},
        {0x89, 128},
        {0x90, 128},
        {-1, -1} };

    int index = 0;

    while (nGpuArchCoresPerSM[index].SM != -1)
    {
        if (nGpuArchCoresPerSM[index].SM == ((major << 4) + minor))
        {
            return nGpuArchCoresPerSM[index].Cores;
        }

        index++;
    }

    printf("SM2Core for SM %d.%d is undefined. Default to use %d Cores/SM\n", major, minor, nGpuArchCoresPerSM[index - 1].Cores);
    return nGpuArchCoresPerSM[index - 1].Cores;
}