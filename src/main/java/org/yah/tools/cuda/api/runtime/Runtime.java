package org.yah.tools.cuda.api.runtime;

import com.sun.jna.Library;
import com.sun.jna.Pointer;
import com.sun.jna.ptr.PointerByReference;
import org.yah.tools.cuda.api.size_t;

public interface Runtime extends Library {

    cudaError cudaRuntimeGetVersion(Pointer runtimeVersion);

    cudaError cudaDriverGetVersion(Pointer driverVersion);

    cudaError cudaGetDevice(Pointer device);

    cudaError cudaSetDevice(int device);

    cudaError cudaGetDeviceCount(Pointer count);

    cudaError cudaGetDeviceProperties(Pointer prop, int device);

    cudaError cudaDeviceReset();

    // device management https://docs.nvidia.com/cuda/archive/11.7.0/cuda-runtime-api/group__CUDART__DEVICE.html#group__CUDART__DEVICE
    cudaError cudaDeviceSynchronize();

    cudaError cudaGetLastError();

    Pointer cudaGetErrorString(cudaError error);

    // memory management https://docs.nvidia.com/cuda/archive/11.7.0/cuda-runtime-api/group__CUDART__MEMORY.html#group__CUDART__MEMORY
    cudaError cudaMalloc(PointerByReference devPtr, long size);

    cudaError cudaFree(Pointer devPtr);

    cudaError cudaMemcpy(Pointer dst, Pointer src, long count, cudaMemcpyKind kind);

    cudaError cudaMemset(Pointer devPtr, int value, long count);

    cudaError cudaMemGetInfo(PointerByReference free, PointerByReference total);

    cudaError cudaHostRegister(PointerByReference ptr, long size, int flags);

    cudaError cudaHostUnregister(Pointer ptr);

    cudaError cudaMallocHost(PointerByReference ptr, long size);

    cudaError cudaHostGetDevicePointer(PointerByReference pDevice, Pointer pHost, int flags);

    cudaError cudaLaunchCooperativeKernel(Pointer func, dim3 gridDim, dim3 blockDim, Pointer[] args, size_t sharedMem, cudaStream stream);

    cudaError cudaLaunchKernel(Pointer func, dim3 gridDim, dim3 blockDim, Pointer[] args, size_t sharedMem, cudaStream stream);

    cudaError cudaStreamCreate(cudaStream.ByReference pStream);

    cudaError cudaStreamDestroy(cudaStream stream);

    cudaError cudaGetFuncBySymbol(PointerByReference functionPtr, Pointer symbolPtr);
}
