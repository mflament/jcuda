package org.yah.tools.cuda.api.driver;

import com.sun.jna.Library;
import com.sun.jna.Memory;
import com.sun.jna.Pointer;
import com.sun.jna.ptr.PointerByReference;
import org.yah.tools.cuda.api.NativeEnum;
import org.yah.tools.cuda.api.size_t;
import org.yah.tools.cuda.support.NativeSupport;

import java.io.IOException;
import java.nio.IntBuffer;
import java.nio.file.Path;
import java.util.Collection;
import java.util.Set;

import static org.yah.tools.cuda.support.DriverSupport.check;
import static org.yah.tools.cuda.support.DriverSupport.driverAPI;
import static org.yah.tools.cuda.support.NativeSupport.readNTS;

public interface Driver extends Library {

    class CUdevice extends Pointer {
        public CUdevice(long peer) {
            super(peer);
        }

        public CUdevice(Pointer pointer) {
            super(Pointer.nativeValue(pointer));
        }

        private static final int MAX_NAME_SIZE = 512;

        public String getDeviceName() {
            try (Memory memory = new Memory(MAX_NAME_SIZE)) {
                check(driverAPI().cuDeviceGetName(memory, MAX_NAME_SIZE, this));
                return readNTS(memory, MAX_NAME_SIZE);
            }
        }

        public int[] getDeviceAttributes(CUdevice_attribute... attributes) {
            int[] values = new int[attributes.length];
            try (Memory memory = new Memory(Integer.BYTES)) {
                for (int i = 0; i < attributes.length; i++) {
                    getDeviceAttribute(memory, attributes[i]);
                    values[i] = memory.getInt(0);
                }
                return values;
            }
        }

        public int getDeviceAttribute(CUdevice_attribute attribute) {
            try (Memory memory = new Memory(Integer.BYTES)) {
                getDeviceAttribute(memory, attribute);
                return memory.getInt(0);
            }
        }

        public void getDeviceAttribute(Memory dst, CUdevice_attribute attribute) {
            check(driverAPI().cuDeviceGetAttribute(dst, attribute.value(), this));
        }

        public long getTotalMem() {
            PointerByReference bytes = new PointerByReference();
            check(driverAPI().cuDeviceTotalMem(bytes, this));
            return Pointer.nativeValue(bytes.getValue());
        }

        /**
         * @param flags mask of {@link CUctx_flags}
         * @return new cuContext
         * Note : In most cases it is recommended to use
         * <a href="https://docs.nvidia.com/cuda/cuda-driver-api/group__CUDA__PRIMARY__CTX.html#group__CUDA__PRIMARY__CTX_1g9051f2d5c31501997a6cb0530290a300">cuDevicePrimaryCtxRetain</a>.
         */
        public CUcontext createContext(CUctx_flags... flags) {
            CUcontext.ByReference ptrRef = new CUcontext.ByReference();
            check(driverAPI().cuCtxCreate(ptrRef, Set.of(flags), this));
            return ptrRef.getValue();
        }

        public CUcontext primaryCtxRetain() {
            CUcontext.ByReference ptrRef = new CUcontext.ByReference();
            check(driverAPI().cuDevicePrimaryCtxRetain(ptrRef, this));
            return ptrRef.getValue();
        }

        public void primaryCtxRelease() {
            check(driverAPI().cuDevicePrimaryCtxRelease(this));
        }

        public static class ByReference extends PointerByReference {
            @Override
            public CUdevice getValue() {
                return new CUdevice(super.getValue());
            }
        }
    }

    class CUcontext extends Pointer implements AutoCloseable {
        public CUcontext(long peer) {
            super(peer);
        }

        public CUcontext(Pointer pointer) {
            super(Pointer.nativeValue(pointer));
        }

        public void setCurrent() {
            check(driverAPI().cuCtxSetCurrent(this));
        }

        @Override
        public void close() {
            check(driverAPI().cuCtxDestroy(this));
        }

        public static CUcontext getCurrent() {
            CUcontext.ByReference ptrRef = new CUcontext.ByReference();
            check(driverAPI().cuCtxGetCurrent(ptrRef));
            return ptrRef.getValue();
        }

        public static class ByReference extends PointerByReference {
            @Override
            public CUcontext getValue() {
                return new CUcontext(super.getValue());
            }
        }
    }

    class CULibrary extends Pointer implements AutoCloseable {
        public CULibrary(long peer) {
            super(peer);
        }

        public CULibrary(Pointer pointer) {
            super(Pointer.nativeValue(pointer));
        }

        @Override
        public void close() throws Exception {
            check(driverAPI().cuLibraryUnload(this));
        }

        public static class ByReference extends PointerByReference {
            @Override
            public CULibrary getValue() {
                return new CULibrary(super.getValue());
            }
        }
    }

    class CUmodule extends Pointer implements AutoCloseable {
        public CUmodule(long peer) {
            super(peer);
        }

        public CUmodule(Pointer pointer) {
            super(Pointer.nativeValue(pointer));
        }

        @Override
        public void close() {
            check(driverAPI().cuModuleUnload(this));
        }

        public static class ByReference extends PointerByReference {
            @Override
            public CUmodule getValue() {
                return new CUmodule(super.getValue());
            }
        }
    }

    class CUKernel extends Pointer {
        public CUKernel(long peer) {
            super(peer);
        }

        public CUKernel(Pointer pointer) {
            super(Pointer.nativeValue(pointer));
        }

        public static class ByReference extends PointerByReference {
            @Override
            public CUKernel getValue() {
                return new CUKernel(super.getValue());
            }
        }
    }

    class CUfunction extends Pointer {
        public CUfunction(long peer) {
            super(peer);
        }

        public CUfunction(Pointer pointer) {
            super(Pointer.nativeValue(pointer));
        }

        public static class ByReference extends PointerByReference {
            @Override
            public CUfunction getValue() {
                return new CUfunction(super.getValue());
            }
        }
    }

    class CUstream extends Pointer {
        public CUstream(long peer) {
            super(peer);
        }

        public CUstream(Pointer pointer) {
            super(Pointer.nativeValue(pointer));
        }

        public static class ByReference extends PointerByReference {
            @Override
            public CUstream getValue() {
                return new CUstream(super.getValue());
            }
        }
    }

    // 6.3. Initialization
    CUresult cuInit(int flags);

    default CUresult cuInit(CUctx_flags... flags) {
        return cuInit(NativeEnum.all(Set.of(flags)));
    }

    CUresult cuDeviceGetCount(IntBuffer count);

    CUresult cuDeviceGet(CUdevice.ByReference device, int ordinal);

    CUresult cuDeviceGetName(Pointer name, int len, CUdevice dev);

    CUresult cuDeviceGetAttribute(Pointer pi, int attrib, CUdevice dev);

    CUresult cuDeviceTotalMem(PointerByReference bytes, CUdevice dev);

    CUresult cuDriverGetVersion(IntBuffer driverVersion);

    CUresult cuDriverGetVersion(Pointer driverVersion);

    // 6.7. Primary Context Management

    CUresult cuDevicePrimaryCtxRetain(CUcontext.ByReference pctx, CUdevice dev);

    CUresult cuDevicePrimaryCtxRelease(CUdevice dev);

    CUresult cuDevicePrimaryCtxSetFlags(CUdevice dev, int flags);

    default CUresult cuDevicePrimaryCtxSetFlags(CUdevice dev, Collection<CUctx_flags> flags) {
        return cuDevicePrimaryCtxSetFlags(dev, NativeEnum.all(flags));
    }

    // 6.8. Context Management
    CUresult cuCtxCreate(CUcontext.ByReference pctx, int flags, CUdevice dev);

    default CUresult cuCtxCreate(CUcontext.ByReference pctx, Collection<CUctx_flags> flags, CUdevice dev) {
        return cuCtxCreate(pctx, NativeEnum.all(flags), dev);
    }

    CUresult cuCtxDestroy(CUcontext ctx);

    CUresult cuCtxGetCurrent(CUcontext.ByReference pctx);

    CUresult cuCtxGetDevice(CUdevice.ByReference device);

    CUresult cuCtxSetCurrent(CUcontext ctx);

    CUresult cuCtxPushCurrent(CUcontext ctx);

    CUresult cuCtxSynchronize();

    // 6.10. Module Management
    CUresult cuModuleLoad(CUmodule.ByReference module, String fname);

    default CUresult cuModuleLoadData(CUmodule.ByReference module, Path file) {
        return cuModuleLoad(module, file.toString());
    }

    CUresult cuModuleLoadData(CUmodule.ByReference module, Pointer image);

    CUresult cuModuleLoadFatBinary(PointerByReference module, Pointer fatCubin);

    default CUresult cuModuleLoadFatBinary(PointerByReference module, Path fatCubin) throws IOException {
        return cuModuleLoadFatBinary(module, NativeSupport.loadFile(fatCubin));
    }

    CUresult cuModuleGetFunction(CUfunction.ByReference hfunc, CUmodule hmod, String name);

    CUresult cuModuleUnload(CUmodule hmod);

    CUresult cuModuleEnumerateFunctions(CUfunction[] functions, int numFunctions, CUmodule mod);

    CUresult cuModuleGetFunctionCount(IntBuffer count, CUmodule mod);

    CUresult cuModuleGetFunctionCount(Memory count, CUmodule mod);

    // 6.12. Library Management
    CUresult cuLibraryLoadData(CULibrary.ByReference library, Pointer code,
                               CUjit_option[] jitOptions, Pointer[] jitOptionsValues, int numJitOptions,
                               CUlibraryOption[] libraryOptions, Pointer[] libraryOptionValues, int numLibraryOptions);

    CUresult cuLibraryLoadFromFile(CULibrary.ByReference library, String fileName,
                                   CUjit_option[] jitOptions, Pointer[] jitOptionsValues, int numJitOptions,
                                   CUlibraryOption[] libraryOptions, Pointer[] libraryOptionValues, int numLibraryOptions);

    CUresult cuLibraryGetKernel(CUKernel.ByReference pKernel, CULibrary library, String name);

    CUresult cuLibraryGetKernelCount(IntBuffer count, CULibrary lib);

    CUresult cuLibraryEnumerateKernels(Pointer[] kernels, int numKernels, CULibrary lib);

    CUresult cuLibraryGetGlobal(PointerByReference dptr, PointerByReference bytes, CULibrary library, String name);

    CUresult cuLibraryGetModule(CUmodule.ByReference pMod, CULibrary library);

    CUresult cuKernelGetName(PointerByReference pointer, CUKernel kernel);

    CUresult cuKernelGetFunction(PointerByReference pFunc, CUKernel kernel);

    CUresult cuKernelGetParamInfo(CUKernel kernel, size_t paramIndex, PointerByReference paramOffset, PointerByReference paramSize);

    CUresult cuKernelGetAttribute(IntBuffer pi, CUfunction_attribute attrib, CUKernel kernel, CUdevice dev);

    CUresult cuKernelSetAttribute(CUfunction_attribute attrib, int val, CUKernel kernel, CUdevice dev);

    CUresult cuKernelSetCacheConfig(Pointer kernel, CUfunc_cache config, CUdevice dev);

    CUresult cuLibraryUnload(CULibrary library);

    // 6.13. Memory Management
    CUresult cuMemAlloc(PointerByReference dptr, long bytesize);

    CUresult cuMemFree(Pointer dptr);

    CUresult cuMemcpyHtoD(Pointer dstDevice, Pointer srcHost, long ByteCount);

    CUresult cuMemcpyDtoH(Pointer dstHost, Pointer srcDevice, long ByteCount);

    CUresult cuMemGetInfo(PointerByReference free, PointerByReference total);

    CUresult cuMemHostRegister(Pointer p, long bytesize, int Flags);

    CUresult cuMemHostUnregister(Pointer p);

    CUresult cuMemHostAlloc(PointerByReference pp, long bytesize, int Flags);

    CUresult cuMemHostGetDevicePointer(PointerByReference pdptr, Pointer p, int Flags);

    CUresult cuMemsetD16(Pointer dstDevice, short us, long N);

    CUresult cuMemsetD32(Pointer dstDevice, int ui, long N);

    CUresult cuMemsetD8(Pointer dstDevice, char uc, long N);

    // 6.22. Execution Control
    CUresult cuLaunchKernel(CUfunction f, int gridDimX, int gridDimY, int gridDimZ, int blockDimX, int blockDimY, int blockDimZ, int sharedMemBytes, Pointer hStream, Pointer kernelParams, Pointer extra);

    CUresult cuFuncGetName(PointerByReference name, CUfunction hfunc);

    CUresult cuFuncGetModule(CUmodule.ByReference hmod, CUfunction hfunc);

    CUresult cuFuncGetParamInfo(CUfunction func, size_t paramIndex, PointerByReference paramOffset, PointerByReference paramSize);

    CUresult cuLaunchCooperativeKernel(CUfunction f, int gridDimX, int gridDimY, int gridDimZ, int blockDimX, int blockDimY, int blockDimZ, int sharedMemBytes, CUstream hStream, Pointer kernelParams);

    CUresult cuLaunchKernelEx(CUlaunchConfig.ByReference config, CUfunction f, Pointer kernelParams, Pointer extra);

    // 6.2 Error handling
    CUresult cuGetErrorName(CUresult error, PointerByReference pStr);

    CUresult cuGetErrorString(CUresult error, PointerByReference pStr);


}
