package org.yah.tools.cuda.gemm;

import com.sun.jna.Memory;
import com.sun.jna.Native;
import com.sun.jna.Pointer;
import com.sun.jna.ptr.PointerByReference;
import org.yah.tools.cuda.TestsHelper;
import org.yah.tools.cuda.api.driver.CUKernel;
import org.yah.tools.cuda.api.driver.CULibrary;
import org.yah.tools.cuda.api.driver.CUcontext;
import org.yah.tools.cuda.api.driver.CUdevice;
import org.yah.tools.cuda.api.driver.CUfunction;
import org.yah.tools.cuda.api.driver.CUmodule;
import org.yah.tools.cuda.api.driver.Driver;
import org.yah.tools.cuda.api.nvrtc.NVRTC;
import org.yah.tools.cuda.api.nvrtc.nvrtcProgram;
import org.yah.tools.cuda.api.nvrtc.nvrtcResult;
import org.yah.tools.cuda.api.runtime.Runtime;
import org.yah.tools.cuda.api.runtime.cudaMemcpyKind;
import org.yah.tools.cuda.api.runtime.dim3;
import org.yah.tools.cuda.support.CudaException;
import org.yah.tools.cuda.support.DriverSupport;
import org.yah.tools.cuda.support.NVRTCSupport;
import org.yah.tools.cuda.support.RuntimeSupport;
import org.yah.tools.cuda.support.library.CudaLibraryBuilder;

import javax.annotation.Nullable;
import java.util.Random;

import static org.yah.tools.cuda.support.DriverSupport.cuCheck;
import static org.yah.tools.cuda.support.DriverSupport.driverAPI;
import static org.yah.tools.cuda.support.NVRTCSupport.nvrtcCheck;
import static org.yah.tools.cuda.support.NativeSupport.readNTS;
import static org.yah.tools.cuda.support.RuntimeSupport.cudaCheck;

public class Sandbox {

    private static final Random randomGenerator = new Random();

    public static void main(String[] args) {
        testModule();
//        testLibrary();
//        testFloatSums();
    }

    private static void testFloatSums() {
        int randomsCount = 1024;
        float[] randoms = new float[randomsCount];
        for (int i = 0; i < randoms.length; i++) {
            randoms[i] = randomGenerator.nextFloat();
        }
        float expected = checkSum(randoms, 1);
        for (int i = 1; i <= 10; i++) {
            test(expected, randoms, i);
        }
    }

    private static void test(float expected, float[] randoms, int batchSize) {
        float error = Math.abs(expected - checkSum(randoms, batchSize));
        if (error > 0)
            System.err.printf("batchSize %d error %.9f%n", batchSize, error);
        else
            System.out.printf("batchSize %d ok%n", batchSize);
    }

    private static float checkSum(float[] randoms, int batchSize) {
        float checkSum = 0;
        for (int i = 0; i < randoms.length; i += batchSize) {
            int bound = Math.min(i + batchSize, randoms.length);
            float batchSum = 0;
            for (int j = i; j < bound; j++) {
                batchSum += randoms[j];
            }
            checkSum += batchSum;
        }
        return checkSum;
    }

    private static Pointer createProgram(@Nullable String smVersion, @Nullable String computeVersion) {
        NVRTC nvrtc = NVRTCSupport.nvrtc();
        nvrtcProgram.ByReference ptrRef = new nvrtcProgram.ByReference();
        nvrtcCheck(nvrtc.nvrtcCreateProgram(ptrRef, TestsHelper.loadSource("gemm.cu"), "sgemm", 0, null, null));
        nvrtcProgram program = new nvrtcProgram(ptrRef.getValue());
        nvrtcResult result = nvrtc.nvrtcCompileProgram(program, 0, null);
        if (result != nvrtcResult.NVRTC_SUCCESS) {
            nvrtcCheck(nvrtc.nvrtcGetProgramLogSize(program, ptrRef));
            long logSize = Pointer.nativeValue(ptrRef.getValue());
            if (logSize > 1L) {
                Memory log = new Memory(logSize + 1);
                nvrtcCheck(nvrtc.nvrtcGetProgramLog(program, log));
                System.err.println("Compilation log:");
                System.err.println("----------------");
                System.err.println(readNTS(log, logSize + 1));
            }
            throw new CudaException("nvrtc", result);
        }
        nvrtcCheck(nvrtc.nvrtcGetPTXSize(program, ptrRef));
        long ptxSize = Pointer.nativeValue(ptrRef.getValue()) + 1;
        Memory ptx = new Memory(ptxSize);
        nvrtcCheck(nvrtc.nvrtcGetPTX(program, ptx));
        System.out.println(readNTS(ptx, ptxSize));
        ptrRef.setValue(program);
        nvrtcCheck(nvrtc.nvrtcDestroyProgram(ptrRef));
//        nvrtcProgram program = NVRTCProgramBuilder.create(TestsHelper.loadSource("gemm.cu"))
//                .smVersion(smVersion)
//                .compileOptions("--std=c++14")
//                .build();
//        Memory ptx = program.getPTX();
//        program.close();
        return ptx;
    }

    private static void testLibrary() {
        Pointer PTX = createProgram("61", null);

        Driver driver = DriverSupport.driverAPI();
        CUdevice cuDevice = DriverSupport.getDevice(0);

//        CUcontext cuContext = cuDevice.createContext();
        CUcontext cuContext = cuDevice.primaryCtxRetain();
        cuContext.setCurrent();

        CULibrary cuLibrary = new CudaLibraryBuilder(PTX)
//                .jitOption(CUjit_option.CU_JIT_TARGET_FROM_CUCONTEXT, null)
                .build();

        CUKernel.ByReference kernelRef = new CUKernel.ByReference();
        cuCheck(driver.cuLibraryGetKernel(kernelRef, cuLibrary, "sgemm"));
        CUKernel kernel = kernelRef.getValue();

        CUfunction.ByReference funcRef = new CUfunction.ByReference();
        cuCheck(driver.cuKernelGetFunction(funcRef, kernel));
        CUfunction function = funcRef.getValue();

        launchKernel(function, true);

        cuLibrary.close();
//        cuContext.close();
        cuDevice.primaryCtxRelease();
    }

    private static void testModule() {
        Pointer PTX = createProgram("61", null);

        Driver driver = driverAPI();
        CUdevice cuDevice = DriverSupport.getDevice(0);

        CUcontext cuContext = cuDevice.createContext();
//        CUcontext cuContext = cuDevice.primaryCtxRetain();
        cuContext.setCurrent();

        CUmodule.ByReference moduleRef = new CUmodule.ByReference();
        cuCheck(driver.cuModuleLoadData(moduleRef, PTX));
        CUmodule cuModule = moduleRef.getValue();


        CUfunction.ByReference funcRef = new CUfunction.ByReference();
        cuCheck(driver.cuModuleGetFunction(funcRef, cuModule, "sgemm"));
        CUfunction function = funcRef.getValue();

        launchKernel(function, false);

        cuCheck(driver.cuModuleUnload(cuModule));
//        cuDevice.primaryCtxRelease();
        cuContext.close();
    }

    private static void launchKernel(CUfunction function, boolean useRuntime) {
        Driver driver = DriverSupport.driverAPI();


        dim3 gridDim = new dim3(1, 1, 1);
        dim3 blockDim = new dim3(1, 1, 1);
//        gridDim.write();
//        blockDim.write();

        int M = 4;
        Memory argsMem = new Memory(Integer.BYTES + Native.POINTER_SIZE);
        argsMem.setInt(0, M);

        long Asize = M * Float.BYTES;
        Memory Ahost = new Memory(Asize);
        for (int i = 0; i < M; i++) Ahost.setFloat(i * Float.BYTES, i * 0.1f);

        PointerByReference Agpu = new PointerByReference();
        if (useRuntime) {
            Runtime runtime = RuntimeSupport.runtimeAPI();
            runtime.cudaSetDevice(0);
            cudaCheck(runtime.cudaMalloc(Agpu, Asize));
            cudaCheck(runtime.cudaMemcpy(Agpu.getValue(), Ahost, Asize, cudaMemcpyKind.cudaMemcpyHostToDevice));
        } else {
            cuCheck(driver.cuMemAlloc(Agpu, M * Float.BYTES));
            cuCheck(driver.cuMemcpyHtoD(Agpu.getValue(), argsMem.share(Integer.BYTES), Asize));
        }

        argsMem.setPointer(Integer.BYTES, Agpu.getValue());
        cuCheck(driver.cuLaunchKernel(function,
                gridDim.x, gridDim.y, gridDim.z,
                blockDim.x, blockDim.y, blockDim.z, 0, null,
                new Pointer[]{argsMem.share(0), argsMem.share(Integer.BYTES)},
                null));
        cuCheck(driver.cuCtxSynchronize());

        argsMem.close();
    }


}
