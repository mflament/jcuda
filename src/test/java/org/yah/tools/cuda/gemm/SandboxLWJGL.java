package org.yah.tools.cuda.gemm;

import org.lwjgl.PointerBuffer;
import org.lwjgl.cuda.CU;
import org.lwjgl.cuda.NVRTC;
import org.lwjgl.system.MemoryStack;
import org.lwjgl.system.MemoryUtil;
import org.yah.tools.cuda.TestsHelper;

import java.nio.ByteBuffer;
import java.nio.FloatBuffer;
import java.nio.IntBuffer;

import static org.lwjgl.cuda.CU.*;
import static org.lwjgl.cuda.NVRTC.*;
import static org.lwjgl.system.MemoryStack.stackPush;
import static org.lwjgl.system.MemoryUtil.*;

public class SandboxLWJGL {

    private static long ctx;

    public static void main(String[] args) {
        ByteBuffer PTX;
        try (MemoryStack stack = stackPush()) {
            PointerBuffer pp = stack.mallocPointer(1);
            checkNVRTC(nvrtcCreateProgram(pp, TestsHelper.loadSource("gemm.cu"), "gemm.cu", null, null));
            long program = pp.get(0);
            int compilationStatus = nvrtcCompileProgram(program, null);
            {
                checkNVRTC(nvrtcGetProgramLogSize(program, pp));
                if (1L < pp.get(0)) {
                    ByteBuffer log = stack.malloc((int) pp.get(0) - 1);

                    checkNVRTC(nvrtcGetProgramLog(program, log));
                    System.err.println("Compilation log:");
                    System.err.println("----------------");
                    System.err.println(MemoryUtil.memASCII(log));
                }
            }
            checkNVRTC(compilationStatus);


            checkNVRTC(nvrtcGetPTXSize(program, pp));
            PTX = memAlloc((int) pp.get(0));
            checkNVRTC(nvrtcGetPTX(program, PTX));
        }

        try (MemoryStack stack = stackPush()) {
            PointerBuffer pp = stack.mallocPointer(1);
            IntBuffer pi = stack.mallocInt(1);
            check(cuInit(0));
            check(cuDeviceGet(pi, 0));
            int device = pi.get(0);
            ByteBuffer pb = stack.malloc(100);
            check(cuDeviceGetName(pb, device));
            System.out.format("> Using device 0: %s\n", memASCII(memAddress(pb)));
            IntBuffer minor = stack.mallocInt(1);
            check(cuDeviceComputeCapability(pi, minor, device));
            System.out.format("> GPU Device has SM %d.%d compute capability\n", pi.get(0), minor.get(0));

            check(cuCtxCreate(pp, 0, device));
            ctx = pp.get(0);
            check(cuModuleLoadData(pp, PTX));
            long module = pp.get(0);
            check(cuModuleGetFunction(pp, module, "sgemm"));
            long function = pp.get(0);

            int M = 4;
            check(cuMemAlloc(pp, Float.BYTES * M));
            long deviceA = pp.get(0);
            FloatBuffer hostA = stack.mallocFloat(M);
            for (int i = 0; i < M; i++) hostA.put(i, i * 0.1f);
            check(cuMemcpyHtoD(deviceA, hostA));

            check(cuLaunchKernel(
                    function, 1, 1, 1,
                    1, 1, 1,
                    0, 0,
                    stack.pointers(
                            memAddress(stack.ints(M)),
                            memAddress(stack.pointers(deviceA))
                    ), null));
            check(cuCtxSynchronize());
        }
    }

    private static void checkNVRTC(int err) {
        if (err != NVRTC.NVRTC_SUCCESS) {
            throw new IllegalStateException(nvrtcGetErrorString(err));
        }
    }

    private static void check(int err) {
        if (err != CU.CUDA_SUCCESS) {
            if (ctx != NULL) {
                cuCtxDetach(ctx);
                ctx = NULL;
            }
            throw new IllegalStateException(Integer.toString(err));
        }
    }

}
