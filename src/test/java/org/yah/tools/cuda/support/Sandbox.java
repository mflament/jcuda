package org.yah.tools.cuda.support;

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
import org.yah.tools.cuda.support.library.CudaLibraryBuilder;
import org.yah.tools.cuda.support.program.NVRTCProgramBuilder;

import java.io.IOException;

import org.yah.tools.cuda.api.nvrtc.nvrtcProgram;
import static org.yah.tools.cuda.support.DriverSupport.cuCheck;
import static org.yah.tools.cuda.support.DriverSupport.driverAPI;
import static org.yah.tools.cuda.support.NativeSupport.readNTS;

public class Sandbox {

    public static void main(String[] args) throws IOException {

        CUdevice device = DriverSupport.getDevice(0);
        try (CUcontext ctx = device.createContext();
             nvrtcProgram program = NVRTCProgramBuilder.create(TestsHelper.loadSource("TestKernel.cu")).computeVersion(device).build()) {
            ctx.setCurrent();

//            Path file = Path.of("src/test/resources/nvrtcTest.cu");
            CULibrary library = new CudaLibraryBuilder(program.getPTX()).build();

            Memory kernelCount = new Memory(Integer.BYTES);
            cuCheck(driverAPI().cuLibraryGetKernelCount(kernelCount, library));
            int numKernels = kernelCount.getInt(0);
            long pointerSize = Native.POINTER_SIZE;
            Pointer kernels = new Memory(numKernels * pointerSize);
            cuCheck(driverAPI().cuLibraryEnumerateKernels(kernels, numKernels, library));
            for (int i = 0; i < numKernels; i++) {
                CUKernel kernel = new CUKernel(kernels.getPointer(i* pointerSize));
                PointerByReference namePtr = new PointerByReference();
                cuCheck(driverAPI().cuKernelGetName(namePtr, kernel));
                String name = readNTS(namePtr.getValue(), 1024 * 4);
                System.out.println("kernel name " + name);
                CUfunction.ByReference funcPtr = new CUfunction.ByReference();
                cuCheck(driverAPI().cuKernelGetFunction(funcPtr, kernel));

                CUfunction function = funcPtr.getValue();
                cuCheck(driverAPI().cuFuncGetName(namePtr, function));
                name = readNTS(namePtr.getValue(), 1024 * 4);
                System.out.println("function name " + name);

//            IntBuffer attribute = ByteBuffer.allocateDirect(Integer.BYTES).order(ByteOrder.nativeOrder()).asIntBuffer();
//            check(driverAPI().cuKernelGetAttribute(attribute, Driver.CUfunction_attribute.CU_FUNC_ATTRIBUTE_MAX_THREADS_PER_BLOCK, kernel, device));
//            System.out.println(attribute.get(0));
            }

        }

    }
}
