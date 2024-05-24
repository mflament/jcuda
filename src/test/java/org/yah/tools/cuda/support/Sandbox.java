package org.yah.tools.cuda.support;

import com.sun.jna.Pointer;
import com.sun.jna.ptr.PointerByReference;
import org.yah.tools.cuda.TestsHelper;
import org.yah.tools.cuda.api.driver.Driver.*;
import org.yah.tools.cuda.support.library.CudaLibraryBuilder;
import org.yah.tools.cuda.support.program.NVRTCProgramBuilder;

import java.io.IOException;
import java.nio.ByteBuffer;
import java.nio.ByteOrder;
import java.nio.IntBuffer;
import java.nio.file.Path;

import static org.yah.tools.cuda.api.driver.Driver.*;
import static org.yah.tools.cuda.api.nvrtc.NVRTC.nvrtcProgram;
import static org.yah.tools.cuda.support.DriverSupport.check;
import static org.yah.tools.cuda.support.DriverSupport.driverAPI;
import static org.yah.tools.cuda.support.NativeSupport.readNTS;

public class Sandbox {

    public static void main(String[] args) throws IOException {

        CUdevice device = DriverSupport.getDevice(0);
        try (CUcontext ctx = device.createContext();
             nvrtcProgram program = NVRTCProgramBuilder.create(TestsHelper.loadSource("TestKernel.cu")).computeVersion(device).build()) {
            ctx.setCurrent();

//            Path file = Path.of("src/test/resources/nvrtcTest.cu");
            CULibrary library = new CudaLibraryBuilder()
                    .code(program.getPTX())
                    .build();

            IntBuffer intBuffer = ByteBuffer.allocateDirect(Integer.BYTES).order(ByteOrder.nativeOrder()).asIntBuffer();
            check(driverAPI().cuLibraryGetKernelCount(intBuffer, library));
            int numKernels = intBuffer.get(0);
            Pointer[] kernels = new Pointer[numKernels];
            check(driverAPI().cuLibraryEnumerateKernels(kernels, numKernels, library));
            for (int i = 0; i < numKernels; i++) {
                CUKernel kernel = new CUKernel(kernels[i]);
                PointerByReference namePtr = new PointerByReference();
                check(driverAPI().cuKernelGetName(namePtr, kernel));
                String name = readNTS(namePtr.getValue(), 1024 * 4);
                System.out.println("kernel name " + name);
                CUfunction.ByReference funcPtr = new CUfunction.ByReference();
                check(driverAPI().cuKernelGetFunction(funcPtr, kernel));

                CUfunction function = funcPtr.getValue();
                check(driverAPI().cuFuncGetName(namePtr, function));
                name = readNTS(namePtr.getValue(), 1024 * 4);
                System.out.println("function name " + name);

//            IntBuffer attribute = ByteBuffer.allocateDirect(Integer.BYTES).order(ByteOrder.nativeOrder()).asIntBuffer();
//            check(driverAPI().cuKernelGetAttribute(attribute, Driver.CUfunction_attribute.CU_FUNC_ATTRIBUTE_MAX_THREADS_PER_BLOCK, kernel, device));
//            System.out.println(attribute.get(0));
            }

        }

    }
}
