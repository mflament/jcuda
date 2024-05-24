package org.yah.tools.cuda.proxy;

import com.sun.jna.Memory;
import com.sun.jna.Pointer;
import com.sun.jna.ptr.PointerByReference;
import org.yah.tools.cuda.proxy.annotations.GridThreads;
import org.yah.tools.cuda.proxy.annotations.Kernel;
import org.yah.tools.cuda.support.DriverSupport;
import org.yah.tools.cuda.support.program.NVRTCProgramBuilder;

import java.util.stream.IntStream;

import static org.yah.tools.cuda.api.driver.Driver.CUcontext;
import static org.yah.tools.cuda.api.driver.Driver.CUdevice;
import static org.yah.tools.cuda.api.nvrtc.NVRTC.nvrtcProgram;
import static org.yah.tools.cuda.support.DriverSupport.check;
import static org.yah.tools.cuda.support.DriverSupport.driverAPI;

public class Example {

    private static final String KERNEL = "extern \"C\" __global__ void sum(int N, int *a, int *b, int *c)\n" +
            "{\n" +
            "    int tid = blockIdx.x * blockDim.x + threadIdx.x;\n" +
            "    if (tid < N)\n" +
            "        c[tid] = a[tid] + b[tid];\n" +
            "}\n";

    /**
     * This is the interface used to map cuda kernels
     */
    public interface SumModule extends Module {
        int BLOCK_DIM = 1024;

        @Kernel(blockDim = {BLOCK_DIM, 1, 1})
        void sum(@GridThreads(exposed = true) int N, // use N as number of grid threads and also pass it to the kernel
                 Pointer a, Pointer b, Pointer c);

    }

    public static void main(String[] args) {
        // initialize Cuda driver API
        CUdevice device = DriverSupport.getDevice(0);
        try (CUcontext ctx = device.createContext()) {
            ctx.setCurrent();

            // create the cuda program (and compile for the current device GPU compute version)
            nvrtcProgram program = NVRTCProgramBuilder.create(KERNEL)
                    .programName("sumKernel")
                    .computeVersion(device)
                    .build();

            // create the module build, it will introspect the given interface and create a proxy to launch any method annotated with @Kernel
            KernelProxyFactory kernelProxyFactory = new KernelProxyFactory();

            int N = 512;
            // create java heap data
            int[] a = IntStream.range(0, N).toArray(), b = IntStream.range(N, 2 * N).toArray();

            // prepare a host memory used to transfer from/to java heap to/from cuda device
            try (Memory hostMemory = new Memory(N * (long) Integer.BYTES);
                 // proxy will unload the module on close
                 SumModule module = kernelProxyFactory.createKernelProxy(program, SumModule.class)) {
                // allocate and copy java a and b array to device
                Pointer aPtr = copyToDevice(a, hostMemory);
                Pointer bPtr = copyToDevice(b, hostMemory);

                // allocate result array on device
                Pointer cPtr = allocateInts(N);

                // launch the kernel
                module.sum(N, aPtr, bPtr, cPtr);
                // wait for completion, check for error
                DriverSupport.synchronizeContext();

                // read back results
                check(driverAPI().cuMemcpyDtoH(hostMemory, cPtr, N * (long) Integer.BYTES));
                for (int i = 0; i < N; i++) {
                    int expected = a[i] + b[i];
                    int actual = hostMemory.getInt(i * (long) Integer.BYTES);
                    if (expected != actual)
                        throw new IllegalStateException(String.format("Error at [%d] : %d != %d", i, actual, expected));
                }
                System.out.println("Ok");
            }
        }

    }

    private static Pointer copyToDevice(int[] a, Memory hostMemory) {
        hostMemory.write(0, a, 0, a.length);
        Pointer ptr = allocateInts(a.length);
        check(driverAPI().cuMemcpyHtoD(ptr, hostMemory, a.length * (long) Integer.BYTES));
        return ptr;
    }

    private static Pointer allocateInts(int n) {
        PointerByReference ptr = new PointerByReference();
        check(driverAPI().cuMemAlloc(ptr, n * (long) Integer.BYTES));
        return ptr.getValue();
    }

}
