package org.yah.tools.cuda.support;

import org.slf4j.Logger;
import org.slf4j.LoggerFactory;
import org.yah.tools.cuda.api.driver.CUresult;
import org.yah.tools.cuda.api.driver.Driver;

import javax.annotation.Nullable;
import java.nio.ByteBuffer;
import java.nio.ByteOrder;
import java.nio.IntBuffer;

import static org.yah.tools.cuda.api.driver.Driver.CUdevice;

public class DriverSupport {

    private static final Logger LOGGER = LoggerFactory.getLogger(DriverSupport.class);

    @Nullable
    private static Driver driver;

    public static synchronized Driver driverAPI() {
        if (driver == null) {
            driver = CudaSupport.createDriverAPI();
            check(driver.cuInit(0));
            IntBuffer intBuffer = ByteBuffer.allocateDirect(Integer.BYTES).order(ByteOrder.nativeOrder()).asIntBuffer();
            check(driver.cuDriverGetVersion(intBuffer));
            int version = intBuffer.get(0);
            LOGGER.info("initialized driver API {}.{}", version / 1000, version % 1000 / 10);
        }
        return driver;
    }

    public static synchronized CUdevice getDevice(int ordinal) {
        CUdevice.ByReference reference = new CUdevice.ByReference();
        check(driverAPI().cuDeviceGet(reference, ordinal));
        CUdevice device = reference.getValue();
        if (LOGGER.isInfoEnabled())
            LOGGER.info("Using device {} : {}", ordinal, device.getDeviceName());
        return device;
    }

    public static void check(CUresult result) {
        if (result != CUresult.CUDA_SUCCESS) {
            throw new CudaException("nvcuda", result);
        }
    }

    public static void synchronizeContext() {
        check(driverAPI().cuCtxSynchronize());
    }
}
