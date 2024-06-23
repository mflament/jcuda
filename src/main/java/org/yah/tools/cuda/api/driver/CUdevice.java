package org.yah.tools.cuda.api.driver;

import com.sun.jna.Memory;
import com.sun.jna.Pointer;
import com.sun.jna.ptr.PointerByReference;

import java.util.Set;

import static org.yah.tools.cuda.support.DriverSupport.cuCheck;
import static org.yah.tools.cuda.support.DriverSupport.driverAPI;
import static org.yah.tools.cuda.support.NativeSupport.readNTS;

public class CUdevice extends Pointer {
    public CUdevice(long peer) {
        super(peer);
    }

    public CUdevice(Pointer pointer) {
        super(Pointer.nativeValue(pointer));
    }

    private static final int MAX_NAME_SIZE = 512;

    public String getDeviceName() {
        try (Memory memory = new Memory(MAX_NAME_SIZE)) {
            cuCheck(driverAPI().cuDeviceGetName(memory, MAX_NAME_SIZE, this));
            return readNTS(memory, MAX_NAME_SIZE);
        }
    }

    public String getComputeCapabilityVersion() {
        int[] cc = getDeviceAttributes(CUdevice_attribute.CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MAJOR, CUdevice_attribute.CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MINOR);
        return String.format("%d%d", cc[0], cc[1]);
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
        cuCheck(driverAPI().cuDeviceGetAttribute(dst, attribute.value(), this));
    }

    public long getTotalMem() {
        PointerByReference bytes = new PointerByReference();
        cuCheck(driverAPI().cuDeviceTotalMem(bytes, this));
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
        cuCheck(driverAPI().cuCtxCreate(ptrRef, Set.of(flags), this));
        return ptrRef.getValue();
    }

    public CUcontext primaryCtxRetain() {
        CUcontext.ByReference ptrRef = new CUcontext.ByReference();
        cuCheck(driverAPI().cuDevicePrimaryCtxRetain(ptrRef, this));
        return ptrRef.getValue();
    }

    public void primaryCtxRelease() {
        cuCheck(driverAPI().cuDevicePrimaryCtxRelease(this));
    }

    public static class ByReference extends PointerByReference {
        @Override
        public CUdevice getValue() {
            return new CUdevice(super.getValue());
        }
    }
}
