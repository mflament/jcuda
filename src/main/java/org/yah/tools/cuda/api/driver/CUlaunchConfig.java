package org.yah.tools.cuda.api.driver;

import com.sun.jna.Structure;

@Structure.FieldOrder({
        "blockDimX",
        "blockDimY",
        "blockDimZ",
        "gridDimX",
        "gridDimY",
        "gridDimZ",
        "hStream",
        "numAttrs",
        "sharedMemBytes"
})
public class CUlaunchConfig extends Structure {
    private int blockDimX;
    private int blockDimY;
    private int blockDimZ;
    private int gridDimX;
    private int gridDimY;
    private int gridDimZ;
    private Driver.CUstream hStream;
    private int numAttrs;
    private int sharedMemBytes;

    public int getSharedMemBytes() {
        return sharedMemBytes;
    }

    public void setSharedMemBytes(int sharedMemBytes) {
        this.sharedMemBytes = sharedMemBytes;
    }

    public int getNumAttrs() {
        return numAttrs;
    }

    public void setNumAttrs(int numAttrs) {
        this.numAttrs = numAttrs;
    }

    public Driver.CUstream gethStream() {
        return hStream;
    }

    public void sethStream(Driver.CUstream hStream) {
        this.hStream = hStream;
    }

    public int getGridDimZ() {
        return gridDimZ;
    }

    public void setGridDimZ(int gridDimZ) {
        this.gridDimZ = gridDimZ;
    }

    public int getGridDimY() {
        return gridDimY;
    }

    public void setGridDimY(int gridDimY) {
        this.gridDimY = gridDimY;
    }

    public int getGridDimX() {
        return gridDimX;
    }

    public void setGridDimX(int gridDimX) {
        this.gridDimX = gridDimX;
    }

    public int getBlockDimZ() {
        return blockDimZ;
    }

    public void setBlockDimZ(int blockDimZ) {
        this.blockDimZ = blockDimZ;
    }

    public int getBlockDimY() {
        return blockDimY;
    }

    public void setBlockDimY(int blockDimY) {
        this.blockDimY = blockDimY;
    }

    public int getBlockDimX() {
        return blockDimX;
    }

    public void setBlockDimX(int blockDimX) {
        this.blockDimX = blockDimX;
    }

    @Override
    public String toString() {
        return "CUlaunchConfig{" +
                "blockDimX=" + getBlockDimX() +
                ", blockDimY=" + getBlockDimY() +
                ", blockDimZ=" + getBlockDimZ() +
                ", gridDimX=" + getGridDimX() +
                ", gridDimY=" + getGridDimY() +
                ", gridDimZ=" + getGridDimZ() +
                ", hStream=" + gethStream() +
                ", numAttrs=" + getNumAttrs() +
                ", sharedMemBytes=" + getSharedMemBytes() +
                '}';
    }

    public static class ByReference extends CUlaunchConfig implements Structure.ByReference {
    }
}
