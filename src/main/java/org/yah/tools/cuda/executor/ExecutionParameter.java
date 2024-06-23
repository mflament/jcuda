package org.yah.tools.cuda.executor;

import org.yah.tools.cuda.api.driver.CUstream;

public class ExecutionParameter {
    public dim3 gridDim;
    public dim3 blockDim;
    public int sharedMem;
    public CUstream stream;
}
