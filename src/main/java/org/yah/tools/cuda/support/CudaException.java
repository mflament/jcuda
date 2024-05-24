package org.yah.tools.cuda.support;

import org.yah.tools.cuda.api.NativeEnum;

public class CudaException extends RuntimeException {
    private final String module;
    private final int status;

    public <E extends Enum<?> & NativeEnum> CudaException(String module, E error) {
        this(module, error.nativeValue(), error.name());
    }

    public CudaException(String module, int status, String errorName) {
        super(String.format("%s error %d: %s", module, status, errorName));
        this.module = module;
        this.status = status;
    }

    public String getModule() {
        return module;
    }

    public int getStatus() {
        return status;
    }
}
