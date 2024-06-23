package org.yah.tools.cuda.support;

import org.yah.tools.cuda.api.NativeEnum;

public class CudaException extends RuntimeException {

    private final String module;
    private final NativeEnum error;

    public CudaException(String module, NativeEnum error) {
        super(String.format("%s error %d: %s", module, error.nativeValue(), error.name()));
        this.module = module;
        this.error = error;
    }

    public String getModule() {
        return module;
    }

    public NativeEnum getError() {
        return error;
    }
}
