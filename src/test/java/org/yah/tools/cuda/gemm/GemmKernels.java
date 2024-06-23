package org.yah.tools.cuda.gemm;

public interface GemmKernels {

//    @Kernel("gemm::sgemm")
    void sgemm(int M);

//    @Kernel("gemm::dgemm")
    void dgemm(int M);


}
