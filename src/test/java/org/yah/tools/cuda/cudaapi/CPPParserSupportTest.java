package org.yah.tools.cuda.cudaapi;

import java.io.IOException;
import java.nio.file.Path;
import java.nio.file.Paths;
import java.util.List;

import static org.yah.tools.cuda.cudaapi.CudaAPIFile.*;

class CPPParserSupportTest {

    private static final Path CUDA_PATH = Paths.get(System.getenv("CUDA_PATH"));

    public static void main(String[] args) throws IOException {
//        String source = TestsHelper.loadSource("gemm.cu");

//        Path path = CUDA_PATH.resolve("include/cublas_api.h");
//        Path path = CUDA_PATH.resolve("include/cuda.h");
//        Path path = CUDA_PATH.resolve("include/cuda_runtime_api.h");
//        Path path = CUDA_PATH.resolve("include/driver_types.h");
//        library_types.h
        Path path = CUDA_PATH.resolve("include/nvrtc.h");
//        Path path = Paths.get("src/test/resources/sandbox.cpp");
        String source = preprocess(path);
//        System.out.println(source);

        CudaAPIFile file = parse(source);
        System.out.println(dumpCudaAPIFile(file));
    }

    private static String dumpCudaAPIFile(CudaAPIFile file) {
        StringBuilder sb = new StringBuilder();
        sb.append("defines\n");
        for (List<DefineDeclaration> value : file.defines.values()) {
            sb.append("  ").append(value).append("\n");
        }
        sb.append("\nenums\n");
        for (List<EnumDeclaration> value : file.enums.values()) {
            sb.append("  ").append(value).append("\n");
        }
        sb.append("\nstructs\n");
        for (List<StructDeclaration> value : file.structs.values()) {
            sb.append("  ").append(value).append("\n");
        }
        sb.append("\ntypedefs\n");
        for (List<TypedefDeclaration> value : file.typedefs.values()) {
            sb.append("  ").append(value).append("\n");
        }
        sb.append("\nfunctions\n");
        for (List<FunctionDeclaration> value : file.functions.values()) {
            sb.append("  ").append(value).append("\n");
        }
        return sb.toString();
    }

    private static String preprocess(Path path) throws IOException {
        long start = System.currentTimeMillis();
        String result = CudaAPIPreprocessor.preprocess(path);
        System.out.println("preprocessed in " + (System.currentTimeMillis() - start) + " ms");
        return result;
    }

    private static CudaAPIFile parse(String source) {
        long start = System.currentTimeMillis();
        CudaAPIFile result = CudaAPIFileParser.parse(source);
        System.out.println("parsed in " + (System.currentTimeMillis() - start) + " ms");
        return result;
    }

}