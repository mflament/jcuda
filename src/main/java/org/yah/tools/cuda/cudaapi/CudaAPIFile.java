package org.yah.tools.cuda.cudaapi;

import javax.annotation.Nullable;
import java.util.ArrayList;
import java.util.LinkedHashMap;
import java.util.List;

public class CudaAPIFile {
    public interface Declaration {
        String name();
    }

    public interface Documented {
        @Nullable
        String doc();

        @Nullable
        String comment();
    }

    public record DefineDeclaration(String name, Object value,
                                    String doc, String comment) implements Declaration, Documented {
    }

    public record EnumDeclaration(String name, List<EnumConstant> enumConstants,
                                  String doc, String comment) implements Declaration, Documented {
    }

    public record EnumConstant(String name, @Nullable Integer value, String doc, String comment) implements Documented {
    }

    public record StructDeclaration(String name, List<StructField> fields,
                                    String doc, String comment) implements Declaration, Documented {
    }

    public record StructField(Type type, String name, @Nullable ArrayDecl arrayDecl,
                              String doc, String comment) implements Documented {
    }

    public record FunctionDeclaration(Type type, String name, List<Parameter> parameters,
                                      String doc, String comment) implements Declaration, Documented {
    }

    public record Parameter(Type type, String name, @Nullable ArrayDecl arrayDecl,
                            String doc, String comment) implements Documented {
    }

    public record TypedefDeclaration(Type type, String name, String doc,
                                     String comment) implements Declaration, Documented {
    }

    public interface Type {
        boolean isConst();
    }

    record PointerType(Type type, boolean isConst) implements Type {
    }

    record StructType(String name, boolean isConst) implements Type {
    }

    record EnumType(String name, boolean isConst) implements Type {
    }

    record PrimitiveType(List<String> names, boolean isConst) implements Type {
    }

    public record ArrayDecl(@Nullable Object length) {
    }

    public static final class DeclarationsMap<D extends Declaration> extends LinkedHashMap<String, List<D>> {
        public void add(D decl) {
            computeIfAbsent(decl.name(), n -> new ArrayList<>()).add(decl);
        }

        @Nullable
        public D getFirst(String name) {
            List<D> decls = get(name);
            return decls == null ? null : decls.getFirst();
        }
    }

    public final DeclarationsMap<DefineDeclaration> defines = new DeclarationsMap<>();
    public final DeclarationsMap<FunctionDeclaration> functions = new DeclarationsMap<>();
    public final DeclarationsMap<StructDeclaration> structs = new DeclarationsMap<>();
    public final DeclarationsMap<EnumDeclaration> enums = new DeclarationsMap<>();
    public final DeclarationsMap<TypedefDeclaration> typedefs = new DeclarationsMap<>();

}
