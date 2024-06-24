package org.yah.tools.cuda.cudaapi;

import org.yah.tools.cuda.cudapi.APIParser;

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

    public record Literal(String text) {
        public int intValue() {
            return parseIntLiteral(text);
        }

        public long longValue() {
            return parseLongLiteral(text);
        }

        public float floatValue() {
            return parseFloatLiteral(text);
        }

        public double doubleValue() {
            return parseDoubleLiteral(text);
        }
    }


    public record DefineDeclaration(String name, Object value,
                                    String doc, String comment) implements Declaration, Documented {
        @Override
        public String toString() {
            return "DefineDeclaration{" +
                    "name='" + name + '\'' +
                    ", value=" + value +
                    '}';
        }
    }

    public record EnumDeclaration(String name, List<EnumConstant> enumConstants,
                                  String doc, String comment) implements Declaration, Documented {
        @Override
        public String toString() {
            return "EnumDeclaration{" +
                    "name='" + name + '\'' +
                    ", enumConstants=" + enumConstants +
                    '}';
        }
    }

    public record EnumConstant(String name, @Nullable Literal value, String doc,
                               String comment) implements Documented {
        @Override
        public String toString() {
            return "EnumConstant{" +
                    "name='" + name + '\'' +
                    ", value=" + value +
                    '}';
        }
    }

    public record StructDeclaration(String name, List<StructField> fields,
                                    String doc, String comment) implements Declaration, Documented {
        @Override
        public String toString() {
            return "StructDeclaration{" +
                    "name='" + name + '\'' +
                    ", fields=" + fields +
                    '}';
        }
    }

    public record StructField(Type type, String name, @Nullable ArrayDecl arrayDecl,
                              String doc, String comment) implements Documented {
        @Override
        public String toString() {
            return "StructField{" +
                    "type=" + type +
                    ", name='" + name + '\'' +
                    ", arrayDecl=" + arrayDecl +
                    '}';
        }
    }

    public record FunctionDeclaration(Type type, String name, List<Parameter> parameters,
                                      String doc, String comment) implements Declaration, Documented {
        @Override
        public String toString() {
            return "FunctionDeclaration{" +
                    "type=" + type +
                    ", name='" + name + '\'' +
                    ", parameters=" + parameters +
                    '}';
        }
    }

    public record Parameter(Type type, String name, @Nullable ArrayDecl arrayDecl,
                            String doc, String comment) implements Documented {
        @Override
        public String toString() {
            return "Parameter{" +
                    "type=" + type +
                    ", name='" + name + '\'' +
                    ", arrayDecl=" + arrayDecl +
                    '}';
        }
    }

    public record TypedefDeclaration(Type type, String name,
                                     String doc, String comment) implements Declaration, Documented {
        @Override
        public String toString() {
            return "TypedefDeclaration{" +
                    "type=" + type +
                    ", name='" + name + '\'' +
                    '}';
        }
    }

    public interface Type {
    }

    public record PointerType(Type type, boolean isConst) implements Type {
    }

    public record StructType(String name, boolean isConst) implements Type {
    }

    public record EnumType(String name, boolean isConst) implements Type {
    }

    public record PrimitiveType(APIParser.PrimitiveTypeContext context, boolean isConst) implements Type {
    }

    public record FunctionPointer(Type type, List<Type> parameterTypes, @Nullable String name) implements Type {
    }

    public record ArrayDecl(@Nullable Literal length) {
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

    private static int parseIntLiteral(String s) {
        if (s.startsWith("0x"))
            return Integer.parseInt(s.substring(2), 16);
        if (s.startsWith("0b"))
            return Integer.parseInt(s.substring(2), 2);
        if (s.startsWith("0") && s.length() > 1)
            return Integer.parseInt(s.substring(1), 8);
        return Integer.parseInt(s, 10);
    }

    private static long parseLongLiteral(String s) {
        if (s.startsWith("0x"))
            return Long.parseLong(s.substring(2), 16);
        if (s.startsWith("0b"))
            return Long.parseLong(s.substring(2), 2);
        if (s.startsWith("0") && s.length() > 1)
            return Long.parseLong(s.substring(1), 8);
        return Long.parseLong(s, 10);
    }

    private static float parseFloatLiteral(String s) {
        return Float.parseFloat(s);
    }

    private static double parseDoubleLiteral(String s) {
        return Double.parseDouble(s);
    }
}
