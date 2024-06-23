package org.yah.tools.cuda.cudaapi;

import org.antlr.v4.runtime.CharStreams;
import org.antlr.v4.runtime.CommonTokenStream;
import org.antlr.v4.runtime.ParserRuleContext;
import org.antlr.v4.runtime.TokenStream;
import org.antlr.v4.runtime.tree.ParseTree;
import org.antlr.v4.runtime.tree.TerminalNode;
import org.yah.tools.cuda.cudapi.APILexer;
import org.yah.tools.cuda.cudapi.APIParser;
import org.yah.tools.cuda.cudapi.APIParserBaseVisitor;

import javax.annotation.Nullable;
import java.util.ArrayList;
import java.util.List;
import java.util.Objects;

import static org.yah.tools.cuda.cudaapi.CudaAPIFile.PrimitiveType;
import static org.yah.tools.cuda.cudaapi.CudaAPIFile.*;
import static org.yah.tools.cuda.cudapi.APIParser.*;

public class CudaAPIFileParser {

    public static CudaAPIFile parse(String source) {
        APILexer lexer = new APILexer(CharStreams.fromString(source));
        TokenStream tokenStream = new CommonTokenStream(lexer);
        APIParser parser = new APIParser(tokenStream);
        return parser.translationUnit().accept(new Visitor());
    }

    private static String getText(ParserRuleContext context) {
        return context == null ? null : context.getText();
    }

    private static String getText(TerminalNode node) {
        return node == null ? null : node.getText();
    }

    private static class Visitor extends APIParserBaseVisitor<CudaAPIFile> {

        private final CudaAPIFile file;

        public Visitor() {
            file = new CudaAPIFile();
        }

        @Override
        public CudaAPIFile visitTranslationUnit(TranslationUnitContext ctx) {
            visitChildren(ctx);
            return file;
        }

        @Override
        public CudaAPIFile visitDefineDirective(DefineDirectiveContext ctx) {
            if (ctx.defineValue() != null && ctx.defineValue().literalDefineValue() != null) {
                Object value;
                if (ctx.defineValue().literalDefineValue().IntegerLiteral() != null)
                    value = Integer.parseInt(ctx.defineValue().literalDefineValue().toString());
                else
                    value = Double.parseDouble(ctx.defineValue().literalDefineValue().toString());
                file.defines.add(new DefineDeclaration(ctx.defineName().getText(), value, getText(ctx.doc()), getText(ctx.comment())));
            }
            return null;
        }

        @Override
        public CudaAPIFile visitDeclaration(DeclarationContext ctx) {
            String doc = getText(ctx.doc());
            String comment = getText(ctx.comment());
            if (ctx.typeDeclaration() != null) {
                visitTypeDeclaration(doc, comment, ctx.typeDeclaration());
            } else {
                visitFunctionDeclaration(doc, comment, ctx.functionDeclaration());
            }
            return super.visitDeclaration(ctx);
        }

        private void visitTypeDeclaration(String doc, String comment, TypeDeclarationContext ctx) {
            if (ctx.enumDeclaration() != null) {
                visitEnumDeclaration(doc, comment, ctx.enumDeclaration());
            } else if (ctx.structDeclaration() != null) {
                visitStructDeclaration(doc, comment, ctx.structDeclaration());
            } else {
                visitTypedefDeclaration(doc, comment, ctx.typedefDeclaration());
            }
        }

        private void visitEnumDeclaration(String doc, String comment, EnumDeclarationContext ctx) {
            if (ctx.enumBody() != null && ctx.Identifier() != null)
                createEnum(doc, comment, ctx.Identifier().getText(), ctx.enumBody());
        }

        private void visitStructDeclaration(String doc, String comment, StructDeclarationContext ctx) {
            if (ctx.structBody() != null && ctx.Identifier() != null)
                createStruct(doc, comment, ctx.Identifier().getText(), ctx.structBody());
        }

        private void visitTypedefDeclaration(String doc, String comment, TypedefDeclarationContext ctx) {
            String name = ctx.Identifier().getText();
            NoPointerTypeContext npType = ctx.type().noPointerType();
            if (npType != null && npType.enumDefinition() != null && npType.enumDefinition().Identifier() != null && npType.enumDefinition().enumBody() != null) {
                createEnum(doc, comment, npType.enumDefinition().Identifier().getText(), npType.enumDefinition().enumBody());
            } else if (npType != null && npType.structDefinition() != null && npType.structDefinition().Identifier() != null && npType.structDefinition().structBody() != null) {
                createStruct(doc, comment, npType.structDefinition().Identifier().getText(), npType.structDefinition().structBody());
            } else {
                Type type = parseType(ctx.type(), name, doc, comment);
                file.typedefs.add(new TypedefDeclaration(type, ctx.Identifier().getText(), doc, comment));
            }
        }

        private void visitFunctionDeclaration(String doc, String comment, FunctionDeclarationContext ctx) {
            String name = ctx.functionSignature().Identifier().getText();
            Type type = parseType(ctx.functionSignature().type(), null, null, null);
            FunctionDeclaration funcDecl = new FunctionDeclaration(type, name, new ArrayList<>(), doc, comment);
            file.functions.add(funcDecl);
        }

        private Type parseType(TypeContext ctx, @Nullable String name, @Nullable String doc, @Nullable String comment) {
            return ctx.accept(new TypeVisitor(name, doc, comment));
        }

        private ArrayDecl parseArrayDecl(ArrayDeclContext ctx) {
            Object length = null;
            if (ctx.IntegerLiteral() != null) {
                length = Integer.parseInt(ctx.IntegerLiteral().getText());
            } else if (ctx.defineName() != null) {
                throw new UnsupportedOperationException("TODO : handle array decl constant initializer " + ctx.getText());
            }
            return new ArrayDecl(length);
        }

        private void createEnum(String doc, String comment, String name, EnumBodyContext ctx) {
            EnumDeclaration enumDeclaration = new EnumDeclaration(name, new ArrayList<>(), doc, comment);
            ctx.accept(new EnumVisitor(enumDeclaration));
            file.enums.add(enumDeclaration);
        }

        private void createStruct(String doc, String comment, String name, StructBodyContext ctx) {
            StructDeclaration structDeclaration = new StructDeclaration(name, new ArrayList<>(), doc, comment);
            ctx.accept(new StructVisitor(structDeclaration));
            file.structs.add(structDeclaration);
        }

        private static class EnumVisitor extends APIParserBaseVisitor<Void> {
            private final EnumDeclaration enumDeclaration;

            public EnumVisitor(EnumDeclaration enumDeclaration) {
                this.enumDeclaration = enumDeclaration;
            }

            @Override
            public Void visitEnumConstant(EnumConstantContext ctx) {
                Integer value = null;
                if (ctx.enumConstantValue() != null) {
                    if (ctx.enumConstantValue().IntegerLiteral() != null) {
                        value = Integer.parseInt(ctx.enumConstantValue().IntegerLiteral().getText());
                    } else {
                        throw new UnsupportedOperationException("TODO: resolve enum value from identifier " + ctx.enumConstantValue().defineName().getText());
                    }
                }
                String name = ctx.Identifier().getText();
                if (enumDeclaration.enumConstants().stream().anyMatch(cst -> cst.name().equals(name)))
                    throw new IllegalStateException("duplicate enum " + enumDeclaration.name() + " constant " + name);
                EnumConstant enumConstant = new EnumConstant(ctx.Identifier().getText(), value, getText(ctx.doc()), getText(ctx.comment()));
                enumDeclaration.enumConstants().add(enumConstant);
                return null;
            }
        }

        private class StructVisitor extends APIParserBaseVisitor<Void> {
            private final StructDeclaration structDeclaration;

            public StructVisitor(StructDeclaration structDeclaration) {
                this.structDeclaration = structDeclaration;
            }

            @Override
            public Void visitStructField(StructFieldContext ctx) {
                String name = ctx.Identifier().getText();
                String doc = getText(ctx.doc());
                String comment = getText(ctx.comment());
                Type type = parseType(ctx.type(), name, doc, comment);
                ArrayDecl arrayDecl = null;
                if (ctx.arrayDecl() != null)
                    arrayDecl = parseArrayDecl(ctx.arrayDecl());
                StructField structField = new StructField(type, name, arrayDecl, doc, comment);
                if (structDeclaration.fields().stream().anyMatch(f -> f.name().equals(name))) {
                    throw new IllegalStateException("duplicate struct " + structDeclaration.name() + " field " + name);
                }
                structDeclaration.fields().add(structField);
                return null;
            }
        }

        private static class FunctionVisitor extends APIParserBaseVisitor<Void> {

        }

        private class TypeVisitor extends APIParserBaseVisitor<Type> {
            private final String name;
            private final String doc;
            private final String comment;

            public TypeVisitor(String name, String doc, String comment) {
                this.name = name;
                this.doc = doc;
                this.comment = comment;
            }

            @Override
            public Type visitPointerType(PointerTypeContext ctx) {
                Type type;
                if (ctx.pointerType() != null)
                    type = visitPointerType(ctx.pointerType());
                else
                    type = visitNoPointerType(ctx.noPointerType());
                return new PointerType(type, ctx.Const() != null);
            }

            @Override
            public Type visitNoPointerType(NoPointerTypeContext ctx) {
                boolean isConst = ctx.Const() != null;
                if (ctx.primitiveType() != null) {
                    List<String> names = ctx.primitiveType().PrimitiveType().stream().map(ParseTree::getText).toList();
                    return new PrimitiveType(names, isConst);
                } else if (ctx.enumDefinition() != null) {
                    EnumDefinitionContext enumCtx = ctx.enumDefinition();
                    if (enumCtx.enumBody() != null && enumCtx.Identifier() != null)
                        createEnum(doc, comment, enumCtx.Identifier().getText(), enumCtx.enumBody());
                    return new EnumType(Objects.requireNonNullElse(getText(enumCtx.Identifier()), name), isConst);
                } else {
                    StructDefinitionContext structCtx = ctx.structDefinition();
                    if (structCtx.structBody() != null && structCtx.Identifier() != null)
                        createStruct(doc, comment, structCtx.Identifier().getText(), structCtx.structBody());
                    return new StructType(Objects.requireNonNullElse(getText(structCtx.Identifier()), name), isConst);
                }
            }
        }

    }


}
