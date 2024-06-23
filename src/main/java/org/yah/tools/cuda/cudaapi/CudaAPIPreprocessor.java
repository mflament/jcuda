package org.yah.tools.cuda.cudaapi;

import org.antlr.v4.runtime.CharStreams;
import org.antlr.v4.runtime.CommonTokenStream;
import org.antlr.v4.runtime.TokenStream;
import org.yah.tools.cuda.cudapi.PreprocessorLexer;
import org.yah.tools.cuda.cudapi.PreprocessorParser;
import org.yah.tools.cuda.cudapi.PreprocessorParserBaseVisitor;

import java.io.IOException;
import java.nio.file.Path;

public class CudaAPIPreprocessor {

    public static String preprocess(Path path) throws IOException {
        PreprocessorLexer lexer = new PreprocessorLexer(CharStreams.fromPath(path));
        TokenStream tokenStream = new CommonTokenStream(lexer);
        PreprocessorParser parser = new PreprocessorParser(tokenStream);
        PreprocessorParser.TranslationUnitContext translationUnitContext = parser.translationUnit();
        return translationUnitContext.accept(new Visitor());
    }

    private static final class Visitor extends PreprocessorParserBaseVisitor<String> {
        private final StringBuilder output = new StringBuilder(1024 * 1024);

        @Override
        public String visitTranslationUnit(PreprocessorParser.TranslationUnitContext ctx) {
            visitChildren(ctx);
            return output.toString();
        }

        @Override
        public String visitPart(PreprocessorParser.PartContext ctx) {
            if (ctx.ifDirective() != null)
                visitIfDirective(ctx.ifDirective());
            else
                output.append(ctx.getText());
            return null;
        }

        @Override
        public String visitIfDirective(PreprocessorParser.IfDirectiveContext ctx) {
            boolean isCPP;
            if (ctx.IfDirective() != null)
                isCPP = isCPP(ctx.condition());
            else
                isCPP = isCPP(ctx.condition());
            if (!isCPP)
                return super.visitChildren(ctx);

            PreprocessorParser.ElseDirectiveContext elseCtx = ctx.elseDirective();
            if (ctx.IfDirective() != null) {
                if (evalIfCondition(ctx.condition())) {
                    if (elseCtx != null)
                        return visitElseDirective(elseCtx);
                } else if (ctx.parts() != null) {
                    visitParts(ctx.parts());
                }
            } else if (ctx.IfdefDirective() != null && elseCtx != null) {
                // ignore if part
                if (elseCtx.ElifDirective() != null)
                    throw new UnsupportedOperationException("TODO : handle #elif " + ctx.getText());
                if (ctx.parts() != null)
                    visitParts(ctx.parts());
                return visitElseDirective(elseCtx);
            } else if (ctx.IfndefDirective() != null && ctx.parts() != null) {
                // ignore else part
                return visitParts(ctx.parts());
            }
            return null;
        }


        private boolean isCPP(PreprocessorParser.ConditionContext directiveConditionContext) {
            return directiveConditionContext.getText().contains("__cplusplus");
        }

        private boolean evalIfCondition(PreprocessorParser.ConditionContext directiveConditionContext) {
            String text = directiveConditionContext.getText();
            if (text.contains("!defined("))
                return false;
            return text.contains("defined(");
        }

    }
}
