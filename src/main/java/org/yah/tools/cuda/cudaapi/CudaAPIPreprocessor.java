package org.yah.tools.cuda.cudaapi;

import org.antlr.v4.runtime.CharStreams;
import org.antlr.v4.runtime.CommonTokenStream;
import org.antlr.v4.runtime.TokenStream;
import org.yah.tools.cuda.cudapi.ConditionLexer;
import org.yah.tools.cuda.cudapi.ConditionParser;
import org.yah.tools.cuda.cudapi.ConditionParserBaseVisitor;
import org.yah.tools.cuda.cudapi.PreprocessorLexer;
import org.yah.tools.cuda.cudapi.PreprocessorParser;
import org.yah.tools.cuda.cudapi.PreprocessorParserBaseVisitor;

import java.io.IOException;
import java.nio.file.Path;
import java.util.Set;

// TODO handle condition without defined (see nvrtc.h@746)
public class CudaAPIPreprocessor {

    public static String preprocess(Path path, Set<String> defines) throws IOException {
        PreprocessorLexer lexer = new PreprocessorLexer(CharStreams.fromPath(path));
        TokenStream tokenStream = new CommonTokenStream(lexer);
        PreprocessorParser parser = new PreprocessorParser(tokenStream);
        PreprocessorParser.TranslationUnitContext translationUnitContext = parser.translationUnit();
        return translationUnitContext.accept(new Visitor(defines));
    }

    private static final class Visitor extends PreprocessorParserBaseVisitor<String> {
        private final Set<String> defines;
        private final StringBuilder output = new StringBuilder(1024 * 1024);

        public Visitor(Set<String> defines) {
            this.defines = defines;
        }

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
            boolean accept;
            String condition = ctx.condition().getText();
            if (ctx.IfDirective() != null)
                accept = evalCondition(condition);
            else if (ctx.IfdefDirective() != null)
                accept = defines.contains(condition);
            else if (ctx.IfndefDirective() != null)
                accept = !defines.contains(condition);
            else
                accept = true;

            PreprocessorParser.PartsContext ifParts = ctx.parts();
            PreprocessorParser.ElseDirectiveContext elseCtx = ctx.elseDirective();
            if (accept) {
                if (ifParts != null)
                    visitParts(ifParts);
            } else {
                if (elseCtx != null)
                    visitElseDirective(elseCtx);
            }
            return null;
        }

        @Override
        public String visitElseDirective(PreprocessorParser.ElseDirectiveContext ctx) {
            if (ctx.ElifDirective() != null) {
                boolean accept = evalCondition(ctx.condition().getText());
                if (accept) {
                    if (ctx.parts() != null)
                        visitParts(ctx.parts());
                } else {
                    if (ctx.elseDirective() != null)
                        visitElseDirective(ctx.elseDirective());
                }
            } else {
                if (ctx.parts() != null)
                    visitParts(ctx.parts());
            }
            return null;
        }

        private boolean evalCondition(String s) {
            ConditionLexer lexer = new ConditionLexer(CharStreams.fromString(s));
            TokenStream tokenStream = new CommonTokenStream(lexer);
            ConditionParser parser = new ConditionParser(tokenStream);
            ConditionParser.ExpressionsContext expressions = parser.expressions();
            return expressions.accept(new ConditionParserBaseVisitor<>() {
                @Override
                public Boolean visitExpressions(ConditionParser.ExpressionsContext ctx) {
                    if (ctx.expression() != null)
                        return ctx.expression().accept(this);
                    if (ctx.Or() != null)
                        return ctx.expressions(0).accept(this) || ctx.expressions(1).accept(this);
                    if (ctx.And() != null)
                        return ctx.expressions(0).accept(this) && ctx.expressions(1).accept(this);
                    return ctx.expressions(0).accept(this);
                }

                @Override
                public Boolean visitExpression(ConditionParser.ExpressionContext ctx) {
                    if (ctx.ignored() != null)
                        return true;
                    if (ctx.Defined() != null)
                        return defines.contains(ctx.Identifier().getText());
                    return !defines.contains(ctx.Identifier().getText());
                }
            });
        }

    }

}
