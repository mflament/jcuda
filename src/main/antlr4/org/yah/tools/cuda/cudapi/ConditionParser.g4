parser grammar ConditionParser;

options {
    tokenVocab = ConditionLexer;
}

expressions
    : LeftParen expressions RightParen
    | expressions Or expressions
    | expressions And expressions
    | expression
    ;

expression
    : Defined LeftParen Identifier RightParen
    | NotDefined LeftParen Identifier RightParen
    | ignored
    ;

ignored: .+?;
