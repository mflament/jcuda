parser grammar PreprocessorParser;

options {
    tokenVocab = PreprocessorLexer;
}

translationUnit
    :  parts? EOF
    ;

parts: part+;
part: ifDirective | line+;
line
    : MultiLineBlockComment Newline?
    | Whitespace* (Word Whitespace* | SingleLineBlockComment)+ endline?
    | Whitespace* endline
    ;
endline
    : (LineComment | SingleLineBlockComment | MultiLineBlockComment) Newline?
    | Newline;

ifDirective
    : Whitespace* ( IfDirective | IfdefDirective | IfndefDirective) Whitespace+ condition endline
        parts?
        elseDirective?
        endifDirective
    ;

elseDirective: Whitespace* (ElseDirective | ElifDirective condition) endline parts? elseDirective?;

condition: (Word | Whitespace)* Word;

endifDirective: Whitespace* EndifDirective Whitespace* endline;
