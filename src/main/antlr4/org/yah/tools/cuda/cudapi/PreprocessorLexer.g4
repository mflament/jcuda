lexer grammar PreprocessorLexer;

// preprocessor directives
IfDirective : '#if';
IfdefDirective : '#ifdef';
IfndefDirective : '#ifndef';
ElseDirective : '#else';
ElifDirective : '#elif';
EndifDirective : '#endif';

SingleLineBlockComment: '/*' ~[\r\n]*? '*/';
MultiLineBlockComment: '/*' .*? '*/';
LineComment: '//' ~ [\r\n]*;
Whitespace: [ \t]+;
Newline: ('\r' '\n'? | '\n');
Word: ~[ \t\r\n]+;