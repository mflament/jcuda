lexer grammar ConditionLexer;

NotDefined: '!defined';
Defined: 'defined';
LeftParen: '(';
RightParen: ')';
Identifier: [_a-zA-Z][_a-zA-Z0-9]*;
Or: '||' | 'OR';
And: '&&' | 'AND';
Whitespace: [ \t]+ -> skip;
Other: .+?;