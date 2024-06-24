lexer grammar APILexer;

// preprocessor directives
DefineDirective : '#define';
OtherDirective : '#' NONDIGIT+?;

BlockComment: '/*' .*? '*/';
LineComment: '//' ~ [\r\n]*;

Ignored
    : ( '__device_builtin__'
      | '__cudart_builtin__'
      | '__inline__'
      | '__host__'
      ) -> skip
    ;

Specifier
    : 'extern'
    | 'static'
    | 'inline'
    | '__CUDA_DEPRECATED'
    | 'CUDA_CB'
    | 'C' NONDIGIT+ 'API'
    ;

IntegerLiteral:
    SIGN? (DecimalLiteral Integersuffix?
          | OctalLiteral Integersuffix?
          | HexadecimalLiteral Integersuffix?
          | BinaryLiteral Integersuffix?
          )
    ;

FloatingLiteral:
    Fractionalconstant Exponentpart? Floatingsuffix?
    | Digitsequence Exponentpart Floatingsuffix?
    ;

fragment NONDIGIT: [a-zA-Z_];

fragment DIGIT: [0-9];

DecimalLiteral: NONZERODIGIT ('\''? DIGIT)*;

OctalLiteral: '0' ('\''? OCTALDIGIT)*;

HexadecimalLiteral: ('0x' | '0X') HEXADECIMALDIGIT ( '\''? HEXADECIMALDIGIT)*;

BinaryLiteral: ('0b' | '0B') BINARYDIGIT ('\''? BINARYDIGIT)*;


fragment NONZERODIGIT: [1-9];

fragment OCTALDIGIT: [0-7];

fragment HEXADECIMALDIGIT: [0-9a-fA-F];

fragment BINARYDIGIT: [01];

DV: '__dv(' [0-9a-fA-F]* ')' -> skip;

Integersuffix:
    Unsignedsuffix Longsuffix?
    | Unsignedsuffix Longlongsuffix?
    | Longsuffix Unsignedsuffix?
    | Longlongsuffix Unsignedsuffix?
    ;

fragment Unsignedsuffix: [uU];

fragment Longsuffix: [lL];

fragment Longlongsuffix: 'll' | 'LL';

fragment Fractionalconstant: Digitsequence? '.' Digitsequence | Digitsequence '.';

fragment Exponentpart: 'e' SIGN? Digitsequence | 'E' SIGN? Digitsequence;

fragment SIGN: [+-];

fragment Digitsequence: DIGIT ('\''? DIGIT)*;

fragment Floatingsuffix: [flFL];

Newline: ('\r' '\n'? | '\n');

Whitespace: [ \t]+ -> skip;

Const: 'const';

Signed: 'signed';

Unsigned: 'unsigned';

PrimitiveType
    : 'bool'
    | 'char'
    | 'short'
    | 'int'
    | 'long'
    | 'float'
    | 'double'
    ;

Void: 'void';

Typedef: 'typedef';

Enum: 'enum';

Struct: 'struct';

Union: 'union';

Template: 'template';

Typeid_: 'typeid';

Typename_: 'typename';

/*Operators*/

LeftParen: '(';

RightParen: ')';

LeftBracket: '[';

RightBracket: ']';

LeftBrace: '{';

RightBrace: '}';

Plus: '+';

Minus: '-';

Star: '*';

Div: '/';

Mod: '%';

Caret: '^';

And: '&';

Or: '|';

Tilde: '~';

Not: '!' | 'not';

Assign: '=';

Less: '<';

Greater: '>';

Equal: '==';

NotEqual: '!=';

LessEqual: '<=';

GreaterEqual: '>=';

AndAnd: '&&' | 'and';

OrOr: '||' | 'or';

Comma: ',';

Colon: ':';

Doublecolon: '::';

Semi: ';';

Dot: '.';

Ellipsis: '...';

Identifier: NONDIGIT (NONDIGIT | DIGIT)*;

Other: .;