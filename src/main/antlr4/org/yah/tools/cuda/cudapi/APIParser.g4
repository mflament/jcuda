parser grammar APIParser;

options {
    tokenVocab = APILexer;
}

translationUnit
    :  declarationseq? EOF
    ;

declarationseq
    :   ( defineDirective
        | declaration
        | comment
        | Newline
        | ignoredStatement)+
    ;

defineDirective
    : doc? DefineDirective defineName defineValue? comment? Newline
    ;
defineName: Identifier | Specifier;
defineValue
    : literalDefineValue
    | otherDefineValue
    ;
literalDefineValue: IntegerLiteral | FloatingLiteral;
otherDefineValue: ~(Newline | BlockComment | LineComment)+;

declaration
    : doc? ( typeDeclaration
           | functionDeclaration
           ) comment?
    ;

typeDeclaration
    : ( enumDeclaration
      | structDeclaration
      | typedefDeclaration
      ) Semi
    ;

functionDeclaration
    : functionSignature functionBody
    | functionSignature Semi
    ;
functionSignature
    : declSpecifiers? type declSpecifiers? Newline? Identifier LeftParen parameterDeclaration* RightParen
    ;
functionBody: openBlock (functionBodyStatement | functionBody)* closeBlock;
functionBodyStatement
    : ~(LeftBrace |  RightBrace)+?
    ;

declSpecifiers
    : Specifier+
    ;
parameterDeclaration
    : type Identifier arrayDecl? Comma? comment? Newline*
    | Void
    ;

enumDeclaration: Enum Identifier enumBody?;
enumDefinition: Enum Identifier? enumBody?;
enumBody: openBlock enumConstant+ closeBlock;
enumConstant: doc? Comma? Identifier enumConstantValue? Comma? comment? Newline*;
enumConstantValue: Assign (IntegerLiteral | defineName);

structDeclaration: Struct Identifier structBody?;
structDefinition: Struct Identifier? structBody?;
structBody: openBlock structField+ closeBlock;
structField: doc? type Identifier arrayDecl? Semi comment? Newline*;

typedefDeclaration: Typedef (type Identifier | functionPointer);

type: pointerType | noPointerType;
pointerType
    : noPointerType Star Const?
    | pointerType Star Const?
    ;
noPointerType
    : Const? ( primitiveType
             | structDefinition
             | enumDefinition
             | Void
             )
    ;

primitiveType: (Signed | Unsigned)? (PrimitiveType+ | Identifier);

arrayDecl: LeftBracket (IntegerLiteral | defineName)? RightBracket;

functionPointer: type LeftParen Specifier* Star Specifier* Identifier RightParen LeftParen functionPointerParameter* RightParen;
functionPointerParameter: type Identifier? Comma?;

ignoredStatement: ~Newline+;

openBlock: Newline* LeftBrace comment? Newline*;
closeBlock: Newline* RightBrace Newline*;

doc: comment Newline;
comment: BlockComment | LineComment;