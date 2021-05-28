"""
This type stub file was generated by pyright.
"""

import sys
from antlr4 import *
from typing import TextIO
from typing.io import TextIO

if sys.version_info[1] > 5:
    ...
else:
    ...
def serializedATN():
    ...

class OverrideParser(Parser):
    grammarFileName = ...
    atn = ...
    decisionsToDFA = ...
    sharedContextCache = ...
    literalNames = ...
    symbolicNames = ...
    RULE_override = ...
    RULE_key = ...
    RULE_packageOrGroup = ...
    RULE_package = ...
    RULE_value = ...
    RULE_element = ...
    RULE_simpleChoiceSweep = ...
    RULE_argName = ...
    RULE_function = ...
    RULE_listContainer = ...
    RULE_dictContainer = ...
    RULE_dictKeyValuePair = ...
    RULE_primitive = ...
    RULE_dictKey = ...
    ruleNames = ...
    EOF = ...
    EQUAL = ...
    TILDE = ...
    PLUS = ...
    AT = ...
    COLON = ...
    SLASH = ...
    DOT_PATH = ...
    POPEN = ...
    COMMA = ...
    PCLOSE = ...
    BRACKET_OPEN = ...
    BRACKET_CLOSE = ...
    BRACE_OPEN = ...
    BRACE_CLOSE = ...
    FLOAT = ...
    INT = ...
    BOOL = ...
    NULL = ...
    UNQUOTED_CHAR = ...
    ID = ...
    ESC = ...
    WS = ...
    QUOTED_VALUE = ...
    INTERPOLATION = ...
    def __init__(self, input: TokenStream, output: TextIO = ...) -> None:
        ...
    
    class OverrideContext(ParserRuleContext):
        def __init__(self, parser, parent: ParserRuleContext = ..., invokingState: int = ...) -> None:
            ...
        
        def EOF(self):
            ...
        
        def key(self):
            ...
        
        def EQUAL(self):
            ...
        
        def TILDE(self):
            ...
        
        def PLUS(self, i: int = ...):
            ...
        
        def value(self):
            ...
        
        def getRuleIndex(self):
            ...
        
        def enterRule(self, listener: ParseTreeListener):
            ...
        
        def exitRule(self, listener: ParseTreeListener):
            ...
        
        def accept(self, visitor: ParseTreeVisitor):
            ...
        
    
    
    def override(self):
        ...
    
    class KeyContext(ParserRuleContext):
        def __init__(self, parser, parent: ParserRuleContext = ..., invokingState: int = ...) -> None:
            ...
        
        def packageOrGroup(self):
            ...
        
        def AT(self):
            ...
        
        def package(self):
            ...
        
        def getRuleIndex(self):
            ...
        
        def enterRule(self, listener: ParseTreeListener):
            ...
        
        def exitRule(self, listener: ParseTreeListener):
            ...
        
        def accept(self, visitor: ParseTreeVisitor):
            ...
        
    
    
    def key(self):
        ...
    
    class PackageOrGroupContext(ParserRuleContext):
        def __init__(self, parser, parent: ParserRuleContext = ..., invokingState: int = ...) -> None:
            ...
        
        def package(self):
            ...
        
        def ID(self, i: int = ...):
            ...
        
        def SLASH(self, i: int = ...):
            ...
        
        def getRuleIndex(self):
            ...
        
        def enterRule(self, listener: ParseTreeListener):
            ...
        
        def exitRule(self, listener: ParseTreeListener):
            ...
        
        def accept(self, visitor: ParseTreeVisitor):
            ...
        
    
    
    def packageOrGroup(self):
        ...
    
    class PackageContext(ParserRuleContext):
        def __init__(self, parser, parent: ParserRuleContext = ..., invokingState: int = ...) -> None:
            ...
        
        def ID(self):
            ...
        
        def DOT_PATH(self):
            ...
        
        def getRuleIndex(self):
            ...
        
        def enterRule(self, listener: ParseTreeListener):
            ...
        
        def exitRule(self, listener: ParseTreeListener):
            ...
        
        def accept(self, visitor: ParseTreeVisitor):
            ...
        
    
    
    def package(self):
        ...
    
    class ValueContext(ParserRuleContext):
        def __init__(self, parser, parent: ParserRuleContext = ..., invokingState: int = ...) -> None:
            ...
        
        def element(self):
            ...
        
        def simpleChoiceSweep(self):
            ...
        
        def getRuleIndex(self):
            ...
        
        def enterRule(self, listener: ParseTreeListener):
            ...
        
        def exitRule(self, listener: ParseTreeListener):
            ...
        
        def accept(self, visitor: ParseTreeVisitor):
            ...
        
    
    
    def value(self):
        ...
    
    class ElementContext(ParserRuleContext):
        def __init__(self, parser, parent: ParserRuleContext = ..., invokingState: int = ...) -> None:
            ...
        
        def primitive(self):
            ...
        
        def listContainer(self):
            ...
        
        def dictContainer(self):
            ...
        
        def function(self):
            ...
        
        def getRuleIndex(self):
            ...
        
        def enterRule(self, listener: ParseTreeListener):
            ...
        
        def exitRule(self, listener: ParseTreeListener):
            ...
        
        def accept(self, visitor: ParseTreeVisitor):
            ...
        
    
    
    def element(self):
        ...
    
    class SimpleChoiceSweepContext(ParserRuleContext):
        def __init__(self, parser, parent: ParserRuleContext = ..., invokingState: int = ...) -> None:
            ...
        
        def element(self, i: int = ...):
            ...
        
        def COMMA(self, i: int = ...):
            ...
        
        def getRuleIndex(self):
            ...
        
        def enterRule(self, listener: ParseTreeListener):
            ...
        
        def exitRule(self, listener: ParseTreeListener):
            ...
        
        def accept(self, visitor: ParseTreeVisitor):
            ...
        
    
    
    def simpleChoiceSweep(self):
        ...
    
    class ArgNameContext(ParserRuleContext):
        def __init__(self, parser, parent: ParserRuleContext = ..., invokingState: int = ...) -> None:
            ...
        
        def ID(self):
            ...
        
        def EQUAL(self):
            ...
        
        def getRuleIndex(self):
            ...
        
        def enterRule(self, listener: ParseTreeListener):
            ...
        
        def exitRule(self, listener: ParseTreeListener):
            ...
        
        def accept(self, visitor: ParseTreeVisitor):
            ...
        
    
    
    def argName(self):
        ...
    
    class FunctionContext(ParserRuleContext):
        def __init__(self, parser, parent: ParserRuleContext = ..., invokingState: int = ...) -> None:
            ...
        
        def ID(self):
            ...
        
        def POPEN(self):
            ...
        
        def PCLOSE(self):
            ...
        
        def element(self, i: int = ...):
            ...
        
        def argName(self, i: int = ...):
            ...
        
        def COMMA(self, i: int = ...):
            ...
        
        def getRuleIndex(self):
            ...
        
        def enterRule(self, listener: ParseTreeListener):
            ...
        
        def exitRule(self, listener: ParseTreeListener):
            ...
        
        def accept(self, visitor: ParseTreeVisitor):
            ...
        
    
    
    def function(self):
        ...
    
    class ListContainerContext(ParserRuleContext):
        def __init__(self, parser, parent: ParserRuleContext = ..., invokingState: int = ...) -> None:
            ...
        
        def BRACKET_OPEN(self):
            ...
        
        def BRACKET_CLOSE(self):
            ...
        
        def element(self, i: int = ...):
            ...
        
        def COMMA(self, i: int = ...):
            ...
        
        def getRuleIndex(self):
            ...
        
        def enterRule(self, listener: ParseTreeListener):
            ...
        
        def exitRule(self, listener: ParseTreeListener):
            ...
        
        def accept(self, visitor: ParseTreeVisitor):
            ...
        
    
    
    def listContainer(self):
        ...
    
    class DictContainerContext(ParserRuleContext):
        def __init__(self, parser, parent: ParserRuleContext = ..., invokingState: int = ...) -> None:
            ...
        
        def BRACE_OPEN(self):
            ...
        
        def BRACE_CLOSE(self):
            ...
        
        def dictKeyValuePair(self, i: int = ...):
            ...
        
        def COMMA(self, i: int = ...):
            ...
        
        def getRuleIndex(self):
            ...
        
        def enterRule(self, listener: ParseTreeListener):
            ...
        
        def exitRule(self, listener: ParseTreeListener):
            ...
        
        def accept(self, visitor: ParseTreeVisitor):
            ...
        
    
    
    def dictContainer(self):
        ...
    
    class DictKeyValuePairContext(ParserRuleContext):
        def __init__(self, parser, parent: ParserRuleContext = ..., invokingState: int = ...) -> None:
            ...
        
        def dictKey(self):
            ...
        
        def COLON(self):
            ...
        
        def element(self):
            ...
        
        def getRuleIndex(self):
            ...
        
        def enterRule(self, listener: ParseTreeListener):
            ...
        
        def exitRule(self, listener: ParseTreeListener):
            ...
        
        def accept(self, visitor: ParseTreeVisitor):
            ...
        
    
    
    def dictKeyValuePair(self):
        ...
    
    class PrimitiveContext(ParserRuleContext):
        def __init__(self, parser, parent: ParserRuleContext = ..., invokingState: int = ...) -> None:
            ...
        
        def QUOTED_VALUE(self):
            ...
        
        def ID(self, i: int = ...):
            ...
        
        def NULL(self, i: int = ...):
            ...
        
        def INT(self, i: int = ...):
            ...
        
        def FLOAT(self, i: int = ...):
            ...
        
        def BOOL(self, i: int = ...):
            ...
        
        def INTERPOLATION(self, i: int = ...):
            ...
        
        def UNQUOTED_CHAR(self, i: int = ...):
            ...
        
        def COLON(self, i: int = ...):
            ...
        
        def ESC(self, i: int = ...):
            ...
        
        def WS(self, i: int = ...):
            ...
        
        def getRuleIndex(self):
            ...
        
        def enterRule(self, listener: ParseTreeListener):
            ...
        
        def exitRule(self, listener: ParseTreeListener):
            ...
        
        def accept(self, visitor: ParseTreeVisitor):
            ...
        
    
    
    def primitive(self):
        ...
    
    class DictKeyContext(ParserRuleContext):
        def __init__(self, parser, parent: ParserRuleContext = ..., invokingState: int = ...) -> None:
            ...
        
        def QUOTED_VALUE(self):
            ...
        
        def ID(self, i: int = ...):
            ...
        
        def NULL(self, i: int = ...):
            ...
        
        def INT(self, i: int = ...):
            ...
        
        def FLOAT(self, i: int = ...):
            ...
        
        def BOOL(self, i: int = ...):
            ...
        
        def UNQUOTED_CHAR(self, i: int = ...):
            ...
        
        def ESC(self, i: int = ...):
            ...
        
        def WS(self, i: int = ...):
            ...
        
        def getRuleIndex(self):
            ...
        
        def enterRule(self, listener: ParseTreeListener):
            ...
        
        def exitRule(self, listener: ParseTreeListener):
            ...
        
        def accept(self, visitor: ParseTreeVisitor):
            ...
        
    
    
    def dictKey(self):
        ...
    


