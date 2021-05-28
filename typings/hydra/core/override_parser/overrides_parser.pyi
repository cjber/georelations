"""
This type stub file was generated by pyright.
"""

from typing import Any, List, Optional
from hydra._internal.grammar.functions import Functions
from hydra.core.config_loader import ConfigLoader
from hydra.core.override_parser.types import Override

KEY_RULES = ...
class OverridesParser:
    functions: Functions
    @classmethod
    def create(cls, config_loader: Optional[ConfigLoader] = ...) -> OverridesParser:
        ...
    
    def __init__(self, functions: Functions, config_loader: Optional[ConfigLoader] = ...) -> None:
        ...
    
    def parse_rule(self, s: str, rule_name: str) -> Any:
        ...
    
    def parse_override(self, s: str) -> Override:
        ...
    
    def parse_overrides(self, overrides: List[str]) -> List[Override]:
        ...
    


def create_functions() -> Functions:
    ...

