"""
This type stub file was generated by pyright.
"""

from typing import Optional, Sequence

class HydraException(Exception):
    ...


class CompactHydraException(HydraException):
    ...


class OverrideParseException(CompactHydraException):
    def __init__(self, override: str, message: str) -> None:
        ...
    


class InstantiationException(CompactHydraException):
    ...


class ConfigCompositionException(CompactHydraException):
    ...


class SearchPathException(CompactHydraException):
    ...


class MissingConfigException(IOError, ConfigCompositionException):
    def __init__(self, message: str, missing_cfg_file: Optional[str], options: Optional[Sequence[str]] = ...) -> None:
        ...
    

