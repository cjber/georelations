"""
This type stub file was generated by pyright.
"""

import decimal
from dataclasses import dataclass
from enum import Enum
from random import shuffle
from typing import Any, Callable, Dict, Iterator, List, Optional, Set, Union
from hydra.core.config_loader import ConfigLoader

class Quote(Enum):
    single = ...
    double = ...


@dataclass(frozen=True)
class QuotedString:
    text: str
    quote: Quote
    def with_quotes(self) -> str:
        ...
    


@dataclass
class Sweep:
    tags: Set[str] = ...


@dataclass
class ChoiceSweep(Sweep):
    list: List[ParsedElementType] = ...
    simple_form: bool = ...
    shuffle: bool = ...


@dataclass
class FloatRange:
    start: Union[decimal.Decimal, float]
    stop: Union[decimal.Decimal, float]
    step: Union[decimal.Decimal, float]
    def __post_init__(self) -> None:
        ...
    
    def __iter__(self) -> Any:
        ...
    
    def __next__(self) -> float:
        ...
    


@dataclass
class RangeSweep(Sweep):
    """
    Discrete range of numbers
    """
    start: Optional[Union[int, float]] = ...
    stop: Optional[Union[int, float]] = ...
    step: Union[int, float] = ...
    shuffle: bool = ...
    def range(self) -> Union[range, FloatRange]:
        ...
    


@dataclass
class IntervalSweep(Sweep):
    start: Optional[float] = ...
    end: Optional[float] = ...
    def __eq__(self, other: Any) -> Any:
        ...
    


ElementType = Union[str, int, float, bool, List[Any], Dict[str, Any]]
ParsedElementType = Optional[Union[ElementType, QuotedString]]
TransformerType = Callable[[ParsedElementType], Any]
class OverrideType(Enum):
    CHANGE = ...
    ADD = ...
    FORCE_ADD = ...
    DEL = ...


class ValueType(Enum):
    ELEMENT = ...
    CHOICE_SWEEP = ...
    GLOB_CHOICE_SWEEP = ...
    SIMPLE_CHOICE_SWEEP = ...
    RANGE_SWEEP = ...
    INTERVAL_SWEEP = ...


@dataclass
class Key:
    key_or_group: str
    package: Optional[str] = ...


@dataclass
class Glob:
    include: List[str] = ...
    exclude: List[str] = ...
    def filter(self, names: List[str]) -> List[str]:
        ...
    


class Transformer:
    @staticmethod
    def identity(x: ParsedElementType) -> ParsedElementType:
        ...
    
    @staticmethod
    def str(x: ParsedElementType) -> str:
        ...
    
    @staticmethod
    def encode(x: ParsedElementType) -> ParsedElementType:
        ...
    


@dataclass
class Override:
    type: OverrideType
    key_or_group: str
    value_type: Optional[ValueType]
    _value: Union[ParsedElementType, ChoiceSweep, RangeSweep, IntervalSweep]
    package: Optional[str] = ...
    input_line: Optional[str] = ...
    config_loader: Optional[ConfigLoader] = ...
    def is_delete(self) -> bool:
        """
        :return: True if this override represents a deletion of a config value or config group option
        """
        ...
    
    def is_add(self) -> bool:
        """
        :return: True if this override represents an addition of a config value or config group option
        """
        ...
    
    def is_force_add(self) -> bool:
        """
        :return: True if this override represents a forced addition of a config value
        """
        ...
    
    def value(self) -> Optional[Union[ElementType, ChoiceSweep, RangeSweep, IntervalSweep]]:
        """
        :return: the value. replaces Quoted strings by regular strings
        """
        ...
    
    def sweep_iterator(self, transformer: TransformerType = ...) -> Iterator[ElementType]:
        """
        Converts CHOICE_SWEEP, SIMPLE_CHOICE_SWEEP, GLOB_CHOICE_SWEEP and
        RANGE_SWEEP to a List[Elements] that can be used in the value component
        of overrides (the part after the =). A transformer may be provided for
        converting each element to support the needs of different sweepers
        """
        ...
    
    def sweep_string_iterator(self) -> Iterator[str]:
        """
        Converts CHOICE_SWEEP, SIMPLE_CHOICE_SWEEP, GLOB_CHOICE_SWEEP and RANGE_SWEEP
        to a List of strings that can be used in the value component of overrides (the
        part after the =)
        """
        ...
    
    def is_sweep_override(self) -> bool:
        ...
    
    def is_choice_sweep(self) -> bool:
        ...
    
    def is_discrete_sweep(self) -> bool:
        """
        :return: true if this sweep can be enumerated
        """
        ...
    
    def is_range_sweep(self) -> bool:
        ...
    
    def is_interval_sweep(self) -> bool:
        ...
    
    def is_hydra_override(self) -> bool:
        ...
    
    def get_key_element(self) -> str:
        ...
    
    def get_value_string(self) -> str:
        """
        return the value component from the input as is (the part after the first =).
        """
        ...
    
    def get_value_element_as_str(self, space_after_sep: bool = ...) -> str:
        """
        Returns a string representation of the value in this override
        (similar to the part after the = in the input string)
        :param space_after_sep: True to append space after commas and colons
        :return:
        """
        ...
    
    def validate(self) -> None:
        ...
    


