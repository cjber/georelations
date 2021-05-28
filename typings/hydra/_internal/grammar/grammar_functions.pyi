"""
This type stub file was generated by pyright.
"""

from typing import Any, Callable, Dict, List, Optional, Union
from hydra.core.override_parser.types import ChoiceSweep, Glob, IntervalSweep, ParsedElementType, RangeSweep, Sweep

ElementType = Union[str, int, bool, float, list, dict]
def apply_to_dict_values(value: Dict[Any, Any], function: Callable[..., Any]) -> Dict[Any, Any]:
    ...

def cast_choice(value: ChoiceSweep, function: Callable[..., Any]) -> ChoiceSweep:
    ...

def cast_interval(value: IntervalSweep, function: Callable[..., Any]) -> IntervalSweep:
    ...

def cast_range(value: RangeSweep, function: Callable[..., Any]) -> RangeSweep:
    ...

CastType = Union[ParsedElementType, Sweep]
def cast_int(*args: CastType, value: Optional[CastType] = ...) -> Any:
    ...

def cast_float(*args: CastType, value: Optional[CastType] = ...) -> Any:
    ...

def cast_str(*args: CastType, value: Optional[CastType] = ...) -> Any:
    ...

def cast_bool(*args: CastType, value: Optional[CastType] = ...) -> Any:
    ...

def choice(*args: Union[str, int, float, bool, Dict[Any, Any], List[Any], ChoiceSweep]) -> ChoiceSweep:
    """
    A choice sweep over the specified values
    """
    ...

def range(start: Union[int, float], stop: Union[int, float], step: Union[int, float] = ...) -> RangeSweep:
    """
    Range is defines a sweeep over a range of integer or floating-point values.
    For a positive step, the contents of a range r are determined by the formula
     r[i] = start + step*i where i >= 0 and r[i] < stop.
    For a negative step, the contents of the range are still determined by the formula
     r[i] = start + step*i, but the constraints are i >= 0 and r[i] > stop.
    """
    ...

def interval(start: Union[int, float], end: Union[int, float]) -> IntervalSweep:
    """
    A continuous interval between two floating point values.
    value=interval(x,y) is interpreted as x <= value < y
    """
    ...

def tag(*args: Union[str, Union[Sweep]], sweep: Optional[Sweep] = ...) -> Sweep:
    """
    Tags the sweep with a list of string tags.
    """
    ...

def shuffle(*args: Union[ElementType, ChoiceSweep, RangeSweep], sweep: Optional[Union[ChoiceSweep, RangeSweep]] = ..., list: Optional[List[Any]] = ...) -> Union[List[Any], ChoiceSweep, RangeSweep]:
    """
    Shuffle input list or sweep (does not support interval)
    """
    ...

def sort(*args: Union[ElementType, ChoiceSweep, RangeSweep], sweep: Optional[Union[ChoiceSweep, RangeSweep]] = ..., list: Optional[List[Any]] = ..., reverse: bool = ...) -> Any:
    """
    Sort an input list or sweep.
    reverse=True reverses the order
    """
    ...

def glob(include: Union[List[str], str], exclude: Optional[Union[List[str], str]] = ...) -> Glob:
    """
    A glob selects from all options in the config group.
    inputs are in glob format. e.g: *, foo*, *foo.
    :param include: a string or a list of strings to use as include globs
    :param exclude: a string or a list of strings to use as exclude globs
    """
    ...

