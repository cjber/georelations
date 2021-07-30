"""
This type stub file was generated by pyright.
"""

from dataclasses import dataclass
from enum import Enum
from typing import Any, Callable

TaskFunction = Callable[[Any], Any]
@dataclass
class TargetConf:
    """
    This class is going away in Hydra 1.2.
    You should no longer extend it or annotate with it.
    instantiate will work correctly if you pass in a DictConfig object or any dataclass that has the
    _target_ attribute.
    """
    _target_: str = ...
    def __post_init__(self) -> None:
        ...
    


class RunMode(Enum):
    RUN = ...
    MULTIRUN = ...


class ConvertMode(Enum):
    """ConvertMode for instantiate, controls return type.

    A config is either config or instance-like (`_target_` field).

    If instance-like, instantiate resolves the callable (class or
    function) and returns the result of the call on the rest of the
    parameters.

    If "none", config-like configs will be kept as is.

    If "partial", config-like configs will be converted to native python
    containers (list and dict), unless they are structured configs (
    dataclasses or attr instances).

    If "all", config-like configs will all be converted to native python
    containers (list and dict).
    """
    NONE = ...
    PARTIAL = ...
    ALL = ...
    def __eq__(self, other: Any) -> Any:
        ...
    

