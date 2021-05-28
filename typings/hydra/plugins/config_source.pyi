"""
This type stub file was generated by pyright.
"""

from abc import abstractmethod
from dataclasses import dataclass
from typing import Dict, List, Optional
from hydra.core.default_element import InputDefault
from hydra.errors import HydraException
from omegaconf import Container
from hydra.core.object_type import ObjectType
from hydra.plugins.plugin import Plugin

@dataclass
class ConfigResult:
    provider: str
    path: str
    config: Container
    header: Dict[str, Optional[str]]
    defaults_list: Optional[List[InputDefault]] = ...
    is_schema_source: bool = ...


class ConfigLoadError(HydraException, IOError):
    ...


class ConfigSource(Plugin):
    provider: str
    path: str
    def __init__(self, provider: str, path: str) -> None:
        ...
    
    @staticmethod
    @abstractmethod
    def scheme() -> str:
        """
        :return: the scheme for this config source, for example file:// or pkg://
        """
        ...
    
    @abstractmethod
    def load_config(self, config_path: str) -> ConfigResult:
        ...
    
    def exists(self, config_path: str) -> bool:
        ...
    
    @abstractmethod
    def is_group(self, config_path: str) -> bool:
        ...
    
    @abstractmethod
    def is_config(self, config_path: str) -> bool:
        ...
    
    @abstractmethod
    def available(self) -> bool:
        """
        :return: True is this config source is pointing to a valid location
        """
        ...
    
    def list(self, config_path: str, results_filter: Optional[ObjectType]) -> List[str]:
        """
        List items under the specified config path
        :param config_path: config path to list items in, examples: "", "foo", "foo/bar"
        :param results_filter: None for all, GROUP for groups only and CONFIG for configs only
        :return: a list of config or group identifiers (sorted and unique)
        """
        ...
    
    def __str__(self) -> str:
        ...
    
    def __repr__(self) -> str:
        ...
    
    def full_path(self) -> str:
        ...
    


