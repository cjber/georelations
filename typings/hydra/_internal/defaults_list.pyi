"""
This type stub file was generated by pyright.
"""

from dataclasses import dataclass
from typing import Dict, List, Optional, Set, Union
from hydra._internal.config_repository import IConfigRepository
from hydra.core.default_element import DefaultsTreeNode, GroupDefault, InputDefault, ResultDefault
from hydra.core.override_parser.types import Override

cs = ...
@dataclass
class Deletion:
    name: Optional[str]
    used: bool = ...


@dataclass
class OverrideMetadata:
    external_override: bool
    containing_config_path: Optional[str] = ...
    used: bool = ...
    relative_key: Optional[str] = ...


@dataclass
class Overrides:
    override_choices: Dict[str, Optional[Union[str, List[str]]]]
    override_metadata: Dict[str, OverrideMetadata]
    append_group_defaults: List[GroupDefault]
    config_overrides: List[Override]
    known_choices: Dict[str, Optional[str]]
    known_choices_per_group: Dict[str, Set[str]]
    deletions: Dict[str, Deletion]
    def __init__(self, repo: IConfigRepository, overrides_list: List[Override]) -> None:
        ...
    
    def add_override(self, parent_config_path: str, default: GroupDefault) -> None:
        ...
    
    def is_overridden(self, default: InputDefault) -> bool:
        ...
    
    def override_default_option(self, default: GroupDefault) -> None:
        ...
    
    def ensure_overrides_used(self) -> None:
        ...
    
    def ensure_deletions_used(self) -> None:
        ...
    
    def set_known_choice(self, default: InputDefault) -> None:
        ...
    
    def is_deleted(self, default: InputDefault) -> bool:
        ...
    
    def delete(self, default: InputDefault) -> None:
        ...
    


@dataclass
class DefaultsList:
    defaults: List[ResultDefault]
    defaults_tree: DefaultsTreeNode
    config_overrides: List[Override]
    overrides: Overrides
    ...


def update_package_header(repo: IConfigRepository, node: InputDefault) -> None:
    ...

def ensure_no_duplicates_in_list(result: List[ResultDefault]) -> None:
    ...

def create_defaults_list(repo: IConfigRepository, config_name: Optional[str], overrides_list: List[Override], prepend_hydra: bool, skip_missing: bool) -> DefaultsList:
    """
    :param repo:
    :param config_name:
    :param overrides_list:
    :param prepend_hydra:
    :param skip_missing: True to skip config group with the value '???' and not fail on them. Useful when sweeping.
    :return:
    """
    ...

def config_not_found_error(repo: IConfigRepository, tree: DefaultsTreeNode) -> None:
    ...

