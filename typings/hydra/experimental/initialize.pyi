"""
This type stub file was generated by pyright.
"""

from typing import Any, Optional

def get_gh_backup() -> Any:
    ...

def restore_gh_from_backup(_gh_backup: Any) -> Any:
    ...

class initialize:
    """
    Initializes Hydra and add the config_path to the config search path.
    config_path is relative to the parent of the caller.
    Hydra detects the caller type automatically at runtime.

    Supported callers:
    - Python scripts
    - Python modules
    - Unit tests
    - Jupyter notebooks.
    :param config_path: path relative to the parent of the caller
    :param job_name: the value for hydra.job.name (By default it is automatically detected based on the caller)
    :param caller_stack_depth: stack depth of the caller, defaults to 1 (direct caller).
    """
    def __init__(self, config_path: Optional[str] = ..., job_name: Optional[str] = ..., caller_stack_depth: int = ...) -> None:
        ...
    
    def __enter__(self, *args: Any, **kwargs: Any) -> None:
        ...
    
    def __exit__(self, exc_type: Any, exc_val: Any, exc_tb: Any) -> None:
        ...
    
    def __repr__(self) -> str:
        ...
    


class initialize_config_module:
    """
    Initializes Hydra and add the config_module to the config search path.
    The config module must be importable (an __init__.py must exist at its top level)
    :param config_module: absolute module name, for example "foo.bar.conf".
    :param job_name: the value for hydra.job.name (default is 'app')
    """
    def __init__(self, config_module: str, job_name: str = ...) -> None:
        ...
    
    def __enter__(self, *args: Any, **kwargs: Any) -> None:
        ...
    
    def __exit__(self, exc_type: Any, exc_val: Any, exc_tb: Any) -> None:
        ...
    
    def __repr__(self) -> str:
        ...
    


class initialize_config_dir:
    """
    Initializes Hydra and add an absolute config dir to the to the config search path.
    The config_dir is always a path on the file system and is must be an absolute path.
    Relative paths will result in an error.
    :param config_dir: absolute file system path
    :param job_name: the value for hydra.job.name (default is 'app')
    """
    def __init__(self, config_dir: str, job_name: str = ...) -> None:
        ...
    
    def __enter__(self, *args: Any, **kwargs: Any) -> None:
        ...
    
    def __exit__(self, exc_type: Any, exc_val: Any, exc_tb: Any) -> None:
        ...
    
    def __repr__(self) -> str:
        ...
    

