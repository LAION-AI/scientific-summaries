"""Registry for dataset transformer functions with automatic module discovery."""

from __future__ import annotations

import importlib
import pkgutil
from types import ModuleType
from typing import Callable


BatchType = dict[str, list[object]]
TransformResult = dict[str, list[str]]
TransformerFn = Callable[[BatchType, str], TransformResult]

_TRANSFORMER_REGISTRY: dict[str, TransformerFn] = {}
_DISCOVERED_PACKAGES: set[str] = set()


def register_transformer(repo_id: str) -> Callable[[TransformerFn], TransformerFn]:
    """Decorator to register a dataset transformer for a repository ID."""
    def decorator(func: TransformerFn) -> TransformerFn:
        if repo_id in _TRANSFORMER_REGISTRY:
            raise ValueError(f"Transformer already registered for repo_id={repo_id}")
        _TRANSFORMER_REGISTRY[repo_id] = func
        return func

    return decorator


def get_transformer(repo_id: str) -> TransformerFn:
    """Retrieve a registered transformer by repository ID."""
    try:
        return _TRANSFORMER_REGISTRY[repo_id]
    except KeyError as exc:
        available_repos = ", ".join(sorted(_TRANSFORMER_REGISTRY))
        raise ValueError(
            f"Unsupported source repo: {repo_id}. Registered repos: [{available_repos}]"
        ) from exc


def discover_transformers(package_name: str = "transformers") -> None:
    """Import all modules from a transformer package to trigger registration.

    This enables an extension model where adding a new Python module inside the
    transformer package is enough to make the transformer discoverable by the pipeline.
    """
    if package_name in _DISCOVERED_PACKAGES:
        return

    package = importlib.import_module(package_name)

    if not hasattr(package, "__path__"):
        raise ValueError(f"Package '{package_name}' is not a valid Python package.")

    for module_info in pkgutil.iter_modules(package.__path__):
        if module_info.ispkg:
            continue
        importlib.import_module(f"{package_name}.{module_info.name}")

    _DISCOVERED_PACKAGES.add(package_name)


def registered_repo_ids() -> tuple[str, ...]:
    """Return all currently registered repository IDs."""
    return tuple(sorted(_TRANSFORMER_REGISTRY))