"""Predicate registry. Operators register via @predicate('name')."""
from typing import Callable

_REGISTRY: dict[str, Callable] = {}


def predicate(name: str):
    def deco(fn: Callable) -> Callable:
        _REGISTRY[name] = fn
        return fn
    return deco


def get(name: str) -> Callable:
    if name not in _REGISTRY:
        raise KeyError(f"Unknown predicate operator: {name}")
    return _REGISTRY[name]


def known() -> list[str]:
    return sorted(_REGISTRY.keys())
