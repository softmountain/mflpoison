from typing import Callable, Dict, Generic, Iterable, TypeVar


T = TypeVar("T")


class Registry(Generic[T]):
    """Small explicit registry used for generators, attacks, and defenses."""

    def __init__(self, kind: str):
        self.kind = str(kind)
        self._items: Dict[str, Callable[..., T]] = {}

    def register(self, name: str, factory: Callable[..., T], replace: bool = False):
        key = str(name).strip().lower()
        if not key:
            raise ValueError(f"{self.kind} name cannot be empty")
        if key in self._items and not replace:
            raise KeyError(f"{self.kind} already registered: {key}")
        self._items[key] = factory
        return factory

    def create(self, name: str, **kwargs) -> T:
        key = str(name).strip().lower()
        try:
            factory = self._items[key]
        except KeyError as exc:
            known = ", ".join(sorted(self._items)) or "<none>"
            raise KeyError(f"unknown {self.kind} '{key}'; available: {known}") from exc
        return factory(**kwargs)

    def names(self) -> Iterable[str]:
        return tuple(sorted(self._items))
