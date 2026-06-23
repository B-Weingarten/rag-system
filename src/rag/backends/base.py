from typing import Protocol, runtime_checkable


@runtime_checkable
class VectorBackend(Protocol):
    def build_index(self) -> None: ...
    def retrieve(self, query: str, k: int) -> list[dict]: ...
