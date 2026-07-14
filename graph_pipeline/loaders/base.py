from abc import ABC, abstractmethod


class DataLoader(ABC):
    @abstractmethod
    def load(self, path: str) -> list[dict]:
        """Load records from path. Always returns a flat list of dicts."""

    @abstractmethod
    def can_handle(self, path: str) -> bool:
        """Return True if this loader handles the given file."""
