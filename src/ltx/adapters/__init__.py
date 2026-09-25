from pathlib import Path

from .base import Adapter
from .ccua import CCUAAdapter


def make_adapter(name: str, root: Path) -> Adapter:
    mapping = {
        "ccua": CCUAAdapter,
    }
    if name not in mapping:
        raise ValueError(f"Unknown adapter: {name}")
    return mapping[name](root)
