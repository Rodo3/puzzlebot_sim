"""
Save and load trained model artifacts using pickle.
Also handles JSON config/metadata files.

Implemented in Phase 3 (used by both model implementations).
"""
import json
import pickle
from pathlib import Path
from typing import Any


def save_pickle(obj: Any, path: Path) -> None:
    """Serialize obj to a pickle file. Implemented in Phase 3."""
    raise NotImplementedError("serialization.save_pickle — implemented in Phase 3")


def load_pickle(path: Path) -> Any:
    """Deserialize a pickle file. Implemented in Phase 3."""
    raise NotImplementedError("serialization.load_pickle — implemented in Phase 3")


def save_json(data: Any, path: Path) -> None:
    """Write data as pretty-printed JSON. Implemented in Phase 3."""
    raise NotImplementedError("serialization.save_json — implemented in Phase 3")


def load_json(path: Path) -> Any:
    """Load and parse a JSON file. Implemented in Phase 3."""
    raise NotImplementedError("serialization.load_json — implemented in Phase 3")


def artifact_size_kb(path: Path) -> float:
    """Return file size in kilobytes. Implemented in Phase 3."""
    raise NotImplementedError("serialization.artifact_size_kb — implemented in Phase 3")
