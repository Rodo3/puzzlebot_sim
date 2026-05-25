"""
Save and load trained model artifacts using pickle and JSON.
"""
import json
import pickle
from pathlib import Path
from typing import Any


def save_pickle(obj: Any, path: Path) -> None:
    """Serialize obj to a pickle file."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, 'wb') as f:
        pickle.dump(obj, f, protocol=pickle.HIGHEST_PROTOCOL)


def load_pickle(path: Path) -> Any:
    """Deserialize a pickle file."""
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Artifact not found: {path}")
    with open(path, 'rb') as f:
        return pickle.load(f)


def save_json(data: Any, path: Path) -> None:
    """Write data as pretty-printed JSON."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, 'w', encoding='utf-8') as f:
        json.dump(data, f, indent=2, ensure_ascii=False)


def load_json(path: Path) -> Any:
    """Load and parse a JSON file."""
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"JSON file not found: {path}")
    with open(path, 'r', encoding='utf-8') as f:
        return json.load(f)


def artifact_size_kb(path: Path) -> float:
    """Return file size in kilobytes."""
    return Path(path).stat().st_size / 1024.0
