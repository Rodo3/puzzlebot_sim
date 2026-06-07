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


def _patch_numpy_core() -> None:
    """Permite cargar pickles guardados con numpy 2.x bajo numpy 1.x.

    numpy 2.x movió internals a numpy._core.*; numpy 1.x no tiene ese módulo.
    Se registran alias en sys.modules para que pickle los encuentre.
    """
    import sys
    import types
    import importlib
    import numpy as _np

    if hasattr(_np, '_core'):
        return  # ya es numpy 2.x, no hace falta

    _alias = sys.modules.setdefault('numpy._core', types.ModuleType('numpy._core'))
    for _sub in [
        'numeric', 'multiarray', 'umath', 'fromnumeric',
        'shape_base', 'function_base', '_methods',
        'arrayprint', 'defchararray', 'records',
    ]:
        _full = f'numpy._core.{_sub}'
        if _full not in sys.modules:
            try:
                _m = importlib.import_module(f'numpy.core.{_sub}')
                sys.modules[_full] = _m
                setattr(_alias, _sub, _m)
            except ImportError:
                pass


def load_pickle(path: Path) -> Any:
    """Deserialize a pickle file."""
    _patch_numpy_core()
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
