"""F1.1 — Carga segura de archivos en la app legacy (utils.leer_dato)."""
import pickle
import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
import utils  # noqa: E402


class _Upload:
    def __init__(self, payload: bytes):
        self._b = payload

    def getvalue(self):
        return self._b


class _Evil:
    def __init__(self, marker):
        self.marker = marker

    def __reduce__(self):
        import os
        return (os.system, (f"touch {self.marker}",))


def test_rce_payload_is_rejected_and_not_executed(tmp_path):
    marker = tmp_path / "pwned"
    with pytest.raises(ValueError):
        utils.leer_dato(_Upload(pickle.dumps(_Evil(marker))))
    assert not marker.exists()


@pytest.mark.parametrize("path", sorted((ROOT / "dev" / "data_test").glob("*.pkl"))[:20])
def test_valid_demo_files_still_load(path):
    v = utils.leer_dato(_Upload(path.read_bytes()))
    assert isinstance(v, np.ndarray) and v.shape == (20,) and np.isfinite(v).all()


@pytest.mark.parametrize(
    "obj",
    [np.full(20, np.nan), np.arange(19.0), np.arange(21.0), np.zeros((1, 20)), [0.1] * 20, "hola", np.array(["a"] * 20)],
)
def test_invalid_inputs_raise_controlled_error(obj):
    with pytest.raises(ValueError):
        utils.leer_dato(_Upload(pickle.dumps(obj)))


def test_garbage_bytes_raise_controlled_error():
    with pytest.raises(ValueError):
        utils.leer_dato(_Upload(b"not a pickle"))
