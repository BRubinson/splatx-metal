"""SXT tensor writer — must stay binary-compatible with Sources/SplatXCore/SXTensor.cpp."""
import struct
import numpy as np

# dtype 2 is reserved for int64 (07 open question #1), 3=uint8 (the uchar kernel contract — valid/masks etc.)
_DTYPES = {np.dtype(np.float32): 0, np.dtype(np.int32): 1, np.dtype(np.uint8): 3}
_NP_FROM_CODE = {code: dt for dt, code in _DTYPES.items()}

def save_sxt(path: str, arr: np.ndarray) -> None:
    arr = np.ascontiguousarray(arr)
    if arr.dtype not in _DTYPES:
        raise ValueError(f"unsupported dtype {arr.dtype}")
    with open(path, "wb") as f:
        f.write(b"SXT1")
        f.write(struct.pack("<II", _DTYPES[arr.dtype], arr.ndim))
        f.write(struct.pack(f"<{arr.ndim}Q", *arr.shape))
        f.write(arr.tobytes())


def load_sxt(path: str) -> np.ndarray:
    """Inverse loader for save_sxt — for fixture chaining (feeding an existing golden output in as the next fixture's input)."""
    with open(path, "rb") as f:
        if f.read(4) != b"SXT1":
            raise ValueError(f"{path}: SXT1 magic mismatch")
        code, ndim = struct.unpack("<II", f.read(8))
        if code not in _NP_FROM_CODE:
            raise ValueError(f"{path}: unsupported dtype code {code}")
        shape = struct.unpack(f"<{ndim}Q", f.read(8 * ndim))
        arr = np.frombuffer(f.read(), dtype=_NP_FROM_CODE[code])
        if arr.size != int(np.prod(shape, dtype=np.int64)):
            raise ValueError(f"{path}: payload size does not match shape {shape}")
    return arr.reshape(shape).copy()  # frombuffer is read-only → writable copy
