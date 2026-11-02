"""Bounded numerical inspection of tensors in an NPZ specimen."""

from __future__ import annotations

import json
import sys
from typing import Any
from zipfile import BadZipFile

import numpy as np

MAX_DIM = 512
MAX_VALUES = 384
MAX_OUTPUT = 12000
OPERATIONS = {
    "values",
    "stats",
    "row_norms",
    "column_norms",
    "cosine_rows",
    "cosine_columns",
    "singular_values",
    "left_singular_vectors",
    "right_singular_vectors",
    "fft_rows_abs",
    "fft_columns_abs",
}


class InspectorError(Exception):
    """A safe request or specimen error."""


def _integer(value: Any, label: str, default: int | None = None) -> int:
    if value is None and default is not None:
        return default
    if type(value) is not int or value < 0:
        raise InspectorError(f"invalid {label}")
    return value


def _tensor(arrays: Any, spec: Any) -> np.ndarray:
    if not isinstance(spec, dict) or set(spec) - {
        "tensor",
        "row_start",
        "rows",
        "col_start",
        "cols",
        "transpose",
    }:
        raise InspectorError("invalid tensor selection")
    name = spec.get("tensor")
    if not isinstance(name, str) or not name or name not in arrays:
        raise InspectorError("unknown tensor")
    arr = arrays[name]
    if arr.ndim not in (1, 2) or arr.size > MAX_DIM * MAX_DIM:
        raise InspectorError("unsupported tensor dimensions")
    if not np.issubdtype(arr.dtype, np.number) or not np.isfinite(arr).all():
        raise InspectorError("tensor must contain finite numeric values")
    if arr.ndim == 1:
        arr = arr.reshape(1, -1)
    r0 = _integer(spec.get("row_start"), "row start", 0)
    c0 = _integer(spec.get("col_start"), "column start", 0)
    nr = _integer(spec.get("rows"), "row count", arr.shape[0] - r0)
    nc = _integer(spec.get("cols"), "column count", arr.shape[1] - c0)
    if (
        r0 > arr.shape[0]
        or c0 > arr.shape[1]
        or r0 + nr > arr.shape[0]
        or c0 + nc > arr.shape[1]
    ):
        raise InspectorError("tensor slice out of bounds")
    if type(spec.get("transpose", False)) is not bool:
        raise InspectorError("invalid transpose")
    result = arr[r0 : r0 + nr, c0 : c0 + nc]
    return result.T if spec.get("transpose", False) else result


def _catalog(arrays: Any) -> dict[str, Any]:
    tensors = {}
    for name in arrays.files if hasattr(arrays, "files") else arrays.keys():
        a = arrays[name]
        if (
            a.ndim not in (1, 2)
            or a.size > MAX_DIM * MAX_DIM
            or not np.issubdtype(a.dtype, np.number)
            or not np.isfinite(a).all()
        ):
            raise InspectorError("specimen contains unsupported tensor")
        tensors[name] = {
            "shape": list(a.shape),
            "finite": True,
            "min": float(a.min()) if a.size else None,
            "max": float(a.max()) if a.size else None,
            "mean": float(a.mean()) if a.size else None,
            "std": float(a.std()) if a.size else None,
            "norm": float(np.linalg.norm(a)),
        }
    return {"tensors": tensors}


def analyze(arrays: Any, request: Any) -> dict[str, Any]:
    if not isinstance(request, dict) or not isinstance(request.get("operation"), str):
        raise InspectorError("invalid request")
    op = request["operation"]
    if op == "catalog":
        if set(request) != {"operation"}:
            raise InspectorError("unsupported request field")
        return _catalog(arrays)
    allowed = {
        "operation",
        "left",
        "right",
        "output_row_start",
        "output_rows",
        "output_col_start",
        "output_cols",
    }
    if op not in OPERATIONS or set(request) - allowed or "left" not in request:
        raise InspectorError("unsupported request")
    left = _tensor(arrays, request["left"])
    if "right" in request:
        right = _tensor(arrays, request["right"])
        if left.shape[1] != right.shape[0]:
            raise InspectorError("matrix dimensions do not match")
        if left.shape[0] > MAX_DIM or right.shape[1] > MAX_DIM:
            raise InspectorError("result dimensions exceed limit")
        matrix = left @ right
    else:
        matrix = left
    if matrix.ndim != 2 or max(matrix.shape, default=0) > MAX_DIM:
        raise InspectorError("matrix dimensions exceed limit")
    if op == "values":
        result = matrix
    elif op == "stats":
        return {
            "shape": list(matrix.shape),
            "stats": {
                "finite": bool(np.isfinite(matrix).all()),
                "min": float(matrix.min()) if matrix.size else None,
                "max": float(matrix.max()) if matrix.size else None,
                "mean": float(matrix.mean()) if matrix.size else None,
                "std": float(matrix.std()) if matrix.size else None,
                "norm": float(np.linalg.norm(matrix)),
            },
        }
    elif op in ("row_norms", "column_norms"):
        result = np.linalg.norm(matrix, axis=1 if op == "row_norms" else 0).reshape(
            -1, 1
        )
    elif op in ("cosine_rows", "cosine_columns"):
        x = matrix if op == "cosine_rows" else matrix.T
        norms = np.linalg.norm(x, axis=1)
        denom = norms[:, None] * norms[None, :]
        result = np.divide(
            x @ x.T, denom, out=np.zeros((len(x), len(x))), where=denom != 0
        )
    elif op == "singular_values":
        result = np.linalg.svd(matrix, compute_uv=False).reshape(-1, 1)
    elif op == "left_singular_vectors":
        result = np.linalg.svd(matrix, full_matrices=False)[0]
    elif op == "right_singular_vectors":
        result = np.linalg.svd(matrix, full_matrices=False)[2]
    elif op == "fft_rows_abs":
        result = np.abs(np.fft.fft(matrix, axis=1))
    else:
        result = np.abs(np.fft.fft(matrix, axis=0))
    result = np.asarray(result)
    if result.ndim == 1:
        result = result.reshape(-1, 1)
    shape = list(result.shape)
    r0 = _integer(request.get("output_row_start"), "output row start", 0)
    c0 = _integer(request.get("output_col_start"), "output column start", 0)
    nr = _integer(request.get("output_rows"), "output row count", 8)
    nc = _integer(request.get("output_cols"), "output column count", 8)
    if r0 > shape[0] or c0 > shape[1]:
        raise InspectorError("output slice out of bounds")
    if nr * nc > MAX_VALUES:
        raise InspectorError("output selection exceeds limit")
    view = result[r0 : min(shape[0], r0 + nr), c0 : min(shape[1], c0 + nc)]
    if view.size > MAX_VALUES:
        raise InspectorError("output selection exceeds limit")
    payload = {
        "shape": shape,
        "slice": {
            "row_start": r0,
            "rows": view.shape[0],
            "col_start": c0,
            "cols": view.shape[1],
        },
        "values": view.tolist(),
    }
    return payload


def main() -> int:
    try:
        import argparse

        parser = argparse.ArgumentParser(add_help=True)
        parser.add_argument("--specimen", required=True)
        args = parser.parse_args()
        request = json.loads(sys.stdin.readline())
        with np.load(args.specimen, allow_pickle=False) as arrays:
            response = analyze(arrays, request)
        encoded = json.dumps(
            {"ok": True, "result": response}, allow_nan=False, separators=(",", ":")
        )
        if len(encoded) > MAX_OUTPUT:
            raise InspectorError("response exceeds size limit")
    except InspectorError as exc:
        encoded = json.dumps({"ok": False, "error": str(exc)}, separators=(",", ":"))
    except (
        OSError,
        ValueError,
        TypeError,
        KeyError,
        IndexError,
        OverflowError,
        BadZipFile,
        np.linalg.LinAlgError,
    ):
        encoded = json.dumps(
            {"ok": False, "error": "invalid specimen or request"}, separators=(",", ":")
        )
    print(encoded)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
