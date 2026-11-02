import json
import subprocess
import sys

import numpy as np
import pytest

import weight_inspector as wi


@pytest.fixture
def arrays():
    return {"a": np.array([[1.0, 2.0], [3.0, 4.0], [0.0, 0.0]]), "b": np.eye(2)}


def test_product_and_output_slice(arrays):
    got = wi.analyze(
        arrays,
        {
            "operation": "values",
            "left": {"tensor": "a"},
            "right": {"tensor": "b"},
            "output_row_start": 1,
            "output_rows": 1,
        },
    )
    assert got["shape"] == [3, 2]
    assert got["slice"] == {"row_start": 1, "rows": 1, "col_start": 0, "cols": 2}
    assert got["values"] == [[3.0, 4.0]]


def test_slice_precedes_transpose(arrays):
    got = wi.analyze(
        arrays,
        {
            "operation": "values",
            "left": {
                "tensor": "a",
                "row_start": 0,
                "rows": 2,
                "col_start": 1,
                "cols": 1,
                "transpose": True,
            },
        },
    )
    assert got["shape"] == [1, 2]
    assert got["values"] == [[2.0, 4.0]]


def test_cosine_zero_rows(arrays):
    got = wi.analyze(arrays, {"operation": "cosine_rows", "left": {"tensor": "a"}})
    assert got["values"][2] == [0.0, 0.0, 0.0]
    assert got["values"][0][0] == pytest.approx(1)


def test_stats_svd_and_fft_dimensions(arrays):
    stats = wi.analyze(arrays, {"operation": "stats", "left": {"tensor": "a"}})
    assert stats["shape"] == [3, 2]
    assert stats["stats"]["finite"] is True
    assert wi.analyze(
        arrays, {"operation": "singular_values", "left": {"tensor": "a"}}
    )["shape"] == [2, 1]
    assert wi.analyze(
        arrays, {"operation": "left_singular_vectors", "left": {"tensor": "a"}}
    )["shape"] == [3, 2]
    assert wi.analyze(
        arrays, {"operation": "right_singular_vectors", "left": {"tensor": "a"}}
    )["shape"] == [2, 2]
    assert wi.analyze(arrays, {"operation": "fft_rows_abs", "left": {"tensor": "a"}})[
        "shape"
    ] == [3, 2]
    assert wi.analyze(
        arrays, {"operation": "fft_columns_abs", "left": {"tensor": "a"}}
    )["shape"] == [3, 2]


def test_catalog_generic(arrays):
    assert wi.analyze(arrays, {"operation": "catalog"})["tensors"]["a"]["shape"] == [
        3,
        2,
    ]


@pytest.mark.parametrize(
    "bad_request",
    [
        None,
        {},
        {"operation": "forward"},
        {"operation": "values", "left": {"tensor": "../secret"}},
        {"operation": "values", "left": {"tensor": "a", "row_start": -1}},
        {"operation": "values", "left": {"tensor": "a", "extra": 1}},
        {"operation": "values", "left": {"tensor": "a"}, "output_rows": 1000},
        {"operation": "values", "left": {"tensor": "a"}, "right": {"tensor": "a"}},
    ],
)
def test_rejects_bad_requests(arrays, bad_request):
    with pytest.raises(wi.InspectorError):
        wi.analyze(arrays, bad_request)


def test_rejects_nonfinite_and_pickle(tmp_path):
    with pytest.raises(wi.InspectorError):
        wi.analyze(
            {"x": np.array([[np.nan]])},
            {"operation": "values", "left": {"tensor": "x"}},
        )
    path = tmp_path / "object.npz"
    np.savez(path, x=np.array([{"bad": True}], dtype=object))
    proc = subprocess.run(
        [sys.executable, "weight_inspector.py", "--specimen", str(path)],
        input='{"operation":"catalog"}\n',
        text=True,
        capture_output=True,
        check=False,
    )
    answer = json.loads(proc.stdout)
    assert answer["ok"] is False
    assert str(tmp_path) not in proc.stdout
    assert "Traceback" not in proc.stdout


def test_cli_success(tmp_path):
    path = tmp_path / "weights.npz"
    np.savez(path, x=np.eye(2))
    proc = subprocess.run(
        [sys.executable, "weight_inspector.py", "--specimen", str(path)],
        input='{"operation":"catalog"}\n',
        text=True,
        capture_output=True,
        check=False,
    )
    assert proc.returncode == 0
    assert json.loads(proc.stdout)["ok"] is True
    assert str(tmp_path) not in proc.stdout
