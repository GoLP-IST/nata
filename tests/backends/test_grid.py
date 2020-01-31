import pytest
import numpy as np

from nata.backends.grid import BaseGrid
from nata.backends.grid import GridArray


@pytest.fixture(name="reset_creation_index")
def _rest_array_creation_index(monkeypatch):
    monkeypatch.setattr("nata.backends.grid._ARRAY_CREATION_COUNT", 0)


def test_GridArray_init_default(reset_creation_index):
    array_backend = GridArray(array=np.arange(10, dtype=float))
    assert isinstance(array_backend, BaseGrid)
    assert array_backend.name == "GridArray"
    np.testing.assert_array_equal(array_backend.axes_names, ["x1"])
    np.testing.assert_array_equal(array_backend.axes_labels, ["x_1"])
    np.testing.assert_array_equal(array_backend.axes_min, [0.0])
    np.testing.assert_array_equal(array_backend.axes_max, [1.0])
    np.testing.assert_array_equal(array_backend.axes_units, [""])
    assert array_backend.iteration == 0
    assert array_backend.time_step == 0.0
    assert array_backend.time_unit == ""
    np.testing.assert_array_equal(array_backend.dataset, np.arange(10))
    assert array_backend.dataset_unit == ""
    assert array_backend.dataset_name == "array_1"
    assert array_backend.dataset_label == ""
    assert array_backend.dim == 1
    assert array_backend.shape == (10,)
    assert array_backend.dtype == np.float


def test_GridArray_init_kwargs(reset_creation_index):
    array_backend = GridArray(
        array=np.arange(10.0),
        dataset_name="custom_datasetname",
        dataset_label="custom datasetlabel",
        dataset_unit="custom datasetunit",
        iteration=10,
        time_step=123.0,
        time_unit="time unit",
        axes_names=["custom_x1"],
        axes_labels=["axes label"],
        axes_units=["axes units"],
        axes_min=[-42.0],
        axes_max=[12.0],
    )
    assert isinstance(array_backend, BaseGrid)
    assert array_backend.name == "GridArray"
    np.testing.assert_array_equal(array_backend.axes_names, ["custom_x1"])
    np.testing.assert_array_equal(array_backend.axes_labels, ["axes label"])
    np.testing.assert_array_equal(array_backend.axes_units, ["axes units"])
    np.testing.assert_array_equal(array_backend.axes_min, [-42.0])
    np.testing.assert_array_equal(array_backend.axes_max, [12.0])
    assert array_backend.iteration == 10
    assert array_backend.time_step == 123.0
    assert array_backend.time_unit == "time unit"
    np.testing.assert_array_equal(array_backend.dataset, np.arange(10))
    assert array_backend.dataset_name == "custom_datasetname"
    assert array_backend.dataset_label == "custom datasetlabel"
    assert array_backend.dataset_unit == "custom datasetunit"
    assert array_backend.dim == 1
    assert array_backend.shape == (10,)
    assert array_backend.dtype == np.float


def test_GridArray_init_reshaping():
    array_backend = GridArray(array=np.arange(10).reshape((1, 10)))
    np.testing.assert_array_equal(array_backend.dataset, np.arange(10))


def test_GridArray_init_automatic_naming(reset_creation_index):
    for i in range(10):
        backend = GridArray(array=np.arange(10))
        assert backend.dataset_name == f"array_{i+1}"

def test_GridArray_init_fix_creation_count(reset_creation_index):
    for i in range(10):
        backend = GridArray(array=np.arange(10), keep_creation_count=True)
        assert backend.dataset_name == f"array_0"

