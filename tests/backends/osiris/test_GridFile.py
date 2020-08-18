# -*- coding: utf-8 -*-
import h5py as h5
import numpy as np
import pytest

from nata.backends.osiris.hdf5 import Osiris_Dev_Hdf5_GridFile
from nata.backends.osiris.hdf5 import Osiris_Hdf5_GridFile
from nata.backends.osiris.zdf import Osiris_zdf_GridFile
from nata.containers import GridDataset
from nata.types import GridBackendType


def test_Osiris_Hdf5_GridFile_isinstance_GridBackendType():
    assert isinstance(Osiris_Hdf5_GridFile, GridBackendType)


def test_Osiris_Dev_Hdf5_GridFile_isinstance_GridBackendType():
    assert isinstance(Osiris_Dev_Hdf5_GridFile, GridBackendType)


def test_Osiris_zdf_GridFile_isinstance_GridBackendType():
    assert isinstance(Osiris_zdf_GridFile, GridBackendType)


def test_GridDatasets_backends_are_registered():
    backends = GridDataset.get_backends()

    assert backends[Osiris_Hdf5_GridFile.name] is Osiris_Hdf5_GridFile
    assert backends[Osiris_Dev_Hdf5_GridFile.name] is Osiris_Dev_Hdf5_GridFile
    assert backends[Osiris_zdf_GridFile.name] is Osiris_zdf_GridFile


@pytest.fixture(name="os_hdf5_grid_444_file", scope="session")
def _generate_valid_Osiris_Hdf5_GridFile(tmp_path_factory):
    """Fixture for valid HDF5 file for Osiris_Hdf5_GridFile backend"""
    tmp_path = tmp_path_factory.mktemp("os_hdf5_grid_444_fixture")
    file_ = tmp_path / "os_hdf5_grid_444_file.h5"
    dtype = np.dtype("f4")

    with h5.File(file_, mode="w") as fp:
        # root attrs
        fp.attrs["NAME"] = np.array([b"test ds"], dtype="|S256")
        fp.attrs["TYPE"] = np.array([b"grid"], dtype="|S4")
        fp.attrs["ITER"] = np.array([12345], dtype="i4")
        fp.attrs["TIME"] = np.array([-321.9], dtype="f4")
        fp.attrs["TIME UNITS"] = np.array([b"time unit"], dtype="|S256")

        # dataset
        # * osiris stores data like a fortran array
        data = np.transpose(np.arange(24, dtype=dtype).reshape((2, 3, 4)))

        ds = fp.create_dataset("test ds", data=data)
        ds.attrs["LONG_NAME"] = np.array(
            [b"\\Latex test dataset label"], dtype="|S256"
        )
        ds.attrs["UNITS"] = np.array([b"test dataset unit"], dtype="|S256")

        # axes
        axis1_data = np.array([-1, 2], dtype=dtype)
        axis2_data = np.array([-2, 3], dtype=dtype)
        axis3_data = np.array([-3, 4], dtype=dtype)

        axis_grp = fp.create_group("AXIS")
        axis1 = axis_grp.create_dataset("AXIS1", data=axis1_data)
        axis1.attrs["NAME"] = np.array([b"axis1 name"], dtype="|S256")
        axis1.attrs["LONG_NAME"] = np.array([b"axis_1"], dtype="|S256")
        axis1.attrs["UNITS"] = np.array([b"axis1 unit"], dtype="|S256")

        axis2 = axis_grp.create_dataset("AXIS2", data=axis2_data)
        axis2.attrs["NAME"] = np.array([b"axis2 name"], dtype="|S256")
        axis2.attrs["LONG_NAME"] = np.array([b"axis_2"], dtype="|S256")
        axis2.attrs["UNITS"] = np.array([b"axis2 unit"], dtype="|S256")

        axis3 = axis_grp.create_dataset("AXIS3", data=axis3_data)
        axis3.attrs["NAME"] = np.array([b"axis3 name"], dtype="|S256")
        axis3.attrs["LONG_NAME"] = np.array([b"axis_3"], dtype="|S256")
        axis3.attrs["UNITS"] = np.array([b"axis3 unit"], dtype="|S256")

    return file_


@pytest.mark.wip
def test_Osiris_Hdf5_GridFile_check_is_valid_backend(os_hdf5_grid_444_file):
    """Check 'Osiris_Hdf5_GridFile' is a valid backend exclusively"""
    assert Osiris_Hdf5_GridFile.is_valid_backend(os_hdf5_grid_444_file) is True

    # backend are registered automatically for GridDatasets
    for (name, backend) in GridDataset.get_backends().items():
        if name == Osiris_Hdf5_GridFile.name:
            continue

        assert backend.is_valid_backend(os_hdf5_grid_444_file) is False


@pytest.mark.wip
def test_Osiris_Hdf5_GridFile_dataset_props(os_hdf5_grid_444_file):
    backend = Osiris_Hdf5_GridFile(os_hdf5_grid_444_file)
    # TODO: check if dataset name is valid identifier
    assert backend.dataset_name == "test ds"
    assert backend.dataset_label == "\\Latex test dataset label"
    assert backend.dataset_unit == "test dataset unit"

    assert backend.shape == (2, 3, 4)
    assert backend.dtype == np.dtype("f4")
    assert backend.ndim == 3


@pytest.mark.wip
def test_Osiris_Hdf5_GridFile_grid_axes_props(os_hdf5_grid_444_file):
    backend = Osiris_Hdf5_GridFile(os_hdf5_grid_444_file)
    # TODO: check if axis names are valid identifier
    expected_names = ("axis1 name", "axis2 name", "axis3 name")
    expected_labels = ("axis_1", "axis_2", "axis_3")
    expected_units = ("axis1 unit", "axis2 unit", "axis3 unit")

    np.testing.assert_array_equal(backend.axes_names, expected_names)
    np.testing.assert_array_equal(backend.axes_labels, expected_labels)
    np.testing.assert_array_equal(backend.axes_units, expected_units)

    np.testing.assert_array_equal(backend.axes_min, [-1, -2, -3])
    np.testing.assert_array_equal(backend.axes_max, [2, 3, 4])


@pytest.mark.wip
def test_Osiris_Hdf5_GridFile_iteration_props(os_hdf5_grid_444_file):
    backend = Osiris_Hdf5_GridFile(os_hdf5_grid_444_file)
    assert backend.iteration == 12345


@pytest.mark.wip
def test_Osiris_Hdf5_GridFile_time_props(os_hdf5_grid_444_file):
    backend = Osiris_Hdf5_GridFile(os_hdf5_grid_444_file)
    np.testing.assert_allclose(backend.time_step, -321.9)
    assert backend.time_unit == "time unit"


@pytest.mark.wip
def test_Osiris_Hdf5_GridFile_reading_data(os_hdf5_grid_444_file):
    backend = Osiris_Hdf5_GridFile(os_hdf5_grid_444_file)
    full_array = np.arange(24, dtype="f4").reshape((2, 3, 4))

    # full array
    np.testing.assert_array_equal(backend.get_data(), full_array)

    # subarray
    index = np.s_[1, :2, ::2]
    np.testing.assert_array_equal(
        backend.get_data(indexing=index), full_array[index]
    )
