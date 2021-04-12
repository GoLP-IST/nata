# -*- coding: utf-8 -*-
import h5py as h5
import numpy as np
import pytest

from nata.backends.osiris.hdf5 import Osiris_Dev_Hdf5_ParticleFile
from nata.backends.osiris.hdf5 import Osiris_Hdf5_ParticleFile
from nata.backends.osiris.zdf import Osiris_zdf_ParticleFile
from nata.containers import ParticleDataset
from nata.types import ParticleBackendType


def test_Osiris_Hdf5_ParticleFile_ParticleBackendType():
    assert isinstance(Osiris_Hdf5_ParticleFile, ParticleBackendType)


def test_Osiris_Dev_Hdf5_ParticleFile_ParticleBackendType():
    assert isinstance(Osiris_Dev_Hdf5_ParticleFile, ParticleBackendType)


def test_Osiris_zdf_ParticleFile_ParticleBackendType():
    assert isinstance(Osiris_zdf_ParticleFile, ParticleBackendType)


def test_ParticleDatasets_backends_are_registered():
    backends = ParticleDataset.get_backends()

    assert backends[Osiris_Hdf5_ParticleFile.name] is Osiris_Hdf5_ParticleFile
    assert (
        backends[Osiris_Dev_Hdf5_ParticleFile.name]
        is Osiris_Dev_Hdf5_ParticleFile
    )
    assert backends[Osiris_zdf_ParticleFile.name] is Osiris_zdf_ParticleFile


@pytest.fixture(name="os_hdf5_particle_444_file", scope="session")
def _generate_valid_Osiris_Hdf5_ParticleFile(tmp_path_factory):
    tmp_path = tmp_path_factory.mktemp("os_hdf5_particle_444_fixture")
    file_ = tmp_path / "os_hdf5_particle_444_file.h5"
    dtype = np.dtype("f4")

    with h5.File(file_, mode="w") as fp:
        # root attrs
        fp.attrs["NAME"] = np.array([b"test ds"], dtype="|S256")
        fp.attrs["TYPE"] = np.array([b"particles"], dtype="|S9")
        fp.attrs["ITER"] = np.array([12345], dtype="i4")
        fp.attrs["TIME"] = np.array([-321.9], dtype="f4")
        fp.attrs["TIME UNITS"] = np.array([b"time unit"], dtype="|S256")

        # charge
        data_q = np.arange(13, dtype=dtype)
        q = fp.create_dataset("q", data=data_q)
        q.attrs["LONG_NAME"] = np.array([b"q label"], dtype="|S256")
        q.attrs["UNITS"] = np.array([b"q unit"], dtype="|S256")

        # quant 1
        data_quant1 = np.arange(13, dtype=dtype) - 10
        quant1 = fp.create_dataset("quant1", data=data_quant1)
        quant1.attrs["LONG_NAME"] = np.array([b"quant1 label"], dtype="|S256")
        quant1.attrs["UNITS"] = np.array([b"quant1 unit"], dtype="|S256")

        # quant 2
        data_quant2 = np.arange(13, dtype=dtype) + 10
        quant2 = fp.create_dataset("quant2", data=data_quant2)
        quant2.attrs["LONG_NAME"] = np.array([b"quant2 label"], dtype="|S256")
        quant2.attrs["UNITS"] = np.array([b"quant2 unit"], dtype="|S256")

        # tags
        tags = np.arange(13 * 2, dtype="i4").reshape((13, 2))
        fp.create_dataset("tag", data=tags)

    return file_


def test_Osiris_Hdf5_ParticleFile_check_is_valid_backend(
    os_hdf5_particle_444_file,
):
    """Check 'Osiris_Hdf5_ParticleFile' is a valid backend exclusively"""
    assert (
        Osiris_Hdf5_ParticleFile.is_valid_backend(os_hdf5_particle_444_file)
        is True
    )

    # backend are registered automatically for ParticleDatasets
    for (name, backend) in ParticleDataset.get_backends().items():
        if name == Osiris_Hdf5_ParticleFile.name:
            continue

        assert backend.is_valid_backend(os_hdf5_particle_444_file) is False


def test_Osiris_Hdf5_ParticleFile_dataset_props(os_hdf5_particle_444_file):
    """Check 'Osiris_Hdf5_ParticleFile' dataset properties"""
    backend = Osiris_Hdf5_ParticleFile(os_hdf5_particle_444_file)
    assert backend.dataset_name == "test ds"
    assert backend.num_particles == 13
    assert backend.dtype == np.dtype(
        [(s, "f4") for s in ("q", "quant1", "quant2")]
    )


def test_Osiris_Hdf5_ParticleFile_quantity_props(os_hdf5_particle_444_file):
    """Check 'Osiris_Hdf5_ParticleFile' quantity properties"""
    backend = Osiris_Hdf5_ParticleFile(os_hdf5_particle_444_file)
    # TODO: check if quantity names are valid identifier
    expected_names = ("q", "quant1", "quant2")
    expected_labels = ("q label", "quant1 label", "quant2 label")
    expected_units = ("q unit", "quant1 unit", "quant2 unit")

    np.testing.assert_array_equal(backend.quantity_names, expected_names)
    np.testing.assert_array_equal(backend.quantity_labels, expected_labels)
    np.testing.assert_array_equal(backend.quantity_units, expected_units)


def test_Osiris_Hdf5_ParticleFile_iteration_props(os_hdf5_particle_444_file):
    """Check 'Osiris_Hdf5_ParticleFile' extraction of iteration"""
    backend = Osiris_Hdf5_ParticleFile(os_hdf5_particle_444_file)
    assert backend.iteration == 12345


def test_Osiris_Hdf5_ParticleFile_time_props(os_hdf5_particle_444_file):
    """Check 'Osiris_Hdf5_ParticleFile' extraction of time props"""
    backend = Osiris_Hdf5_ParticleFile(os_hdf5_particle_444_file)
    np.testing.assert_allclose(backend.time_step, -321.9)
    assert backend.time_unit == "time unit"


def test_Osiris_Hdf5_ParticleFile_reading_data(os_hdf5_particle_444_file):
    """Check 'Osiris_Hdf5_ParticleFile' reading array correctly"""
    backend = Osiris_Hdf5_ParticleFile(os_hdf5_particle_444_file)
    dtype = np.dtype([(quant, "f4") for quant in ("q", "quant1", "quant2")])

    full_array = np.zeros(13, dtype=dtype)
    full_array["q"] = np.arange(13, dtype="f4")
    full_array["quant1"] = np.arange(13, dtype="f4") - 10
    full_array["quant2"] = np.arange(13, dtype="f4") + 10

    # full data
    np.testing.assert_array_equal(backend.get_data(), full_array)

    # --- subdata ---
    # select every 2nd particle
    index = np.s_[::2]
    np.testing.assert_array_equal(
        backend.get_data(indexing=index), full_array[index]
    )
    # select two quantities
    np.testing.assert_array_equal(
        backend.get_data(fields=["quant1", "quant2"]),
        full_array[["quant1", "quant2"]],
    )
    # select one quantity
    np.testing.assert_array_equal(
        backend.get_data(fields="quant1"), full_array["quant1"],
    )
    # select one quantity and every 3rd particle
    np.testing.assert_array_equal(
        backend.get_data(indexing=index, fields="quant1"),
        full_array["quant1"][index],
    )


@pytest.fixture(name="os_hdf5_particle_dev_file", scope="session")
def _generate_valid_Osiris_Dev_Hdf5_ParticleFile(tmp_path_factory):
    tmp_path = tmp_path_factory.mktemp("os_hdf5_particle_dev_fixture")
    file_ = tmp_path / "os_hdf5_particle_dev_file.h5"
    dtype = np.dtype("f4")

    with h5.File(file_, mode="w") as fp:
        # root attrs
        fp.attrs["NAME"] = np.array([b"test ds"], dtype="|S256")
        fp.attrs["TYPE"] = np.array([b"particles"], dtype="|S9")
        fp.attrs["QUANTS"] = np.array(
            [b"q", b"quant1", b"quant2"], dtype="|S256"
        )
        fp.attrs["LABELS"] = np.array(
            [b"q label", b"quant1 label", b"quant2 label"], dtype="|S256"
        )
        fp.attrs["UNITS"] = np.array(
            [b"q unit", b"quant1 unit", b"quant2 unit"], dtype="|S256"
        )
        fp.attrs["ITER"] = np.array([12345], dtype="i4")
        fp.attrs["TIME"] = np.array([-321.9], dtype="f4")
        fp.attrs["TIME UNITS"] = np.array([b"time unit"], dtype="|S256")

        # charge
        data_q = np.arange(13, dtype=dtype)
        fp.create_dataset("q", data=data_q)

        # # quant 1
        data_quant1 = np.arange(13, dtype=dtype) - 10
        fp.create_dataset("quant1", data=data_quant1)

        # # quant 2
        data_quant2 = np.arange(13, dtype=dtype) + 10
        fp.create_dataset("quant2", data=data_quant2)

        # tags
        tags = np.arange(13 * 2, dtype="i4").reshape((13, 2))
        fp.create_dataset("tag", data=tags)

    return file_


def test_Osiris_Dev_Hdf5_ParticleFile_check_is_valid_backend(
    os_hdf5_particle_dev_file,
):
    """Check 'Osiris_Dev_Hdf5_ParticleFile' is a valid backend exclusively"""
    assert (
        Osiris_Dev_Hdf5_ParticleFile.is_valid_backend(os_hdf5_particle_dev_file)
        is True
    )

    # backend are registered automatically for ParticleDatasets
    for (name, backend) in ParticleDataset.get_backends().items():
        if name == Osiris_Dev_Hdf5_ParticleFile.name:
            continue

        assert backend.is_valid_backend(os_hdf5_particle_dev_file) is False


def test_Osiris_Dev_Hdf5_ParticleFile_dataset_props(os_hdf5_particle_dev_file):
    """Check 'Osiris_Dev_Hdf5_ParticleFile' dataset properties"""
    backend = Osiris_Dev_Hdf5_ParticleFile(os_hdf5_particle_dev_file)
    assert backend.dataset_name == "test ds"
    assert backend.num_particles == 13
    assert backend.dtype == np.dtype(
        [(s, "f4") for s in ("q", "quant1", "quant2")]
    )


def test_Osiris_Dev_Hdf5_ParticleFile_quantity_props(os_hdf5_particle_dev_file):
    """Check 'Osiris_Dev_Hdf5_ParticleFile' quantity properties"""
    backend = Osiris_Dev_Hdf5_ParticleFile(os_hdf5_particle_dev_file)
    # TODO: check if quantity names are valid identifier
    expected_names = ("q", "quant1", "quant2")
    expected_labels = ("q label", "quant1 label", "quant2 label")
    expected_units = ("q unit", "quant1 unit", "quant2 unit")

    np.testing.assert_array_equal(backend.quantity_names, expected_names)
    np.testing.assert_array_equal(backend.quantity_labels, expected_labels)
    np.testing.assert_array_equal(backend.quantity_units, expected_units)


def test_Osiris_Dev_Hdf5_ParticleFile_iteration_props(
    os_hdf5_particle_dev_file,
):
    """Check 'Osiris_Dev_Hdf5_ParticleFile' extraction of iteration"""
    backend = Osiris_Dev_Hdf5_ParticleFile(os_hdf5_particle_dev_file)
    assert backend.iteration == 12345


def test_Osiris_Dev_Hdf5_ParticleFile_time_props(os_hdf5_particle_dev_file):
    """Check 'Osiris_Dev_Hdf5_ParticleFile' extraction of time props"""
    backend = Osiris_Dev_Hdf5_ParticleFile(os_hdf5_particle_dev_file)
    np.testing.assert_allclose(backend.time_step, -321.9)
    assert backend.time_unit == "time unit"


def test_Osiris_Dev_Hdf5_ParticleFile_reading_data(os_hdf5_particle_dev_file):
    """Check 'Osiris_Hdf5_ParticleFile' reading array correctly"""
    backend = Osiris_Dev_Hdf5_ParticleFile(os_hdf5_particle_dev_file)
    dtype = np.dtype([(quant, "f4") for quant in ("q", "quant1", "quant2")])

    full_array = np.zeros(13, dtype=dtype)
    full_array["q"] = np.arange(13, dtype="f4")
    full_array["quant1"] = np.arange(13, dtype="f4") - 10
    full_array["quant2"] = np.arange(13, dtype="f4") + 10

    # full data
    np.testing.assert_array_equal(backend.get_data(), full_array)

    # --- subdata ---
    # select every 2nd particle
    index = np.s_[::2]
    np.testing.assert_array_equal(
        backend.get_data(indexing=index), full_array[index]
    )
    # select two quantities
    np.testing.assert_array_equal(
        backend.get_data(fields=["quant1", "quant2"]),
        full_array[["quant1", "quant2"]],
    )
    # select one quantity
    np.testing.assert_array_equal(
        backend.get_data(fields="quant1"), full_array["quant1"],
    )
    # select one quantity and every 3rd particle
    np.testing.assert_array_equal(
        backend.get_data(indexing=index, fields="quant1"),
        full_array["quant1"][index],
    )
