# -*- coding: utf-8 -*-
from pathlib import Path
from typing import Callable
from typing import Optional
from uuid import uuid4

import h5py as h5
import numpy as np
import pytest
from numpy.lib.recfunctions import rename_fields
from numpy.lib.recfunctions import unstructured_to_structured

from nata.backends.osiris.hdf5 import Osiris_Dev_Hdf5_ParticleFile
from nata.backends.osiris.hdf5 import Osiris_Hdf5_ParticleFile
from nata.containers.particle import ParticleDataReader


@pytest.fixture(name="particles_path")
def _particles_path(tmp_path: Path) -> Path:
    base_path = tmp_path / "particle_files"
    # ensures it exists
    base_path.mkdir(parents=True, exist_ok=True)
    return base_path


def make_osiris_444_particles_hdf(path: Path, data: np.ndarray, name: str):
    """
    Example particle HDF5 file generated by OSIRIS 4.4.4

    The associated data types are taken from an example output file.
    """
    # makes sure we have data with a 'charge'
    if "q" not in data.dtype.fields:
        raise ValueError("structured dataset with a field 'q' required")

    with h5.File(path, mode="w") as fp:
        # root attrs
        fp.attrs["NAME"] = np.array([name], dtype="|S256")
        fp.attrs["TYPE"] = np.array(["particles"], dtype="|S9")
        fp.attrs["ITER"] = np.array([12345], dtype="i4")
        fp.attrs["TIME"] = np.array([-321.9], dtype="f4")
        fp.attrs["TIME UNITS"] = np.array([b"time unit"], dtype="|S256")

        # other quantities
        for field in data.dtype.fields:
            d = data[field]
            quants = fp.create_dataset(field, data=d)
            quants.attrs["LONG_NAME"] = np.array([f"{field} label"], dtype="|S256")
            quants.attrs["UNITS"] = np.array([f"{field} unit"], dtype="|S256")

        # tags
        tags = np.arange(len(data) * 2, dtype="i4").reshape((len(data), 2))
        fp.create_dataset("tag", data=tags)


def make_osiris_dev_particles_hdf(path: Path, data: np.ndarray, name: str):
    """
    Example particle HDF5 file generated by the dev branch of OSIRIS (May 2021)

    The associated data types are taken from an example output file.
    """
    # makes sure we have data with a 'charge'
    if "q" not in data.dtype.fields:
        raise ValueError("structured dataset with a field 'q' required")

    with h5.File(path, mode="w") as fp:
        # root attrs
        fp.attrs["NAME"] = np.array([name], dtype="|S256")
        fp.attrs["TYPE"] = np.array(["particles"], dtype="|S9")
        fp.attrs["ITER"] = np.array([12345], dtype="i4")
        fp.attrs["TIME"] = np.array([-321.9], dtype="f4")
        fp.attrs["TIME UNITS"] = np.array([b"time unit"], dtype="|S256")

        data_fields = data.dtype.fields
        fp.attrs["QUANTS"] = np.array([str.encode(f) for f in data_fields])
        fp.attrs["LABELS"] = np.array([str.encode(f"{f} label") for f in data_fields])
        fp.attrs["UNITS"] = np.array([str.encode(f"{f} unit") for f in data_fields])

        # other quantities
        for field in data.dtype.fields:
            d = data[field]
            fp.create_dataset(field, data=d)


@pytest.fixture(name="make_prt_file")
def _make_prt_file(
    particles_path: Path,
) -> Callable[[str, np.ndarray], Path]:
    def make_file(file_type: str, data: np.ndarray, name: str = "test particles"):
        if not data.dtype.fields:
            raise ValueError("requires array with structured data")

        if not data.ndim == 1:
            raise ValueError("only 1d data is accepted")

        file_ = particles_path / f"{file_type}_{str(uuid4())[:8]}.h5"
        if file_type == "osiris_4.4.4_particles_hdf5":
            make_osiris_444_particles_hdf(file_, data, name)
        elif file_type == "osiris_dev_particles_hdf5":
            make_osiris_dev_particles_hdf(file_, data, name)
        else:
            raise ValueError("invlid file type provided")

        return file_

    return make_file


def test_Osiris_Hdf5_ParticleFile_isinstance_ParticleDataReader():
    assert isinstance(Osiris_Hdf5_ParticleFile, ParticleDataReader)


def test_Osiris_Hdf5_ParticleFile_is_valid_backend(
    make_prt_file: Callable[[str, np.ndarray, Optional[str]], Path]
):
    data = unstructured_to_structured(np.random.random((10, 4)))
    data = rename_fields(data, {"f0": "q"})

    prt_path = make_prt_file("osiris_4.4.4_particles_hdf5", data)

    assert Osiris_Hdf5_ParticleFile.is_valid_backend(prt_path)


def test_Osiris_Hdf5_ParticleFile_properties(
    make_prt_file: Callable[[str, np.ndarray, Optional[str]], Path]
):
    data = unstructured_to_structured(np.random.random((10, 4)))
    data = rename_fields(data, {"f0": "q"})

    prt_path = make_prt_file("osiris_4.4.4_particles_hdf5", data, name="some particles")
    backend = Osiris_Hdf5_ParticleFile(prt_path)

    assert backend.name == "osiris_4.4.4_particles_hdf5"
    assert backend.location == prt_path
    assert backend.dataset_name == "some_particles"
    assert backend.dataset_label == "some particles"

    assert backend.quantity_names == ["f1", "f2", "f3", "q"]
    assert backend.quantity_labels == [
        "f1 label",
        "f2 label",
        "f3 label",
        "q label",
    ]
    assert backend.quantity_units == [
        "f1 unit",
        "f2 unit",
        "f3 unit",
        "q unit",
    ]

    assert backend.shape == (10,)
    assert backend.dtype == np.dtype(
        [
            ("f1", float),
            ("f2", float),
            ("f3", float),
            ("q", float),
        ]
    )

    # taken function 'make_osiris_444_particles_hdf'
    assert backend.iteration == 12345
    assert np.isclose(backend.time_step, -321.9)
    assert backend.time_unit == "time unit"

    # check reading of data
    for indexing in (np.s_[0], np.s_[-1], np.s_[:], np.s_[3:7], np.s_[4:1]):
        expected_data = data[indexing]
        np.testing.assert_array_equal(backend.get_data((indexing,)), expected_data)


def test_Osiris_Dev_Hdf5_ParticleFile_isinstance_ParticleDataReader():
    assert isinstance(Osiris_Dev_Hdf5_ParticleFile, ParticleDataReader)


def test_Osiris_Dev_Hdf5_ParticleFile_is_valid_backend(
    make_prt_file: Callable[[str, np.ndarray, Optional[str]], Path]
):
    data = unstructured_to_structured(np.random.random((10, 4)))
    data = rename_fields(data, {"f0": "q"})

    prt_path = make_prt_file("osiris_dev_particles_hdf5", data)

    assert Osiris_Dev_Hdf5_ParticleFile.is_valid_backend(prt_path)


def test_Osiris_Dev_Hdf5_ParticleFile_properties(
    make_prt_file: Callable[[str, np.ndarray, Optional[str]], Path]
):
    data = unstructured_to_structured(np.random.random((10, 4)))
    data = rename_fields(data, {"f0": "q"})

    prt_path = make_prt_file("osiris_dev_particles_hdf5", data, name="some particles")
    backend = Osiris_Dev_Hdf5_ParticleFile(prt_path)

    assert backend.name == "osiris_dev_particles_hdf5"
    assert backend.location == prt_path
    assert backend.dataset_name == "some_particles"
    assert backend.dataset_label == "some particles"

    assert backend.quantity_names == ["f1", "f2", "f3", "q"]
    assert backend.quantity_labels == [
        "f1 label",
        "f2 label",
        "f3 label",
        "q label",
    ]
    assert backend.quantity_units == [
        "f1 unit",
        "f2 unit",
        "f3 unit",
        "q unit",
    ]

    assert backend.shape == (10,)
    assert backend.dtype == np.dtype(
        [
            ("f1", float),
            ("f2", float),
            ("f3", float),
            ("q", float),
        ]
    )

    # taken function 'make_osiris_444_particles_hdf'
    assert backend.iteration == 12345
    assert np.isclose(backend.time_step, -321.9)
    assert backend.time_unit == "time unit"

    # check reading of data
    for indexing in (np.s_[0], np.s_[-1], np.s_[:], np.s_[3:7], np.s_[4:1]):
        expected_data = data[indexing]
        np.testing.assert_array_equal(backend.get_data((indexing,)), expected_data)
