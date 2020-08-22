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
    tmp_path = tmp_path_factory.mktemp("os_hdf5_grid_444_fixture")
    file_ = tmp_path / "os_hdf5_particle_444_file.h5"
    dtype = np.dtype("f4")

    with h5.File(file_, mode="w") as fp:
        # root attrs
        fp.attrs["NAME"] = np.array([b"test ds"], dtype="|S256")
        fp.attrs["TYPE"] = np.array([b"particles"], dtype="|S9")

        # charge
        data_q = np.arange(13, dtype=dtype)
        fp.create_dataset("q", data=data_q)

    return file_


@pytest.mark.wip
def test_Osiris_Hdf5_ParticleFile_check_is_valid_backend(
    os_hdf5_particle_444_file,
):
    """Check 'Osiris_Hdf5_ParticleFile' is a valid backend exclusively"""
    assert (
        Osiris_Hdf5_ParticleFile.is_valid_backend(os_hdf5_particle_444_file)
        is True
    )

    # backend are registered automatically for GridDatasets
    for (name, backend) in ParticleDataset.get_backends().items():
        if name == Osiris_Hdf5_ParticleFile.name:
            continue

        assert backend.is_valid_backend(os_hdf5_particle_444_file) is False


@pytest.mark.wip
def test_Osiris_Hdf5_ParticleFile_dataset_props(os_hdf5_particle_444_file):
    """Check 'Osiris_Hdf5_ParticleFile' dataset properties"""
    backend = Osiris_Hdf5_ParticleFile(os_hdf5_particle_444_file)
    assert backend.dataset_name == "test ds"
    assert backend.num_particles == 13
