# -*- coding: utf-8 -*-
from nata.backends.osiris.hdf5 import Osiris_Dev_Hdf5_GridFile
from nata.backends.osiris.hdf5 import Osiris_Dev_Hdf5_ParticleFile
from nata.backends.osiris.hdf5 import Osiris_Hdf5_GridFile
from nata.backends.osiris.hdf5 import Osiris_Hdf5_ParticleFile
from nata.backends.osiris.zdf import Osiris_zdf_GridFile
from nata.backends.osiris.zdf import Osiris_zdf_ParticleFile
from nata.containers import GridDataset
from nata.containers import ParticleDataset
from nata.types import GridBackendType
from nata.types import ParticleBackendType


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
