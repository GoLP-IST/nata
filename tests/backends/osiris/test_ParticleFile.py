# -*- coding: utf-8 -*-
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
