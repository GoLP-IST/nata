# -*- coding: utf-8 -*-
from nata.backends.osiris.hdf5_grid import Osiris_Hdf5_GridFile
from nata.backends.osiris.hdf5_grid_dev import Osiris_Dev_Hdf5_GridFile
from nata.backends.osiris.zdf_grid import Osiris_zdf_GridFile
from nata.containers import GridDataset
from nata.types import GridBackendType


def test_Osiris_Hdf5_GridFile_isinstance_GridBackendType():
    assert isinstance(Osiris_Hdf5_GridFile, GridBackendType)


def test_Osiris_Dev_Hdf5_GridFile_isinstance_GridBackendType():
    assert isinstance(Osiris_Dev_Hdf5_GridFile, GridBackendType)


def test_Osiris_zdf_GridFile_isinstance_GridBackendType():
    assert isinstance(Osiris_zdf_GridFile, GridBackendType)


def test_backends_are_registered():
    backends = GridDataset.get_backends()

    # avoids name collisions
    assert backends[Osiris_Hdf5_GridFile.name] is Osiris_Hdf5_GridFile
    assert backends[Osiris_Dev_Hdf5_GridFile.name] is Osiris_Dev_Hdf5_GridFile
    assert backends[Osiris_zdf_GridFile.name] is Osiris_zdf_GridFile
