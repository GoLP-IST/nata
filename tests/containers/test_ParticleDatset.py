# -*- coding: utf-8 -*-
from pathlib import Path
from typing import Mapping
from typing import Union

import numpy as np
import pytest

from nata.containers import ParticleDataset
from nata.types import AxisType
from nata.types import ParticleBackendType
from nata.types import ParticleDatasetType


@pytest.fixture(name="TestParticleBackend")
def _TestParticleBackend():
    """Fixture for returning ParticleBackend"""

    class TestParticleBackend:
        """Test backend for ParticleDatasets"""

        name = "TestParticleBackend"
        location = None

        dataset_name = "test_dataset_name"
        num_particles = 10

        quantity_names = ("quant1", "quant2", "quant3")
        quantity_labels = ("quant1 label", "quant2 label", "quant3 label")
        quantity_units = ("quant1 unit", "quant2 unit", "quant3 unit")

        iteration = 42

        time_step = 12.3
        time_unit = "time unit"

        dtype = np.dtype(
            [("quant1", np.int), ("quant2", np.float), ("quant3", np.int)]
        )

        @staticmethod
        def is_valid_backend(location: Union[Path, str]) -> bool:
            return (
                True
                if Path(location) == Path("TestParticleBackend_location")
                else False
            )

        def __init__(self, location) -> None:
            pass

    # makes sure dummy backend is of valid type
    assert isinstance(TestParticleBackend, ParticleBackendType)

    ParticleDataset.add_backend(TestParticleBackend)
    yield TestParticleBackend
    # teardown
    ParticleDataset.remove_backend(TestParticleBackend)


def test_ParticleDataset_isinstance_ParticleDatasetType():
    """Checks ParticleDataset passes isinstance test of ParticleDatasetType"""
    assert isinstance(ParticleDataset, ParticleDatasetType)


def test_ParticleDataset_registration(TestParticleBackend):
    """Check if fixture registers backend properly"""
    assert TestParticleBackend.name in ParticleDataset.get_backends()


@pytest.mark.parametrize(
    "attr, value",
    [("backend", "TestParticleBackend"), ("name", "test_dataset_name")],
    ids=["backend", "name"],
)
def test_ParticleDataset_attr_propagation_from_Backend(
    TestParticleBackend, attr, value
):
    """Parameterized check for different props of ParticleDataset"""
    ds = ParticleDataset("TestParticleBackend_location")
    assert getattr(ds, attr) == value


def test_ParticleDataset_num_particles_from_Backend(TestParticleBackend):
    """Tests extraction correct type for . and its proper extraction"""
    ds = ParticleDataset("TestParticleBackend_location")
    assert isinstance(ds.num_particles, AxisType)


def test_ParticleDataset_axes_from_Backend(TestParticleBackend):
    """Tests extraction correct type for .axes and its proper extraction"""
    ds = ParticleDataset("TestParticleBackend_location")

    assert isinstance(ds.axes, Mapping)

    for axes_name, type_ in [
        ("iteration", AxisType),
        ("time", AxisType),
    ]:
        assert axes_name in ds.axes
        assert isinstance(ds.axes.get(axes_name), type_)


@pytest.mark.parametrize(
    "attr, value",
    [("name", "iteration"), ("label", "iteration"), ("unit", "")],
    ids=["name", "label", "unit"],
)
def test_ParticleDataset_iteration_axis_from_Backend(
    TestParticleBackend, attr, value
):
    """Extraction is correct for iteration axis. Check attributes for axis"""
    ds = ParticleDataset("TestParticleBackend_location")
    assert getattr(ds.axes["iteration"], attr) == value


@pytest.mark.parametrize(
    "attr, value",
    [("name", "time"), ("label", "time"), ("unit", "time unit")],
    ids=["name", "label", "unit"],
)
def test_ParticleDataset_time_axis_from_Backend(
    TestParticleBackend, attr, value
):
    """Extraction is correct for iteration axis. Check attributes for axis"""
    ds = ParticleDataset("TestParticleBackend_location")
    assert getattr(ds.axes["time"], attr) == value


def test_ParticleDataset_quantities_from_Backend(TestParticleBackend):
    """Checks if quantities propagate from backend to ParticleDataset"""
    ds = ParticleDataset("TestParticleBackend_location")

    assert isinstance(ds.quantities, Mapping)

    expected_names = TestParticleBackend.quantity_names
    expected_labels = TestParticleBackend.quantity_labels
    expected_units = TestParticleBackend.quantity_units

    # check key entries
    for key, expected_name in zip(ds.quantities.keys(), expected_names):
        assert key == expected_name

    # check value entries
    for quant, expected_name in zip(ds.quantities.values(), expected_names):
        assert quant.name == expected_name

    for quant, expected_label in zip(ds.quantities.values(), expected_labels):
        assert quant.label == expected_label

    for quant, expected_unit in zip(ds.quantities.values(), expected_units):
        assert quant.unit == expected_unit


@pytest.mark.skip
def test_ParticleDataset_from_array():
    """Check init for ParticleDataset from array"""
    pass


@pytest.mark.skip
def test_ParticleDataset_from_path():
    """Check init for ParticleDataset from path"""
    pass
