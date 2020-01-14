from pathlib import Path
import re

import pytest
import numpy as np

from nata.containers.base import register_backend
from nata.containers.particles import ParticleDataset


@pytest.fixture(name="patch_location_exist")
def _patch_Path_exists_True(monkeypatch):
    monkeypatch.setattr(Path, "exists", lambda _: True)


@pytest.fixture(name="dummy_particles_backend")
def _dummy_backend(patch_location_exist):
    previous_backends = ParticleDataset._backends.copy()
    ParticleDataset._backends.clear()

    assert len(ParticleDataset._backends) == 0

    @register_backend(ParticleDataset)
    class ParticleArray:
        @staticmethod
        def is_valid_backend(s):
            if "particles" == s.name.split(".")[0]:
                return True
            return False

        def __init__(self, location: Path):
            self.location = location
            self.name = "particles_backend"
            self.short_name = "prt"
            self.long_name = "particles"

            suffixes = [s.strip(".") for s in location.suffixes]
            quantaties = suffixes[0].split("-")
            self.quantities_list = np.array(quantaties)

            splits = [re.split(r"(\d+)", s) for s in quantaties]
            self.quantities_long_names = np.array(
                ["_".join([s for s in l if s]) for l in splits]
            )

            self.quantities_units = np.array(
                [s + " unit" for s in self.quantities_list]
            )

            self.num_particles = int(suffixes[1])
            self.iteration = int(suffixes[2])
            self.time_step = self.iteration * 0.1
            self.time_unit = "time unit"

            self.dtype = np.dtype(
                [(name, np.float) for name in self.quantities_list]
            )
            self.has_tags = False
            self.dataset = np.zeros(self.num_particles, dtype=self.dtype)
            for q in self.quantities_list:
                self.dataset[q] = np.arange(self.num_particles)

    assert ParticleArray in ParticleDataset._backends

    yield

    # teardown code
    ParticleDataset._backends = previous_backends
    assert ParticleArray not in ParticleDataset._backends


def test_ParticleDataset_append(dummy_particles_backend):
    particles = ParticleDataset("particles.x1-x2-p1-p2.129.0")
    assert len(particles.iterations) == 1
    np.testing.assert_array_equal(particles.iterations, [0])
    assert len(particles.time) == 1
    np.testing.assert_array_equal(particles.time, [0.0])
    assert len(particles._num_particles) == 1

    particles.append(ParticleDataset("particles.x1-x2-p1-p2.129.1"))
    assert len(particles.iterations) == 2
    np.testing.assert_array_equal(particles.iterations, [0, 1])
    assert len(particles.time) == 2
    np.testing.assert_array_equal(particles.time, [0.0, 0.1])
    assert len(particles._num_particles) == 2


def test_ParticleDataset_data(dummy_particles_backend):
    # single data
    particles = ParticleDataset("particles.x1-x2-p1-p2.129.0")
    assert len(particles.data) == 129
    assert particles.data.shape == (129,)
    np.testing.assert_array_equal(
        particles.data.dtype.names, ["x1", "x2", "p1", "p2"]
    )

    # multi data
    particles.append(ParticleDataset("particles.x1-x2-p1-p2.10.1"))
    assert particles.data.shape == (2, 129)
    assert isinstance(particles.data, np.ma.MaskedArray)
    np.testing.assert_array_equal(
        particles.data.dtype.names, ["x1", "x2", "p1", "p2"]
    )
    assert np.count_nonzero(particles.data.mask[0, :]) == 0
    assert np.count_nonzero(particles.data.mask[1, :]) == 119


@pytest.fixture(name="particles")
def _single_entry_dataset(dummy_particles_backend):
    return ParticleDataset("particles.x1-x2-p1-p2.129.0")


@pytest.mark.parametrize(
    "parameter, value, type_",
    [
        ("backend", "particles_backend", str),
        ("name", "prt", str),
        ("quantities", ["x1", "x2", "p1", "p2"], np.ndarray),
        ("quantities_labels", ["x_1", "x_2", "p_1", "p_2"], np.ndarray),
        (
            "quantities_units",
            ["x1 unit", "x2 unit", "p1 unit", "p2 unit"],
            np.ndarray,
        ),
        ("tagged", False, bool),
    ],
)
def test_ParticleDataset_props(particles, parameter, value, type_):
    if type_ == np.ndarray:
        np.testing.assert_array_equal(getattr(particles, parameter), value)
    else:
        assert getattr(particles, parameter) == value
    assert isinstance(getattr(particles, parameter), type_)

@pytest.fixture(name="particles_multipleIteration")
def _multiple_entry_dataset(particles):
    for i in range(1, 129):
        particles.append(ParticleDataset(f"particles.x1-x2-p1-p2.129.{i}"))
    return particles

@pytest.mark.parametrize(
    "selection, iterations",
    [
        (np.s_[5], np.arange(5, 6)),
        (np.s_[:5], np.arange(5)),
        (np.s_[5:], np.arange(5, 129)),
        (np.s_[6:98], np.arange(6, 98)),
        (np.s_[7:57:5], np.arange(7, 57, 5)),
        # TODO: does not work now. FIXME
        # (np.s_[:], np.arange(129)),
        # (np.s_[::], np.arange(129)),
    ]
)
def test_ParticleDataset_getitem(particles_multipleIteration, selection, iterations):
    new_particles = particles_multipleIteration[selection]
    assert new_particles != particles_multipleIteration
    assert new_particles.store != particles_multipleIteration.store
    np.testing.assert_array_equal(new_particles.iterations, iterations)

