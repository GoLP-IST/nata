# -*- coding: utf-8 -*-
from pathlib import Path
from typing import Any
from typing import Dict
from typing import Optional
from typing import Set
from typing import TypeVar
from typing import Union

import attr
import numpy as np
from attr.validators import deep_iterable
from attr.validators import instance_of
from attr.validators import optional

from nata.backends.particles import ParticleArray
from nata.backends.particles import ParticleBackend
from nata.utils.attrs import attrib_equality
from nata.utils.attrs import subdtype_of

from .axes import IterationAxis
from .axes import ParticleQuantity
from .axes import TimeAxis
from .axes import UnnamedAxis
from .base import BaseDataset

ParticleBackendBased = TypeVar(
    "ParticleBackendBased", str, Path, ParticleBackend
)


@attr.s(init=False, eq=False)
class ParticleDataset(BaseDataset):
    """Container class storing particle datasets"""

    _backends: Set[ParticleBackend] = set((ParticleArray,))
    backend: Optional[str] = attr.ib(validator=optional(subdtype_of(np.str_)))
    appendable = True

    name: str = attr.ib(validator=subdtype_of(np.str_))
    dtype: np.dtype = attr.ib(validator=instance_of(np.dtype))

    iteration: IterationAxis = attr.ib(validator=instance_of(IterationAxis))
    time: TimeAxis = attr.ib(validator=instance_of(TimeAxis))
    num_particles: UnnamedAxis = attr.ib(validator=instance_of(UnnamedAxis))

    quantities: np.ndarray = attr.ib(
        validator=deep_iterable(
            member_validator=subdtype_of(np.str_),
            iterable_validator=instance_of(np.ndarray),
        ),
        eq=False,
    )
    quantity_labels: np.ndarray = attr.ib(
        validator=deep_iterable(
            member_validator=subdtype_of(np.str_),
            iterable_validator=instance_of(np.ndarray),
        ),
        eq=False,
    )
    quantity_units: np.ndarray = attr.ib(
        validator=deep_iterable(
            member_validator=subdtype_of(np.str_),
            iterable_validator=instance_of(np.ndarray),
        ),
        eq=False,
    )

    def info(self, full: bool = False):  # pragma: no cover
        return self.__repr__()

    def __init__(
        self,
        particles: Optional[ParticleBackendBased],
        **kwargs: Dict[str, Any],
    ):
        if particles is None:
            self._init_from_kwargs(**kwargs)
        else:
            self._init_from_backend(particles)

        attr.validate(self)

    def _init_from_kwargs(self, **kwargs: Dict[str, Any]):
        raise NotImplementedError

    def _init_from_backend(self, particles: ParticleBackendBased):
        if not isinstance(particles, ParticleBackend):
            particles = self._convert_to_backend(particles)

        self.backend = particles.name

        self.name = particles.dataset_name
        self.dtype = particles.dtype

        self.iteration = IterationAxis(data=particles.iteration)
        self.time = TimeAxis(data=particles.time_step, unit=particles.time_unit)
        self.num_particles = UnnamedAxis(data=particles.num_particles)

        self.quantities = np.asarray(particles.quantities)
        self.quantity_labels = np.asarray(particles.quantity_labels)
        self.quantity_units = np.asarray(particles.quantity_units)

        for quantity, label, unit in zip(
            self.quantities, self.quantity_labels, self.quantity_units
        ):
            q = ParticleQuantity(
                data=[particles],
                len=[particles.num_particles],
                dtype=self.dtype[quantity],
                name=quantity,
                label=label,
                unit=unit,
            )
            setattr(self, quantity, q)

    def _check_dataset_equality(self, other: Union["ParticleDataset", Any]):
        if not isinstance(other, self.__class__):
            return False

        if not attrib_equality(self, other):
            return False

        return True

    def append(self, other: Union["ParticleDataset", Any]):
        self._check_appendability(other)

        self.iteration.append(other.iteration)
        self.time.append(other.time)
        self.num_particles.append(other.num_particles)

        for q in self.quantities:
            getattr(self, q).append(getattr(other, q))
