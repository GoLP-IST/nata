from abc import ABC, abstractmethod
from pathlib import Path

import attr


@attr.s
class BaseParticles(ABC):
    location: Path = attr.ib(converter=Path)

    @staticmethod
    @abstractmethod
    def is_valid_backend(file_path):
        pass

    @property
    @abstractmethod
    def name(self):
        pass

    @property
    @abstractmethod
    def short_name(self):
        pass

    @property
    @abstractmethod
    def num_particles(self):
        pass

    @property
    @abstractmethod
    def has_tags(self):
        pass

    @property
    @abstractmethod
    def tags(self):
        pass

    @property
    @abstractmethod
    def quantities_list(self):
        pass

    @property
    @abstractmethod
    def dataset(self):
        pass

    @property
    def quantities_long_names(self):
        pass

    @property
    @abstractmethod
    def quantities_units(self):
        pass

    @property
    @abstractmethod
    def dtype(self):
        pass

    @property
    @abstractmethod
    def iteration(self):
        pass

    @property
    @abstractmethod
    def time_step(self):
        pass

    @property
    @abstractmethod
    def time_unit(self):
        pass
