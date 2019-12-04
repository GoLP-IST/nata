from typing import Union

from .grid import GridDataset
from .particles import ParticleDataset

DatasetTypes = Union[GridDataset, ParticleDataset]
