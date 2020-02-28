# -*- coding: utf-8 -*-
# flake8: noqa: E402
from typing import Union


from .base import register_backend
from .grid import GridDataset
from .particles import ParticleDataset

DatasetTypes = Union[GridDataset, ParticleDataset]

from .collection import DatasetCollection
