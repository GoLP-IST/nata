# -*- coding: utf-8 -*-
from dataclasses import dataclass
from dataclasses import fields
from typing import Optional
from typing import Sequence
from typing import Union

from .elements import Colorbar
from .elements import Scale

Numbers = Union[int, float]


@dataclass
class PlotKind:
    def to_dict(self):

        kind_dict = {}
        for field in fields(self):
            kind_dict[field.name] = getattr(self, field.name)

        return kind_dict


@dataclass
class Line(PlotKind):
    color: Optional[str] = None
    style: Optional[str] = None
    width: Optional[Numbers] = None
    alpha: Optional[Numbers] = None


@dataclass
class Scatter(PlotKind):
    color: Optional[str] = None
    colorrange: Optional[Sequence[Numbers]] = None
    colorscale: Optional[Union[Scale, str]] = None
    colormap: Optional[str] = None
    colorbar: Optional[Colorbar] = None
    style: Optional[str] = None
    size: Optional[Numbers] = None
    alpha: Optional[Numbers] = None


@dataclass
class Image(PlotKind):
    colorrange: Optional[Sequence[Numbers]] = None
    colorscale: Optional[Union[Scale, str]] = None
    colormap: Optional[str] = None
    colorbar: Optional[Colorbar] = None
