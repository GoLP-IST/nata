# -*- coding: utf-8 -*-
from dataclasses import dataclass
from typing import Optional
from typing import Sequence
from typing import Union

import matplotlib as mpl
from pkg_resources import resource_filename

Numbers = Union[int, float]


@dataclass
class Theme:
    name: Optional[str] = None
    path: Optional[str] = None
    extra: Optional[dict] = None

    @property
    def rc(self):
        rc = {}

        if self.name:
            rc.update(
                mpl.rc_params_from_file(
                    resource_filename(__name__, "themes/" + self.name + ".rc"),
                    use_default_template=False,
                )
            )
        if self.path:
            rc.update(mpl.rc_params_from_file(self.path, use_default_template=False))

        if self.extra:
            rc.update(self.extra)

        return rc


@dataclass
class Ticks:
    values: Optional[Sequence[Numbers]] = None
    labels: Optional[Sequence[Union[str, Numbers]]] = None


@dataclass
class Colorbar:
    label: Optional[str] = None
    ticks: Optional[Ticks] = None
    visible: bool = True


class Scale:
    def __init__(self, name: str = "linear"):
        self.name = name


class LinearScale(Scale):
    def __init__(self):
        super().__init__(name="linear")


class LogScale(Scale):
    def __init__(self, base: Numbers = 10):
        super().__init__(name="log")
        self.base = base


class SymmetricalLogScale(Scale):
    def __init__(self, base: Numbers = 10, linthresh: Optional[Numbers] = None):
        super().__init__(name="symlog")
        self.base = base
        self.linthresh = linthresh


def scale_from_str(name: str):
    if name == "linear":
        return LinearScale()
    elif name == "log":
        return LogScale()
    elif name == "symlog":
        return SymmetricalLogScale()
    else:
        raise ValueError("invalid scale name")


def mpl_norm_from_scale(scale: Scale, crange: Sequence[Numbers]):

    if not scale or isinstance(scale, LinearScale):
        return mpl.colors.Normalize(vmin=crange[0], vmax=crange[1])

    elif isinstance(scale, LogScale):
        return mpl.colors.LogNorm(vmin=crange[0], vmax=crange[1])

    elif isinstance(scale, SymmetricalLogScale):
        return mpl.colors.SymLogNorm(
            vmin=crange[0],
            vmax=crange[1],
            base=scale.base,
            linthresh=scale.linthresh or 1.0,
        )


def mpl_scale_from_scale(scale: Scale):
    if isinstance(scale, LinearScale):
        return mpl.scale.LinearScale(axis=None)

    elif isinstance(scale, LogScale):
        return mpl.scale.LogScale(axis=None, base=scale.base)

    elif isinstance(scale, SymmetricalLogScale):
        return mpl.scale.SymmetricalLogScale(
            axis=None, base=scale.base, linthresh=scale.linthresh or 1.0
        )
