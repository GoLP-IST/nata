# -*- coding: utf-8 -*-
from typing import Dict
from typing import List
from typing import Optional

from nata.containers import GridDataset
from nata.plots.axes import Axes
from nata.plots.figure import Figure


class PlotPlan:
    def __init__(
        self,
        dataset: GridDataset = object(),
        quants: Optional[List[str]] = list(),
        style: Optional[Dict] = dict(),
    ):
        self.dataset = dataset
        self.quants = quants
        self.style = style


class AxesPlan:
    def __init__(
        self,
        plots: List[PlotPlan] = list(),
        axes: Optional[Axes] = object(),
        style: Optional[dict] = dict(),
    ):
        self.plots = plots
        self.axes = axes
        self.style = style

    @property
    def datasets(self) -> list:
        ds = []
        for p in self.plots:
            ds.append(p.dataset)
        return ds


class FigurePlan:
    def __init__(
        self,
        axes: List[AxesPlan] = list(),
        fig: Optional[Figure] = object(),
        style: Optional[dict] = dict(),
    ):
        self.axes = axes
        self.fig = fig
        self.style = style

    @property
    def datasets(self) -> list:
        ds = []
        for a in self.axes:
            for d in a.datasets:
                ds.append(d)
        return ds
