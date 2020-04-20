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
        self._dataset: dict = dataset
        self._quants: List[str] = quants
        self._style: dict = style

    @property
    def dataset(self) -> GridDataset:
        return self._dataset

    @property
    def quants(self) -> List[str]:
        return self._quants

    @property
    def style(self) -> dict:
        return self._style


class AxesPlan:
    def __init__(
        self,
        plots: List[PlotPlan] = list(),
        axes: Optional[Axes] = object(),
        style: Optional[dict] = dict(),
    ):
        self._plots: dict = plots
        self._axes: Axes = axes
        self._style: dict = style

    @property
    def plots(self) -> List[PlotPlan]:
        return self._plots

    @property
    def axes(self) -> Axes:
        return self._axes

    @property
    def style(self) -> dict:
        return self._style

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
        self._axes: dict = axes
        self._fig: list = fig
        self._style: dict = style

    @property
    def axes(self) -> List[AxesPlan]:
        return self._axes

    @property
    def fig(self) -> Figure:
        return self._fig

    @property
    def style(self) -> dict:
        return self._style

    @property
    def datasets(self) -> list:
        ds = []
        for a in self.axes:
            for d in a.datasets:
                ds.append(d)
        return ds
