# -*- coding: utf-8 -*-
from typing import Union

from .plots.line import LinePlot
from .plots.color import ColorPlot
from .plots.scatter import ScatterPlot

from .data import PlotDataAxis, PlotData
from .figure import Figure
from .axes import Axes

PlotTypes = Union[LinePlot]

LabelablePlotTypes = Union[LinePlot]

DefaultGridPlotTypes = {1: LinePlot, 2: ColorPlot}

DefaultParticlePlotType = ScatterPlot
