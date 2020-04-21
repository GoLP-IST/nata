# -*- coding: utf-8 -*-
from typing import Union

from .types import BasePlot
from .types import LinePlot
from .types import ColorPlot
from .types import ScatterPlot

from .data import PlotDataAxis, PlotData

PlotTypes = Union[LinePlot]

# default plots for data types
DefaultGridPlotTypes = {1: LinePlot, 2: ColorPlot}
DefaultParticlePlotType = ScatterPlot
