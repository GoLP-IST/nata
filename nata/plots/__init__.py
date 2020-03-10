# -*- coding: utf-8 -*-
from typing import Union

from .plots.line import LinePlot
from .plots.color import ColorPlot
from .plots.scatter import ScatterPlot
from .data import PlotDataAxis, PlotData

PlotTypes = Union[LinePlot]

# default plots for data types
DefaultGridPlotTypes = {1: LinePlot, 2: ColorPlot}
DefaultParticlePlotType = ScatterPlot
