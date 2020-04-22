# -*- coding: utf-8 -*-
from typing import Union

from .base import BasePlot

from .line import LinePlot
from .color import ColorPlot
from .scatter import ScatterPlot

PlotTypes = Union[LinePlot]

# default plots for data types
DefaultGridPlotTypes = {1: LinePlot, 2: ColorPlot}
DefaultParticlePlotType = ScatterPlot
