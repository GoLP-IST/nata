from typing import Union

from .plots.line import LinePlot
from .plots.color import ColorPlot
# from Scatter import ScatterPlot

PlotTypes = Union[
    LinePlot
]

LabelablePlotTypes = Union[
    LinePlot
]

DefaultGridPlotTypes = {
    1: LinePlot,
    2: ColorPlot
}

# DefaultParticlePlotType = ScatterPlot

from .data import PlotDataAxis, PlotData
from .figure import Figure
from .axes import Axes