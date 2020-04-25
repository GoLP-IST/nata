# -*- coding: utf-8 -*-
# from typing import List
# from typing import Optional

# import numpy as np

# from IPython.display import display
# from ipywidgets import Layout
# from ipywidgets import widgets
# from nata.containers import GridDataset
# from nata.plots import DefaultGridPlotTypes
# from nata.plots.axes import Axes
# from nata.plots.data import PlotData
# from nata.plots.data import PlotDataAxis
# from nata.plots.figure import Figure
# from nata.plots.helpers import filter_style
# from nata.plots.plans import AxesPlan
# from nata.plots.plans import FigurePlan
# from nata.plots.plans import PlotPlan
# from nata.plugins.register import register_container_plugin
# from nata.utils.env import inside_notebook

# @register_container_plugin(ParticleDataset, name="plot_data")
# def particle_plot_data(
#     dataset: ParticleDataset, quants: List[str] = []
# ) -> PlotData:

#     a = []
#     d = []

#     for quant in quants:
#         q = getattr(dataset, quant)
#         new_a = PlotDataAxis(name=q.name, label=q.label, units=q.unit)

#         a.append(new_a)
#         d.append(np.array(q))

#     p_d = PlotData(
#         name=dataset.name,
#         label=dataset.name,
#         units="",
#         data=d,
#         time=np.array(dataset.time),
#         time_units=dataset.time.unit,
#         axes=a,
#     )

#     return p_d

# @register_container_plugin(ParticleDataset, name="plot_type")
# def particle_plot_type(dataset: ParticleDataset) -> PlotData:
#     return DefaultParticlePlotType

# @register_container_plugin(ParticleDataset, name="plot")
# def plot_particle_dataset(
#     dataset: ParticleDataset,
#     quants: List[str] = [],
#     fig: Optional[Figure] = None,
#     axes: Optional[Axes] = None,
#     style: dict = dict(),
#     interactive: bool = True,
#     n: int = 0,
# ):

#     p_plan = PlotPlan(
#         dataset=dataset,
#         quants=quants,
#         style=filter_style(dataset.plot_type(), style),
#     )


#     if len(dataset.iteration) > 1 and inside_notebook() and interactive:
#         build_interactive_tools(f_plan, n)

#     else:
#         # build figure
#         fig = build_figure(f_plan)
#         return fig


# @register_container_plugin(DatasetCollection, name="plot")
# def plot_collection(
#     collection: DatasetCollection,
#     order: Optional[list] = list(),
#     styles: Optional[dict] = dict(),
#     quants: Optional[Dict[str, list]] = list(),
#     interactive: bool = True,
#     n: int = 0,
# ) -> Figure:

#     # check if collection is not empty
#     if not collection.store:
#         raise ValueError("Collection is empty.")

#     # check if time and iteration arrays are equal
#     for check in ["iteration", "time"]:
#         arr = [
#             np.array(getattr(dataset, check))
#             for dataset in collection.store.values()
#         ]
#         if not all(np.array_equal(arr[0], i) for i in arr):
#             raise ValueError(
#                 f"Attribute `{check}` is not the same for all datasets "
#                 + "in the collection."
#             )

#     f_a = []

#     for dataset in collection.store.values():
#         i_style = (
#             styles[dataset.name] if dataset.name in styles.keys() else None
#         )

#         if isinstance(dataset, ParticleDataset):
#             if dataset.name not in quants.keys():
#                 raise ValueError("quants not passed!")
#             i_quants = quants[dataset.name]
#         else:
#             i_quants = None

#         p_plan = PlotPlan(
#             dataset=dataset,
#             quants=i_quants,
#             style=filter_style(dataset.plot_type(), i_style),
#         )

#         a_plan = AxesPlan(
#             axes=None, plots=[p_plan], style=filter_style(Axes, i_style)
#         )

#         f_a.append(a_plan)

#     f_plan = FigurePlan(
#         fig=None,
#         axes=f_a,
#         style=filter_style(
#             Figure, styles["fig"] if "fig" in styles.keys() else None
#         ),
#     )

#     if len(dataset.iteration) > 1 and inside_notebook() and interactive:
#         build_interactive_tools(f_plan, n)

#     else:
#         # build figure
#         fig = build_figure(f_plan)
#         return fig
