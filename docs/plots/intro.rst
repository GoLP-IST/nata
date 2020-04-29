Introduction
=============================

**nata** provides a simple way of plotting the supported dataset types, such as
:class:`nata.containers.GridDataset` or
:class:`nata.containers.ParticleDataset`.

For example, to plot a :class:`nata.containers.GridDataset` ``dataset``, simply
call

.. code:: python

   dataset.plot()

Apart from using the data itself in ``dataset``, the plot will be built using
all available metadata, such as derived axes labels and units or titles.

Plot calls on single iteration datasets return a :class:`nata.plots.Figure`
object. Figures can be also be shown by calling the
:meth:`nata.plots.Figure.show` method.

.. code-block:: python

   fig = dataset.plot()
   fig.show()

By default, :class:`nata.plots.Figure` objects are shown when represented in
HTML. Since this is the default
representation method in notebook environment calls, there is no need to call
the :meth:`nata.plots.Figure.show` method in order to show a
:class:`nata.plots.Figure` object in a jupyter notebook if ``.plot()`` is the
last instruction in the cell.

Costumizing plots
-----------------

A big effort is continuously put into **nata** to produce out-of-the-box nearly
publication-ready plots. However, the plots are highly customizable through the
``style`` parameter, a dictionary that takes a combination of figure, axes and
plot style parameters. For example, we can set the figure size, the horizontal
axes scale and label and the line color of our plot in ``style`` altogether:

.. code:: python

   dataset.plot(
       style=dict(
           figsize=(5,4),
           xscale="log",
           xlabel="$x$ [m]",
           color="blue",
       )
   )

All style parameters that are by default inferred from the represented
dataset(s) are overriden if specified in ``style``.

Naturally, plot type specific style parameters will only be applicable if that
plot type is drawn in the current call. A list of all available style parameters
can be found on the description of all classes in :doc:`/plots/api/index`.

Combining plots
---------------

Plots can be combined by setting the ``fig`` and ``axes`` attributes in the
dataset plot plugin calls. If ``fig`` is provided and is an existing
:class:`nata.plots.Figure` instance, the current plot will be added to that
instance. Additionally, if ``axes`` is provided and is an existing
:class:`nata.plots.Axes` child object of ``fig``, the current plot will be added
to that axes.

For example, to combine two line plots of the datasets ``ds_1`` and ``ds_2`` in
one figure, but in different axes, we can do:

.. code:: python

   fig = ds_1.plot()
   fig = ds_2.plot(fig=fig)

When only ``fig`` (and not ``axes``) is provided in a ``.plot()`` call, a new
axes is added to the existing figure. If the figure has all its axes occupied,
a new row for axes is created.

We can also represent the two line plots in the same axes, by doing:

.. code:: python

   fig = ds_1.plot()
   fig = ds_2.plot(fig=fig, axes=fig.axes[0])

For more details about combining plots and the automatic restyling of the
corresponding axes and figures, see for example
:meth:`nata.containers.GridDataset.plot`.

Combining plots is also possible by applying addition and multiplication
operators (``+`` and ``*``) to :class:`nata.plots.Figure` objects. These are
shortcuts for the instructions given above.

For example, to represent the two datasets ``ds_1`` and ``ds_2`` in the same
figure, but in different axes, we can do:

.. code:: python

   fig = ds_1.plot() + ds_2.plot()

To represent the two datasets in the same axes, we can do instead:

.. code:: python

   fig = ds_1.plot() * ds_2.plot()

If the two figures involved in the ``*`` operation have more than one axes, then
all plots in axes with matching indices will be combined.
