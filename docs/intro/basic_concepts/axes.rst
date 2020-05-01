Axes
----

.. py:currentmodule:: nata.types

.. note:: This section characterizes protocols for axes. The purpose of a \
          protocol is to provide **support for nominal and structural \
          subtyping**.

Axis protocols
^^^^^^^^^^^^^^

One essential building block for datasets like `GridDatasetType` and
`ParticleDatasetType` are `AxisType` and `GridAxisType`. Similar to datasets,
protocols are in place to provide available attributes.

.. autoclass:: AxisType
  :show-inheritance:
  :members:
  :exclude-members: equivalent, append

In addition, `AxisType` provides methods to check for equivalence between
axes and to append another axis.

.. automethod:: AxisType.equivalent
.. automethod:: AxisType.append

As grid axes provide uniformity for each dimension of a grid, the `GridAxisType`
extends `AxisType`.

.. autoclass:: GridAxisType
  :show-inheritance:
  :members:

Axes container
^^^^^^^^^^^^^^

Axes serve the purpose of providing meta information for `DatasetType` and
are in general occurring in a combination with other axes. For this, two
special purpose axes container exist, `GridDatasetAxes` and
`ParticleDatasetAxes`.

.. autoclass:: GridDatasetAxes
  :show-inheritance:
  :members:

.. autoclass:: ParticleDatasetAxes
  :show-inheritance:
  :members:
