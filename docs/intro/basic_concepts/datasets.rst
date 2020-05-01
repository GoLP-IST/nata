Datasets
--------

.. py:currentmodule:: nata.types

.. note:: This section characterizes protocols for datasets. The purpose of \
          a protocol is to provide **support for nominal and structural \
          subtyping**.

Datasets, as a protocol, are mutable containers with the base type of
`DatasetType`. The mutability arises naturally from the need of
allowing data to be appended, e.g. a dataset which contain similar
information, but at a different time. In addition to it, datasets have the
possibility to interact with backends to obtain required data.

.. autoclass:: DatasetType
  :show-inheritance:
  :members: _backends

To interact with the dataset store, each dataset has to provide the following
class methods:

.. automethod:: DatasetType.add_backend
.. automethod:: DatasetType.remove_backend
.. automethod:: DatasetType.is_valid_backend
.. automethod:: DatasetType.get_backends

Next to having a tight connection to backends, datasets provide a way of
interacting with other datasets, especially with similar datasets. For this,
two methods are part the `DatasetType` protocol.

.. automethod:: DatasetType.equivalent
.. automethod:: DatasetType.append

Datasets for grids
^^^^^^^^^^^^^^^^^^

`GridDatasetType` extends the `DatasetType` protocol to include additional
information for grids.

.. autoclass:: GridDatasetType
  :show-inheritance:
  :members:

Datasets for particles
^^^^^^^^^^^^^^^^^^^^^

`ParticleDatasetType` extends the `DatasetType` protocol to include additional
information for grids.

.. autoclass:: ParticleDatasetType
  :show-inheritance:
  :members:

Particle datasets are in general containers which store particle quantities
which follow the `QuantityType` protocol.

.. autoclass:: QuantityType
  :show-inheritance:
  :members:
  :exclude-members: append, equivalent

Particle quantities following the `QuantityType` protocol have in addition
methods to append more data to them.

.. automethod:: QuantityType.equivalent
.. automethod:: QuantityType.append
