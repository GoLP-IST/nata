Array interface
---------------

.. py:currentmodule:: nata.types

.. note:: This section characterizes protocols for backends. The purpose of \
          a protocol is to provide **support for nominal and structural \
          subtyping**.

Next to backends, dataset, and axes, nata provides objects to have an array
interface. Inside nata, a interface is determined by `HasArrayInterface`
protocol. It is especially important to allow dispatching for numpy.

.. autoclass:: HasArrayInterface
  :show-inheritance:
  :members:

  .. automethod:: __array__
