Backends
--------
.. py:currentmodule:: nata.types

.. note:: This section characterizes protocols for backends. The purpose of \
          a protocol is to provide **support for nominal and structural \
          subtyping**.

Backend provide access to stored data. In general, each backend is of type
:class:`BackendType`.

.. autoclass:: BackendType
  :show-inheritance:
  :members:
  :exclude-members: is_valid_backend


To avoid validity checks at initialization level, each backend has a
simple static-method of receiving a location and deducing if the backend
is a valid backend, given a location.

.. automethod:: BackendType.is_valid_backend

Backends for grids
^^^^^^^^^^^^^^^^^^

For retrieving grid based data, the :class:`GridBackendType`
extends the base backend type.

.. autoclass:: GridBackendType
  :show-inheritance:
  :members:

.. autoclass:: GridDataReader
  :show-inheritance:

  .. automethod:: get_data

Backends for particles
^^^^^^^^^^^^^^^^^^^^^^

For retrieving particle based data, the
:class:`ParticleBackendType` extends the base backend type.

.. autoclass:: ParticleBackendType
  :show-inheritance:
  :members:

In addition, for reading the underlying particle array the
:class:`ParticleDataReader` extends the :class:`ParticleBackendType`.

.. autoclass:: ParticleDataReader
  :show-inheritance:

  .. automethod:: get_data
