Basic concepts
==============

Natas idea of generalizing data reading, data processing, and data
visualization is build on top of three core concepts, *backends*,
*datasets*, and *plugins*. It is important to understand the core principles
of this concepts. All supported types are stored in :mod:`nata.types` and are
Protocols_ which mimic static-typing-like behavior in a duck-typing
environment. They don't provide any functionality except allowing to have a
"ground truth" to the objects inside nata and to support of type checking.
Most of the objects allow you to easily check if an object fulfills the
protocol by ``isinstance(instance_of_object, some_protocol)``.

.. toctree::
  :caption: Basic components of nata

  basic_concepts/backends
  basic_concepts/datasets
  basic_concepts/axes
  basic_concepts/array_interface
  basic_concepts/plugins

.. _Protocols: https://mypy.readthedocs.io/en/stable/protocols.html
