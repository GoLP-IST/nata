Plugins
-------

.. py:currentmodule:: nata.plugins

Plugins provide a way of extending the functionality of a dataset. In
particular, they allow to add a method to a dataset on which they operator.
This allows for developing a pipeline-like interface. For this, the decorator
`register_container_plugin` is provided.

.. autodecorator:: register_container_plugin
