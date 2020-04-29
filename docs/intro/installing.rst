Installing nata
===============

Nata is available on PyPI. You can install it by running the following
command inside your terminal

.. code:: bash

  pip install nata

It is intended to be used inside a `jupyter`_ together with ipywidgets_.
Hence, you might need to run after the installation

.. code:: bash

  # can be skipped for notebook version 5.3 and above
  jupyter nbextension enable --py --sys-prefix widgetsnbextension

and if you want to use it inside JupyterLab (note that this requires nodejs
to be installed)

.. code:: bash

  jupyter labextension install @jupyter-widgets/jupyterlab-manager

In case of issues, please visit the `installation section of ipywidgets`_ for
further details.

.. _jupyter: https://jupyter.org/
.. _ipywidgets: https://github.com/jupyter-widgets/ipywidgets
.. _`installation section of ipywidgets`: https://github.com/jupyter-widgets/ipywidgets/blob/master/docs/source/user_install.md
