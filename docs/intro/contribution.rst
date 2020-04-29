Contributing to nata
====================

Any type of contribution to nata is appreciated. If you have any issues,
please report them here_. But if you wish to directly contribute to nata, we
recommend to setup a local development environment.

Getting the source code
-----------------------

The source code is hosted on GitHub_. Simply create a fork and apply your
changes. You can always push any changes to your local fork, but if you would
like to share your contribution, please create a pull request, so that it can
be reviewed.

Local development environment
-----------------------------

For the local development environment, we use poetry_. This allows us to deal
better with dependencies issues and to ensure coding standards without the
burden of manually fixing and checking for styles. To use poetry, simply
install it using the latest version by running the following command in your
terminal.

.. code:: bash

  curl -sSL https://raw.githubusercontent.com/python-poetry/poetry/master/get-poetry.py | python

Afterwards, simply run

.. code:: bash

  poetry install

which will install all the dependencies (including development packages) in a
virtual environment. If you wish to run a command inside the virtual
environment, simply run it by ``poetry run <your command>``, e.g. to run all
the test

.. code:: bash

  poetry run pytest

Alternatively, you can directly use the virtual environment by running the
command

.. code:: bash

  poetry shell

which will spawn a shell inside the virtual environment. With this, you can
even run jupyter outside of the source directory.

In addition, we use pre-commit_ to help us keep consistency in our development
without any additional burden. Please use it as well. Inside the root
directory of the repository, run the command

.. code:: bash

  poetry run pre-commit install

which will create commit hooks for you and modify files during git commits,
keeping a consistent structure.

.. _here: https://github.com/GoLP-IST/nata/issues
.. _GitHub: https://github.com/GoLP-IST/nata
.. _poetry: https://python-poetry.org/
.. _pre-commit: https://pre-commit.com/
