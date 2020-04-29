<p align="center">
     <img 
          src="https://raw.githubusercontent.com/GoLP-IST/nata/master/docs/_static/nata-logo.png?token=AAKKMGXZXXRCACBKWRAZDFC6WLTBO" 
          alt="nata logo" 
          width=460
     />
</p>

**Nata** is a python package for post-processing and visualizing simulation
output for particle-in-cell codes. It utilizes the numpy interface to provide
a simple way to read, manipulate, and represent simulation output.

## Installing nata

Nata is available on PyPI. You can install it by running the following
command inside your terminal

```shell
pip install nata
```

It can be used inside an IPython shell or [jupyter notebook](https://jupyter.org/) together
with [ipywidgets](https://github.com/jupyter-widgets/ipywidgets). Hence, you
might need to run after the installation

```shell
# can be skipped for notebook version 5.3 and above
jupyter nbextension enable --py --sys-prefix widgetsnbextension
```

and if you want to use it inside JupyterLab (note that this requires nodejs
to be installed)

```bash
jupyter labextension install @jupyter-widgets/jupyterlab-manager
```

In case of issues, please visit the [installation section of ipywidgets](https://github.com/jupyter-widgets/ipywidgets/blob/master/docs/source/user_install.md)
for further details.

## Contributing to nata

Any type of contribution to nata is appreciated. If you have any issues,
please report them [by adding an issue on GitHub](https://github.com/GoLP-IST/nata/issues). But if
you wish to directly contribute to nata, we recommend to setup a local
development environment. Follow the instruction below for more details.

### Getting the source code

The source code is hosted on GitHub. Simply create a fork and apply your
changes. You can always push any changes to your local fork, but if you would
like to share your contribution, please create a pull request, so that it can
be reviewed.

### Local Development Environment

For the local development environment, we use
[poetry](https://python-poetry.org/). This allows us to deal better with
dependencies issues and to ensure coding standards without the burden of
manually fixing and checking for styles. To use poetry, simply install it
using the latest version by running the following command in your terminal.

```shell
curl -sSL https://raw.githubusercontent.com/python-poetry/poetry/master/get-poetry.py | python
```

Afterwards, simply run

```shell
poetry install
```

which will install all the dependencies (including development packages) in a
virtual environment. If you wish to run a command inside the virtual
environment, simply run it by `poetry run <your command>`, e.g. to run all
the test

```shell
poetry run pytest tests
```

Alternatively, you can directly use the virtual environment by running the
command

```shell
poetry shell
```

which will spawn a shell inside the virtual environment. With this, you can
even run jupyter outside of the source directory.

In addition, we use [pre-commit](https://pre-commit.com/) to help us keep
consistency in our development without any additional burden. Please use it
as well. Inside the root directory of the repository, run the command

```shell
poetry run pre-commit install
```

which will create commit hooks for you and modify files keeping a consistent
structure.

# Credits

**Nata** is created and maintained by [Anton Helm](https://github.com/ahelm)
and [Fábio Cruz](https://github.com/fabiocruz).

The development is kindly supported by the [Group for Lasers and Plasmas (GoLP)](http://epp.tecnico.ulisboa.pt/>).
