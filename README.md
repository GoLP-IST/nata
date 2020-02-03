# NATA

## Contributing to nata

For contributing to nata, simply fork this repository and apply your desired
changes. You can always push your changes to your local fork. As soon as you
are done with your feature or fix, create a pull request.


### Local Development Environment

For the local development environment, we use
[poetry](https://python-poetry.org/). This allows us to ensure coding
standards without the burden of manually fixing and checking for styles. To
use poetry, simply install it using the latest version by running the following command in your terminal.

```shell
curl -sSL https://raw.githubusercontent.com/python-poetry/poetry/master/get-poetry.py | python
```

Afterwards, simply run

```shell
poetry install
```

which will install all the dependencies (including development packages) in a
virtual environment. If you wish to run a command inside the virtual
environment, simply run

```shell
poetry run pytest tests
```

to run all tests available in the tests directory. Alternatively, you can directly use the virtual environment by running the command

```shell
poetry shell
```

which will spawn a shell inside the virtual environment.


# Credits

`nata` is written and maintained by [Anton Helm](https://github.com/ahelm)
and [Fábio Cruz](https://github.com/fabiocruz).

The development is kindly supported by the [Group for Lasers and Plasmas
(GoLP)](http://epp.tecnico.ulisboa.pt/>).
