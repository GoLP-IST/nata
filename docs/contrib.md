# Contribution

Any contribution to nata is appreciated. If you have any issues, please report
them [here][issues]. But if you wish to contribute to nata directly, we
recommend to use Visual Studio Code as it allows to utilize devcontainers.

## Getting the source code

Nata's can be found on [GitHub][nata on GitHub]. Create a fork and apply your
changes. You can always push any changes to your local fork, but if you would
like to share your contribution, please create a pull request so that the
maintainers can review it.

## Devcontainer

Nata source ships with a [devcontainer setup][devcontainer]. Suppose you are
using *Visual Studio Code* as your editor. In that case, you can open the
project inside a devcontainer, and vscode will set up the development
environment for you. The same applies if you are using [GitHub's
Codespaces][codespaces].

!!! info
    If you like to change something and contribute it back quickly, using
    devcontainer/codespaces is the easiest and fastest way to get you started.

## Local development environment

For the local setup, you need to setup [][poetry] and optionally [][pre-commit].
The use of both tools allows to deal better with dependencies issues, and helps
to ensure coding standards without the the burden of manually fixing issues with
styling.

### Poetry

To use poetry, install it using the latest version by running the following
command in your terminal.

```sh
curl -sSL \
  https://raw.githubusercontent.com/python-poetry/poetry/master/get-poetry.py \
  | python
```

Afterward, run

```sh
poetry install
```

To install all the dependencies (including development packages) in a virtual
environment. If you wish to run a command inside the virtual environment, run it
by prefixing your comment by `poetry run <your command>`, e.g., to run all the
test

```sh
poetry run pytest
```

Alternatively, you can directly use the virtual environment by running the
command

```sh
poetry shell
```

That will spawn a shell inside the virtual environment. With this, you can even
run `jupyter` outside of the source directory.

### Pre-commit

We use [pre-commit][pre-commit] to help us keep consistency in our development
without any additional burden. It is **not required** to be installed but helps
you to push code changes to nata. Please follow the instructions for
installation found [here][pre-commit install]. Afterward, inside the project
directory run

```sh
pre-commit install
```

To create commit hooks for you and modify files during git commits, keeping a
consistent structure.

[issues]: https://github.com/GoLP-IST/nata/issues
[nata on GitHub]: https://github.com/GoLP-IST/nata
[devcontainer]: https://code.visualstudio.com/docs/remote/containers
[codespaces]: https://github.com/features/codespaces
[poetry]: https://python-poetry.org/
[pre-commit]: https://pre-commit.com/
[pre-commit install]: https://pre-commit.com/#installation
