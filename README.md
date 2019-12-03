## Contributing to `nata`

### Fork, change and contribute

For contributing to `nata`, simply fork this repository and apply your desired
changes. You can always push your changes to your local fork. As soon as you
are done with your feature or fix, create a pull request.

### Local Development Environment

Use a virtual environment for development. Simply use the built in `venv`
module of python, e.g.

```shell
python -m venv --prompt nata env
source env/bin/activate
```

After this, simply run

```shell
pip install -e ".[dev]"
```

which installs all the requirements needed for development. In addition, if you
wish to test something inside a notebook, you can install it inside the created
virtual environment.

### Getting a specific python version

For nata, only the latest python versions are supported. In particular only
`python>=3.6` is supported. If you don't have this version installed, you can
use a tool like [`pyenv`](https://github.com/pyenv/pyenv).

To install the version `3.6.0` you can run

```shell
pyenv install 3.6.0  # you can use any version here
```

and use this particular version for development, by running

```shell
pyenv local 3.6.0
```
