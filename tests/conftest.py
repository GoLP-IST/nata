# -*- coding: utf-8 -*-
import os
from contextlib import contextmanager

from hypothesis import Verbosity
from hypothesis import settings


def pytest_configure(config):
    config.addinivalue_line(
        "markers", "wip: work-in-progress marker to run currently "
    )


@contextmanager
def does_not_raise():
    yield


# taken from:
# https://hypothesis.readthedocs.io/en/latest/settings.html#settings-profiles
settings.register_profile("ci", max_examples=1000)
settings.register_profile("dev", max_examples=10)
settings.register_profile(
    "debug", max_examples=10, report_multiple_bugs=False,
)
settings.register_profile(
    "verbose", max_examples=10, verbosity=Verbosity.verbose,
)
settings.load_profile(os.getenv("HYPOTHESIS_PROFILE", "default"))
