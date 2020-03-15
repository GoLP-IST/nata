# -*- coding: utf-8 -*-
from contextlib import contextmanager


def pytest_configure(config):
    config.addinivalue_line(
        "markers", "wip: work-in-progress marker to run currently "
    )


@contextmanager
def does_not_raise():
    yield
