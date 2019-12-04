import pytest


def pytest_configure(config):
    config.addinivalue_line(
        "markers", "wip: work-in-progress marker to run currently "
    )
