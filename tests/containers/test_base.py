from pathlib import Path

import pytest
import attr

from nata.containers import BaseDataset
from nata.containers import register_backend


@pytest.fixture(name="remove_abstractmethods")
def _unabstract_BaseDataset(monkeypatch):
    monkeypatch.setattr(BaseDataset, "__abstractmethods__", set())


@pytest.fixture(name="BaseDataset_final_cleanup")
def _cleanup_added_backend():
    yield
    BaseDataset._backends.clear()


@pytest.fixture(name="BaseDataset_subclass")
def _create_dummy_subclass(remove_abstractmethods):
    class Dummy(BaseDataset):
        _backends = set()

    return Dummy


def test_init_BaseDataset_raises_TypeError():
    with pytest.raises(TypeError, match="backend_name|info"):
        BaseDataset()


def test_BaseDataset_add_backend(
    BaseDataset_final_cleanup, BaseDataset_subclass
):
    assert BaseDataset._backends == set()
    assert BaseDataset_subclass._backends == set()

    def backend():
        pass

    BaseDataset.add_backend(backend)
    assert backend in BaseDataset._backends
    assert len(BaseDataset._backends) == 1
    assert BaseDataset_subclass._backends == set()


def test_BaseDataset_required_props():
    # general
    assert hasattr(BaseDataset, "backend_name")
    assert hasattr(BaseDataset, "appendable")
    assert hasattr(BaseDataset, "_backends")

    # based on attrs
    assert attr.has(BaseDataset)
    assert hasattr(attr.fields(BaseDataset), "location")


def test_BaseDataset_register_plugin():
    assert hasattr(BaseDataset, "some_plugin") == False

    def foo():
        pass

    BaseDataset.register_plugin("some_plugin", foo)

    assert hasattr(BaseDataset, "some_plugin") == True
    assert BaseDataset.some_plugin == foo


def test_init_BaseDataset_location(remove_abstractmethods):
    ds = BaseDataset(".")

    assert BaseDataset.appendable == False
    assert ds.appendable == False

    assert isinstance(ds.location, Path)
    assert ds.location == Path(".")


def test_register_backend(BaseDataset_final_cleanup, BaseDataset_subclass):
    assert BaseDataset._backends == set()
    assert BaseDataset_subclass._backends == set()

    @register_backend(BaseDataset)
    def backend():
        pass

    @register_backend(BaseDataset_subclass)
    def backend_for_subclass():
        pass

    assert backend in BaseDataset._backends
    assert len(BaseDataset._backends) == 1
    assert backend_for_subclass in BaseDataset_subclass._backends
    assert len(BaseDataset_subclass._backends) == 1


def test_register_backend_raises_invalid_subclass():
    with pytest.raises(ValueError, match="Invalid container"):

        @register_backend(int)
        def backend():
            pass
