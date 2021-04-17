# -*- coding: utf-8 -*-
from nata.containers.utils import get_doc_heading


def test_get_doc_heading_without_docs():
    def func():
        pass

    assert get_doc_heading(func) == ""


def test_get_doc_heading_without_newline():
    def func():
        """Some long string

        This should be hidden
        """
        pass

    assert get_doc_heading(func) == "Some long string"


def test_get_doc_heading_with_newline():
    def func():
        """
        Some long string

        This should be hidden
        """
        pass

    assert get_doc_heading(func) == "Some long string"
