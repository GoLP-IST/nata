# -*- coding: utf-8 -*-
from nata.containers.formatting import Table


def test_Table_label():
    table = Table("label of the table", content={})
    assert table.label == "label of the table"


def test_Table_content():
    table = Table("label", {"name": "content"})
    assert table.content == {"name": "content"}


def test_Table_foldable():
    # default option
    table = Table("", {})
    assert table.foldable is True

    table = Table("", {}, foldable=True)
    assert table.foldable is True

    table = Table("", {}, foldable=False)
    assert table.foldable is False


def test_Table_render_as_html_using_foldable_table():
    table = Table(
        "some label",
        {
            "some key": "some value",
            "some other key": "some other value",
        },
        foldable=True,
    )
    expected_html = (
        "<details>"
        "<summary>some label</summary>"
        "<table style='margin-left: 1rem;'>"
        "<tr>"
        "<td style='font-weight: bold; text-align: left;'>some key</td>"
        "<td style='text-align: left;'>some value</td>"
        "</tr>"
        "<tr>"
        "<td style='font-weight: bold; text-align: left;'>some other key</td>"
        "<td style='text-align: left;'>some other value</td>"
        "</tr>"
        "</table>"
        "</details>"
    )
    assert table.render_as_html() == expected_html


def test_Table_render_as_html_without_foldable_table():
    table = Table(
        "some label",
        {
            "some key": "some value",
            "some other key": "some other value",
        },
        foldable=False,
    )
    expected_html = (
        "<div>some label</div>"
        "<table style='margin-left: 1rem;'>"
        "<tr>"
        "<td style='font-weight: bold; text-align: left;'>some key</td>"
        "<td style='text-align: left;'>some value</td>"
        "</tr>"
        "<tr>"
        "<td style='font-weight: bold; text-align: left;'>some other key</td>"
        "<td style='text-align: left;'>some other value</td>"
        "</tr>"
        "</table>"
    )
    assert table.render_as_html() == expected_html
