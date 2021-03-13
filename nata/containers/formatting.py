# -*- coding: utf-8 -*-
from dataclasses import dataclass
from typing import Dict


@dataclass
class Table:
    label: str
    content: Dict[str, str]
    foldable: bool = True
    fold_closed: bool = True

    def render_as_html(self) -> str:
        inner = ""
        for key, value in self.content.items():
            inner += (
                "<tr>"
                f"<td style='font-weight: bold; text-align: left;'>{key}</td>"
                f"<td style='text-align: left;'>{value}</td>"
                "</tr>"
            )

        if self.foldable:
            return (
                f"<details{'' if self.fold_closed else ' open'}>"
                f"<summary>{self.label}</summary>"
                f"<table style='margin-left: 1rem;'>{inner}</table>"
                "</details>"
            )
        else:
            return (
                f"<div>{self.label}</div>"
                f"<table style='margin-left: 1rem;'>{inner}</table>"
            )
