from pathlib import Path
import attr
from nata.config import INDENT

# TODO: change in a dictionary printer -> use pretty print
@attr.s
class PrettyPrinter:
    header: str = attr.ib(converter=str)
    _indent_lvl: int = attr.ib(default=0, converter=int)
    _text_body: str

    def __attrs_post_init__(self) -> None:
        self._text_body = (
            f"\n{self.header}\n" + ("=" * len(self.header)) + "\n" * 2
        )

    def new_linebreak(self):
        self._text_body += "\n"

    def add_line(self, line: str) -> None:
        self._text_body += (INDENT * self._indent_lvl) + line + "\n"

    def indent(self, times: int = 1) -> None:
        self._indent_lvl += times

    def undent(self, times: int = 1) -> None:
        if times > self._indent_lvl:
            raise ValueError(
                "Undentation to high!"
                + f"Current indentation level is `{self._indent_lvl}`!"
            )
        self._indent_lvl -= times
        self._text_body += "\n"

    def flush(self) -> None:
        print(self._text_body, flush=True)
