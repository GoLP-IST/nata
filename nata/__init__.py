# -*- coding: utf-8 -*-
from nata.backends import *

# from nata.comfort import load
# from nata.comfort import activate_logging
from nata.containers import *
from nata.plugins import *

from collections import namedtuple
from pathlib import Path

_Examples = namedtuple("Examples", ["grids", "particles"])
examples = _Examples(
    Path(__file__).parent / ".." / "examples" / "data" / "grids",
    Path(__file__).parent / ".." / "examples" / "data" / "particles",
)
