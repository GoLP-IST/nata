from setuptools import setup, find_packages
import re
from pathlib import Path

CWD = Path(__file__).parent

## PACKAGE VARIABLES
NAME = "nata"
META_PATH = CWD / "nata" / "__init__.py"
META_FILE = META_PATH.read_text()
KEYWORDS = ["particle-in-cell", "postprocessing", "visualization"]
PROJECT_URLS = {
    "Source Code": "https://github.com/GoLP-IST/nata",
    "Bug Tracker": "https://github.com/GoLP-IST/nata/issues",
    "Documentation": "TODO",
}
CLASSIFIERS = [
    "Development Status :: 3 - Alpha",
    "Intended Audience :: Education",
    "Intended Audience :: Science/Research",
    "Topic :: Scientific/Engineering :: Physics",
    "Topic :: Scientific/Engineering :: Visualization",
    "Natural Language :: English",
    "Operating System :: OS Independent",
    "Programming Language :: Python",
    "Programming Language :: Python :: 3.6",
    "Programming Language :: Python :: 3.7",
    "Programming Language :: Python :: 3.8",
]

# TODO: check license classifier


def parse_requirement_txt(filename):
    """Parse txt file and return a list of strings."""
    path = Path(__file__).parent / "requirements" / filename
    requirements = path.read_text().split("\n")
    while "" in requirements:
        requirements.remove("")
    return requirements


REQUIREMENTS = {
    "base": parse_requirement_txt("base.txt"),
    "tests": parse_requirement_txt("tests.txt"),
    "docs": parse_requirement_txt("docs.txt"),
    "devtools": parse_requirement_txt("devtools.txt"),
}

INSTALL_REQUIRES = REQUIREMENTS["base"]
EXTRAS_REQUIRE = {
    "docs": REQUIREMENTS["docs"],
    "tests": REQUIREMENTS["tests"],
    "dev": REQUIREMENTS["devtools"]
    + REQUIREMENTS["tests"]
    + REQUIREMENTS["docs"],
}

##
def find_in_meta(tag):
    """Finds in nata/__init__.py meta information."""
    match = re.search(
        r"^__{meta}__ = ['\"]([^'\"]*)['\"]".format(meta=tag), META_FILE, re.M
    )
    if match:
        return match.group(1)
    raise RuntimeError(f"Unable to find tag `__{tag}__` in meta file.")


def read(path):
    """Reads the text of file."""
    return path.read_text()


LONG = read(CWD / "README.rst") + "\n" * 2 + read(CWD / "AUTHORS.rst")


if __name__ == "__main__":
    setup(
        name=NAME,
        description=find_in_meta("description"),
        license=find_in_meta("license"),
        url=find_in_meta("url"),
        project_urls=PROJECT_URLS,
        version=find_in_meta("version"),
        author=find_in_meta("author"),
        author_email=find_in_meta("email"),
        maintainer=find_in_meta("author"),
        maintainer_email=find_in_meta("email"),
        keywords=KEYWORDS,
        long_description=LONG,
        long_description_content_type="text/x-rst",
        packages=find_packages(include=["nata"]),
        python_requires=">=3.6, !=2.*",
        classifiers=CLASSIFIERS,
        install_requires=INSTALL_REQUIRES,
        extras_require=EXTRAS_REQUIRE,
        include_package_data=True,
    )
