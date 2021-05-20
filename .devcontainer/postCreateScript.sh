#!/bin/bash

echo -e "\e[104m>>> Installing nata \e[0m"
poetry install

echo -e "\e[104m>>> Installing Jupyter tooling \e[0m"
poetry install -E jupyter

echo -e "\e[104m>>> Installing ipdb \e[0m"
poetry run pip install ipdb

echo -e "\e[104m>>> Installing pytest-watch \e[0m"
poetry run pip install pytest-watch

echo -e "\e[104m>>> Installing graphviz \e[0m"
poetry run pip install graphviz

echo -e "\e[104m>>> Installing h5glance \e[0m"
poetry run pip install h5glance
