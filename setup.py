#!/bin/env python

#######################################################################  NEEDS TO BE CHANGED!!!!!!!
# Copyright (C) 2019 Julian Dosch
#
# This file is part of greedyFAS.
#
#  greedyFAS is free software: you can redistribute it and/or modify
#  it under the terms of the GNU General Public License as published by
#  the Free Software Foundation, either version 3 of the License, or
#  (at your option) any later version.
#
#  greedyFAS is distributed in the hope that it will be useful,
#  but WITHOUT ANY WARRANTY; without even the implied warranty of
#  MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
#  GNU General Public License for more details.
#
#  You should have received a copy of the GNU General Public License
#  along with greedyFAS.  If not, see <http://www.gnu.org/licenses/>.
#
#######################################################################

from setuptools import setup, find_packages

with open("README.md", "r") as input:
    long_description = input.read()

setup(
    name="fasml",
    version="0.1",
    python_requires='>=3.9.0',
    description="",
    long_description=long_description,
    long_description_content_type="text/markdown",
    author="",
    author_email="",
    url="",
    packages=find_packages(),
    package_data={'': ['*']},
    install_requires=[
        'pandas',
        'tensorflow',
    ],
    entry_points={
        'console_scripts': ["fasml.input_gen2d = fasml.file_handling.input_gen_2d:get_args",
                            "fasml.query_gen2d = fasml.file_handling.input_gen_2d:query_gen_entry",
                            "fasml.create_dense = fasml.create_dense_NN:get_args",
                            "fasml.apply_dense = fasml.apply_dense_NN:get_args"
                            ],
    },
    license="GPL-3.0",
    classifiers=[
        "Environment :: Console",
        "Development Status :: 3 - Alpha",
        "Intended Audience :: End Users/Desktop",
        "License :: OSI Approved :: GNU General Public License v3 or later (GPLv3+)",
        "Natural Language :: English",
        "Programming Language :: Python :: 3",
    ],
)
