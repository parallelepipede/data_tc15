#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Jul  4 17:05:20 2020

@author: paul
"""

from setuptools import setup,find_packages


with open("README.md", "r") as fh:
    long_description = fh.read()

setup(
    name="try",
    version="0.0.1",
    author="Paul Zanoncelli",
    author_email="paul.zanoncelli@ecole.ensicaen.fr",
    description="A python library for graph edit distance",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/parallelepipede/data_tc15",

    packages=find_packages(),
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: GNU",
        "Operating System :: OS Independent",
    ],
    python_requires='>=3.6'
    """
    projects_url = {
            'Documentation':"link",
            'Source' : "other link"
    }
    """
)
