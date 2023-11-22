#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Nov 21 19:08:33 2023

@author: mariano
"""

# setup.py

from setuptools import setup
from Cython.Build import cythonize

setup(
    ext_modules=cythonize("recursive_fir_filter.pyx"),
)
