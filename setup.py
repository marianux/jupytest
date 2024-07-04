#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jun 10 09:29:10 2024

@author: mariano
"""

# setup.py

from setuptools import setup
from Cython.Build import cythonize
import numpy as np

setup(
    ext_modules = cythonize(["promediador_rt_init.pyx", "promediador_rt.pyx", "filtro_peine_DCyArmonicas.pyx"]),
    include_dirs=[np.get_include()]
)
