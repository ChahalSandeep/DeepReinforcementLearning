# !/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
multi_armed_testbed.py
contains multi armed bandit test bud suite
"""

# Built-in/standard library
from datetime import datetime

# Third/Other party packages
import matplotlib.pyplot as plt

# Owned/local source

__author__ = "Sandeep Chahal"
__email__ = "sandeep.chahal@mavs.uta.edu", "chahal.sdp@gmail.com"


def datetime_to_str():
    """
    converts current date time string can be used for unique names
    :return: str
    """
    dt = str(datetime.now()).replace(' ', '-')
    dt = dt.replace(':', '-')

    # for r in ((".", "_"), (":", "_"), (" ", "_"), ("-", "_")):
    #     dt_string = dt_string.replace(*r)
    return dt.replace('.', '-')