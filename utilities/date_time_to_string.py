# !/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
multi_armed_testbed.py
contains multi armed bandit test bud suite
"""

# Built-in/standard library
import re
from datetime import datetime as dt

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
    dt_string = str(dt.now())

    for r in ((".", "_"), (":", "_"), (" ", "_"), ("-", "_")):
        dt_string = dt_string.replace(*r)

    # rep = {"cond1": ".", "cond2": ":", "cond3": "-", "cond4": " "}
    # rep = dict((re.escape(k), v) for k, v in rep.items())
    # pattern = re.compile("|".join(rep.keys()))
    # return pattern.sub(lambda m: rep[re.escape(m.group(0))], str(dt.now()))
    return dt_string