# !/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
disable_enable_terminal_print.py
responsible for enabling and disabling print on terminal/console
"""

# Built-in/standard library
import sys
import os

# Third/Other party packages


# Owned/local source


__author__ = "Sandeep Chahal"
__email__ = "sandeep.chahal@mavs.uta.edu", "chahal.sdp@gmail.com"


# Disable
def disable_print():
    sys.stdout = open(os.devnull, 'w')


# Restore
def enable_print():
    sys.stdout = sys.__stdout__
