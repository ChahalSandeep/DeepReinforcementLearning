# !/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
multi_armed_testbed.py
contains multi armed bandit test bud suite
"""

# Built-in/standard library
import pathlib

# Third/Other party packages
import matplotlib.pyplot as plt

# Owned/local source
from utilities.utilities import datetime_to_str as unique_name

__author__ = "Sandeep Chahal"
__email__ = "sandeep.chahal@mavs.uta.edu", "chahal.sdp@gmail.com"


def generate_chart_two_list_scalar_values(lst1, lst2, lst1_label="list_1", lst2_label="list_2",
                                          fig_x_label="Index", fig_y_label="Values",
                                          fig_title="comparisons of list"):
    if len(lst1) != len(lst2):
        raise Warning('lst1, lst2, lst3 are not equal!')

    if not isinstance(fig_x_label, str):
        raise ValueError("label must be string")

    if not isinstance(fig_y_label, str):
        raise ValueError("label must be string")

    if not isinstance(fig_title, str):
        raise ValueError("label must be string")

    if not isinstance(lst1_label, str):
        raise ValueError("lst1_label must be string")

    if not isinstance(lst2_label, str):
        raise ValueError("lst2_label must be string")

    plt.plot(lst1, label=lst1_label)
    plt.plot(lst2, label=lst2_label)
    plt.xlabel(fig_x_label)
    plt.ylabel(fig_y_label)
    plt.title(fig_title)
    plt.show()


def generate_chart_three_list_scalar_values(lst1, lst2, lst3,
                                            lst1_label="list_1", lst2_label="list_2", lst3_label="list_3",
                                            fig_x_label="Index", fig_y_label="Values",
                                            fig_title="comparisons of list", image_loc=None):
    if len(lst1) != len(lst2) != len(lst2):
        raise ValueError('lst1, lst2, lst3 are not equal!')

    if not isinstance(fig_x_label, str):
        raise ValueError("label must be string")

    if not isinstance(fig_y_label, str):
        raise ValueError("label must be string")

    if not isinstance(fig_title, str):
        raise ValueError("label must be string")

    if not isinstance(lst1_label, str):
        raise ValueError("lst1_label must be string")

    if not isinstance(lst2_label, str):
        raise ValueError("lst2_label must be string")

    if not isinstance(lst3_label, str):
        raise ValueError("lst2_label must be string")

    if image_loc is None:
        location_and_name_of_image = str(pathlib.Path().absolute()) + unique_name() + '.png'
    else:
        location_and_name_of_image = image_loc

    index_fig = range(0, len(lst1))
    plt.plot(index_fig, lst1, label=lst1_label)
    plt.plot(index_fig, lst2, label=lst2_label)
    plt.plot(index_fig, lst3, label=lst3_label)
    plt.xlabel(fig_x_label)
    plt.ylabel(fig_y_label)
    plt.title(fig_title)
    plt.legend()
    plt.savefig(location_and_name_of_image)
