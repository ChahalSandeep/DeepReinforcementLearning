# !/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
action_value_methods.py

"""

# Built-in/standard library
import copy
import pathlib

# Third/Other party packages


# Owned/local source
from ActionValueMethods.action_value_methods import GreedyKArmedBandit as Avm
from utilities.generate_chart_for_list_comparision import generate_chart_three_list_scalar_values as generate_chart
from utilities.date_time_to_string import datetime_to_str as unique_name

__author__ = "Sandeep Chahal"
__email__ = "sandeep.chahal@mavs.uta.edu", "chahal.sdp@gmail.com"


avm_obj = Avm(k=5, n_iter=1000)
avm1 = copy.copy(avm_obj)   # shallow copy
avm2 = copy.copy(avm_obj)
avm3 = copy.copy(avm_obj)

# exploration = 0.1
avm1.epsilon = 0.1
avm1.run()
# mean_reward_overtime

# exploration = 0.5
avm2.epsilon = 0.5
avm2.run()

# exploration = 0
avm3.epsilon = 0
avm3.run()

generate_chart(lst1=avm1.mean_reward_overtime, lst2=avm2.mean_reward_overtime, lst3=avm3.mean_reward_overtime,
               lst1_label="exploration=0.1", lst2_label="exploration=0.5", lst3_label="exploration=0",
               fig_y_label="reward", fig_title="different exploration_action_value_method",
               image_loc= str(pathlib.Path().absolute()) + '/figures/' + unique_name() + '.png')
