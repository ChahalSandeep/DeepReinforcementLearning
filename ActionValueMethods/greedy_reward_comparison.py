# !/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
action_value_methods.py

"""

# Built-in/standard library
# import copy
# import pathlib
# import sys
# import os
import argparse

# Third/Other party packages


# Owned/local source
from ActionValueMethods.action_value_methods import GreedyKArmedBandit as Avm
# from utilities.generate_chart_for_list_comparision import generate_chart_three_list_scalar_values as generate_chart
# from utilities.date_time_to_string import datetime_to_str as unique_name
# from utilities.disable_enable_terminal_print import disable_print

__author__ = "Sandeep Chahal"
__email__ = "sandeep.chahal@mavs.uta.edu", "chahal.sdp@gmail.com"

# number_of_arms = 10
# number_of_steps = 1000
# number_of_runs = 1000
distribution = 'same'  # 'same'|'unique'


class GreedyRewardComparison(object):

    def __init__(self, total_arms=10, total_iter=1000, total_runs=1000, total_agents=None, same_means=True):
        self.total_arms, self.total_steps, self.total_runs = total_arms, total_iter, total_runs
        self.total_agents, self.same_means = total_agents, same_means
        # todo take this also as and argument
        self.exploration_list = list(range(0, total_agents-1))
        self.exploration_list = [x / (total_agents *10) for x in self.exploration_list]
        self.exploration_list.append(1.0)
        # create an base arm
        self.base_arm = Avm(k=self.total_arms, n_iter=self.total_steps)
        self.base_mean = self.base_arm.means
        self._define_arms()
        self.run_comparison()
        # todo generate charts

    def _define_arms(self):
        if self.total_agents is None:
            raise ValueError("total_agents cant be None")
        if not isinstance(self.total_agents, int):
            raise ValueError("total_agents has to be int")
        self.agents  = {}
        for ind in range(self.total_agents):
            self.agents [str(ind)] = Avm(k=self.total_arms, n_iter=self.total_steps)
            self.agents [str(ind)].epsilon = self.exploration_list[ind]
            if self.same_means:
                self.agents [str(ind)].means = self.base_arm.means

    def one_run_reward(self):
        ...

    def run_comparison(self):
        avg_dict = dict()
        for x in range(self.total_runs):
            for ag in self.agents.keys():
                self.agents[ag].run(print_info=False)
                print("arms: " + ag + " real_mean: ", sum(self.agents[ag].means)/len(self.agents[ag].means), " result: ",
                      self.agents[ag].mean_reward, " exp: ", self.agents[ag].epsilon)
        print("here")


def greedy_reward_comp(total_arms, total_iter, total_runs, total_agents):
    GreedyRewardComparison(total_arms=total_arms, total_iter=total_iter, total_runs=total_runs,
                           total_agents=total_agents)

    # # create action value object
    # avm_obj = Avm(k=trial_k, n_iter=trial_n_iter)
    # # get means of armed bandit object
    # means = avm_obj.means
    #
    # avm_1_run_reward = [0]
    # avm_2_run_reward = [0]
    # avm_3_run_reward = [0]
    #
    # for x in range(runs):
    #     avm1 = Avm(k=trial_k, n_iter=trial_n_iter)  # shallow copy
    #     avm2 = Avm(k=trial_k, n_iter=trial_n_iter)
    #     avm3 = Avm(k=trial_k, n_iter=trial_n_iter)
    #     avm1.means = avm2.means = avm3.means = means
    #     # exploration = 0
    #     avm1.epsilon = 0
    #     avm1.run()
    #
    #     # exploration = 0.5
    #     avm2.epsilon = 0.01
    #     avm2.run()
    #
    #     # exploration = 0
    #     avm3.epsilon = 0.1
    #     avm3.run()
    #
    #     avm_1_run_reward.append(avm1.mean_reward_overtime[-1])
    #     avm_2_run_reward.append(avm2.mean_reward_overtime[-1])
    #     avm_3_run_reward.append(avm3.mean_reward_overtime[-1])
    #
    # generate_chart(lst1=avm_1_run_reward, lst2=avm_2_run_reward, lst3=avm_3_run_reward,
    #                lst1_label="exploration=0", lst2_label="exploration=0.01", lst3_label="exploration=0.1",
    #                fig_y_label="reward", fig_title="different exploration_action_value_method",
    #                image_loc= str(pathlib.Path().absolute()) + '/figures/runs_' + unique_name() + '.png')


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('-ta', '--total_arms', type=int, default=10, help="number of arms (trial_k)")
    parser.add_argument('-tit', '--total_iter', type=int, default=10000, help="number of iterations/steps")
    parser.add_argument('-tr', '--total_runs', type=int, default=1, help="number of runs")
    parser.add_argument('-tag', '--total_agents', type=int, default=5, help="number of runs")
    # parser.add_argument('-tag', '--total_agents', type=int, default=1000, help="number of runs")
    # todo add explorations list
    args = parser.parse_args()
    greedy_reward_comp(total_arms=args.total_arms, total_iter=args.total_iter, total_runs=args.total_runs,
                       total_agents=args.total_agents)


if __name__ == '__main__':
    main()
