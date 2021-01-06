# !/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
action_value_methods.py

"""

# Built-in/standard library
# import copy
# import pathlib
# import sys
import os
import argparse

# Third/Other party packages
import matplotlib.pyplot as plt

# Owned/local source
from ActionValueMethods.action_value_methods import GreedyKArmedBandit as Avm
from utilities.utilities import datetime_to_str
from utilities.disable_enable_terminal_print import disable_print, enable_print
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
        self.time_id = datetime_to_str()
        self.total_arms, self.total_steps, self.total_runs = total_arms, total_iter, total_runs
        self.total_agents, self.same_means = total_agents, same_means
        if total_agents > 3:
            self.exploration_list = list(range(0, total_agents-2))
            self.exploration_list = [x / (total_agents * 10) for x in self.exploration_list]
            self.exploration_list.append(0.01)
            self.exploration_list.append(1.0)
        elif total_agents == 3:
            self.exploration_list = [0, 0.01, 0.1]
        elif total_agents == 2:
            self.exploration_list[0, 0.01]
        else:
            exit("please use more than 1 agent")
        # create an base arm
        print("here is an example of input parameters might vary")
        self.base_arm = Avm(k=self.total_arms, n_iter=self.total_steps)
        self.base_mean = self.base_arm.means
        disable_print()
        self._define_arms()
        enable_print()
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

    def run_comparison(self):
        for x in range(self.total_runs):
            print("================== Run " + str(x) + " ================== ")
            for ag in self.agents.keys():
                self.agents[ag].run(print_info=False)
                print("arms: " + ag + " real_mean: ", sum(self.agents[ag].means)/len(self.agents[ag].means), " result: ",
                      self.agents[ag].mean_reward, " exp: ", self.agents[ag].epsilon)
            self.visualize_runs(x)

    def visualize_runs(self, curr_run):
        # mean_reward_overtime[-1] = final avg reward
        ind_ch = list(range(0, len(self.agents['0'].mean_reward_overtime), 1))
        for ag in self.agents.keys():
            # plt.scatter(ind_ch, self.agents[ag].mean_reward_overtime)
            plt.plot(ind_ch, self.agents[ag].mean_reward_overtime, label="explr = " + str(self.agents[ag].epsilon))
            plt.legend()
        plt.title("Average Reward per agent overtime")
        plt.xlabel("iterations")
        plt.ylabel("reward")
        dir_path = './figures/' + self.time_id + '/'
        if not os.path.exists(dir_path):
            os.makedirs(dir_path)
        plt.savefig(dir_path + str(curr_run) + '_run_average_reward.png')


def greedy_reward_comp(total_arms, total_iter, total_runs, total_agents):
    GreedyRewardComparison(total_arms=total_arms, total_iter=total_iter, total_runs=total_runs,
                           total_agents=total_agents)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('-ta', '--total_arms', type=int, default=10, help="number of arms (trial_k)")
    parser.add_argument('-tit', '--total_iter', type=int, default=2000, help="number of iterations/steps")
    parser.add_argument('-tr', '--total_runs', type=int, default=1, help="number of runs")
    parser.add_argument('-tag', '--total_agents', type=int, default=5, help="number of runs")
    # parser.add_argument('-tag', '--total_agents', type=int, default=1000, help="number of runs")
    # todo add explorations list
    args = parser.parse_args()
    greedy_reward_comp(total_arms=args.total_arms, total_iter=args.total_iter, total_runs=args.total_runs,
                       total_agents=args.total_agents)


if __name__ == '__main__':
    main()
