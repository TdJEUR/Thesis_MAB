import matplotlib.pyplot as plt
import numpy as np
from team_MAB_Matrix import team_MAB_Matrix
import pandas as pd
from Draw_Results import plot_results_constant_alphas
from Helpers import vertical_avg


def generate_MAB_Matrix(number_of_trials, probabilities_of_success, number_of_rounds):
    number_of_arms = len(number_of_trials)
    best_arm = np.argmax(probabilities_of_success)
    MAB_Matrix = np.random.binomial(n=number_of_trials,
                                    p=probabilities_of_success,
                                    size=[number_of_rounds, number_of_arms])
    return MAB_Matrix, best_arm


# MAB variables
# true_arm_rewards = [0.5, 0.8, 0.3, 0.5]
# true_arm_stds = [0.4, 0.3, 0.1, 0.3, 0.6]
# true_arm_stds = [0, 0, 0, 0]
# number_of_rounds = 200

# Input MAB Variables
number_of_trials = [1, 1, 1, 1]
number_of_rounds = 100
probabilities_of_success = [0.5, 0.8, 0.3, 0.5]
no_arms = len(number_of_trials)


# Team variables
alphas = [0.05, 0.05, 0.05, 0.05]
taus = np.linspace(0.1, 1, num=5)


# Simulation variables
num_sims = 50


MAB_Matrix, best_arm = generate_MAB_Matrix(number_of_trials, probabilities_of_success, number_of_rounds)

def simulate_constant_alphas():
    """ Simulate a team playing a Multi Armed Bandit with the mean rewards for each arm
    corresponding to true_arm_rewards (list) and standard deviation for each arm
    corresponding to true_arm_stds (list). The team plays for a certain number_of_rounds
    (int). The verbosity of each team member is defined by alphas (list of floats). The
    different exploration-exploitation strategies of the team are defined by taus (list
    of floats). The number of simulations can be determined using num_sims (int) """
    # Initialize total average acc reward, rate of best reward and total acc regret
    all_acc_rewards = []
    all_rate_of_best_rewards = []
    all_acc_regret = []
    # For each simulation:
    for i in range(num_sims):
        # Initialize acc reward, rate of best reward and total acc regret
        acc_rewards = []
        rate_of_best_rewards = []
        acc_regret = []
        # For each strategy (tau):
        for tau in taus:
            # Complete simulation of team playing Multi Armed Bandit
            res = team_MAB_Matrix(alphas, tau, MAB_Matrix, best_arm)
            # Add data for each strategy to current simulation data
            acc_rewards.append(res[0])
            rate_of_best_rewards.append(res[1])
            acc_regret.append(res[2])
        # Add data for each simulation to total data set
        all_acc_rewards.append(acc_rewards)
        all_rate_of_best_rewards.append(rate_of_best_rewards)
        all_acc_regret.append(acc_regret)
    # Average data over all simulations to obtain total average acc reward, rate of best reward and total acc regret
    avg_acc_rewards = vertical_avg(all_acc_rewards)
    avg_rate_of_best_rewards = vertical_avg(all_rate_of_best_rewards)
    avg_acc_regret = vertical_avg(all_acc_regret)
    # Plot results
    plot_results_constant_alphas(number_of_rounds, taus, avg_acc_rewards, avg_rate_of_best_rewards,
                                 avg_acc_regret, alphas, num_sims)


if __name__ == '__main__':
    simulate_constant_alphas()
    plt.show()
