from Old_Run_Team_MAB import team_MAB
from Draw_Results import plot_results_constant_tau
from Helpers import vertical_avg
import numpy as np


# MAB variables
true_arm_rewards = [0.1, 0.1, 0.1, 0.1, 0.9]
# true_arm_stds = [0.4, 0.3, 0.1, 0.3, 0.6]
true_arm_stds = [0, 0, 0, 0, 0]
number_of_rounds = 100


# Team variables
alphas = [[0.05, 0.05, 0.1, 0.1],
          [0.8, 0.8, 0.8, 0.8],
          [0.5, 0.5, 0.5, 0.5],
          [0.1, 0.1, 0.8, 0.8],
          [0.5, 0.5, 0.8, 0.8]]
tau = 0.75

# Simulation variables
num_sims = 500


def simulate_constant_tau():
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
        # For each strategy (alpha):
        for alpha in alphas:
            # Complete simulation of team playing Multi Armed Bandit
            res = team_MAB(alpha, tau, true_arm_rewards, true_arm_stds, number_of_rounds)
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
    plot_results_constant_tau(number_of_rounds, alphas, avg_acc_rewards, avg_rate_of_best_rewards,
                              avg_acc_regret, tau, num_sims)


if __name__ == '__main__':
    simulate_constant_tau()
