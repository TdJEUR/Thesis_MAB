from Run_Team_MAB import team_MAB
from Plot_results import plot_results
import numpy as np


""" Simulate a team player playing a MAB with rewards for each arm 
corresponding to true_arm_rewards (list) for a certain number_of_rounds 
(int). The verbosity of each team member is defined by alphas (list of
floats). The exploration-exploitation strategy of the team is defined 
by tau (list of floats), and is increased incrementally to show the effect
of different strategies. To use this script please input the desired values 
below and press run """


# MAB variables
true_arm_rewards = [0.1, 0.8, 0.1]
number_of_rounds = 50
# true_arm_stds = []

# Team variables
alphas = [0.5, 0.5, 0.5, 0.5]
tau = np.linspace(0.01, 2, num=5)


def simulate_different_taus():
    # Run simulation with constant alpha and different values of tau
    acc_rewards = []
    rate_of_best_rewards = []
    acc_regret = []
    for i in tau:
        res = team_MAB(alphas, i, true_arm_rewards, number_of_rounds)
        acc_rewards.append(res[0])
        rate_of_best_rewards.append(res[1])
        acc_regret.append(res[2])
    # Plot results
    plot_results(number_of_rounds, acc_rewards, rate_of_best_rewards, acc_regret)


if __name__ == '__main__':
    simulate_different_taus()