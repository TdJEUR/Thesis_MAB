from Run_Team_MAB import team_MAB
from Plot_Results import plot_results
import numpy as np


""" Simulate a team playing a MAB with rewards for each arm 
corresponding to true_arm_rewards (list) for a certain number_of_rounds 
(int). The verbosity of each team member is defined by alphas (list of
floats). The exploration-exploitation strategy of the team is defined 
by tau (list of floats), and is increased incrementally to show the effect
of different strategies. To use this script please input the desired values 
below and press run """


# MAB variables
true_arm_rewards = [0.1, 0.8, 0.1]
number_of_rounds = 200
# true_arm_stds = []

# Team variables
alphas = [0.5, 0.5, 0.5, 0.5]
tau = np.linspace(0.05, 2, num=5)

num_sims = 300


def simulate_different_taus():
    # Run simulation with constant alpha and different values of tau
    number_of_taus = len(tau)
    all_acc_rewards = []
    all_rate_of_best_rewards = []
    all_acc_regret = []
    for i in range(num_sims):
        acc_rewards = []
        rate_of_best_rewards = []
        acc_regret = []
        for j in tau:
            res = team_MAB(alphas, j, true_arm_rewards, number_of_rounds)
            acc_rewards.append(res[0])
            rate_of_best_rewards.append(res[1])
            acc_regret.append(res[2])
        all_acc_rewards.append(acc_rewards)
        all_rate_of_best_rewards.append(rate_of_best_rewards)
        all_acc_regret.append(acc_regret)
    avg_acc_rewards = vertical_avg(all_acc_rewards)
    avg_rate_of_best_rewards = vertical_avg(all_rate_of_best_rewards)
    avg_acc_regret = vertical_avg(all_acc_regret)
    # Plot results
    plot_results(number_of_rounds, tau, avg_acc_rewards, avg_rate_of_best_rewards, avg_acc_regret)


def vertical_avg(lst):
    num_sublists = len(lst)
    num_subsublists = len(lst[0])
    result = []
    for i in range(num_subsublists):
        sub_result = []
        for j in range(len(lst[0][0])):
            total = 0
            for k in range(num_sublists):
                total += lst[k][i][j]
            sub_result.append(total / num_sublists)
        result.append(sub_result)
    return result


if __name__ == '__main__':
    simulate_different_taus()
