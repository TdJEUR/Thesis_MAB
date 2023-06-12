from Sim_MAB_CP import Sim_MAB_CP, Sim_MAB_CP_Regret_Only, Sim_MAB_CP_Reward_Only
from Sim_MAB_Beliefs import Sim_MAB_Beliefs, Sim_MAB_Beliefs_Regret_Only, Sim_MAB_Beliefs_Reward_Only
from Helpers import vertical_avg
import numpy as np
import matplotlib.pyplot as plt


def Single_MAB_Multiple_Sims(MAB_Matrix, best_arm, num_sims, taus, alphas):
    """ Simulate a team playing a Multi Armed Bandit with the mean rewards for each arm
    corresponding to true_arm_rewards (list) and standard deviation for each arm
    corresponding to true_arm_stds (list). The team plays for a certain number_of_rounds
    (int). The verbosity of each team member is defined by alphas (list of floats). The
    different exploration-exploitation strategies of the team are defined by taus (list
    of floats). The number of simulations can be determined using num_sims (int) """
    count = 0
    number_of_rounds = len(MAB_Matrix)
    # Initialize total average acc reward, rate of best reward and total acc regret
    all_acc_rewards_cp = []
    all_acc_rewards_b = []
    all_rate_of_best_rewards_cp = []
    all_rate_of_best_rewards_b = []
    all_acc_regret_cp = []
    all_acc_regret_b = []
    # For each simulation:
    for i in range(num_sims):
        # Initialize acc reward, rate of best reward and total acc regret
        acc_rewards_cp = []
        acc_rewards_b = []
        rate_of_best_rewards_cp = []
        rate_of_best_rewards_b = []
        acc_regret_cp = []
        acc_regret_b = []
        # For each strategy (tau):
        for tau in taus:
            # Complete simulation of team playing Multi Armed Bandit
            res_cp = Sim_MAB_CP(alphas, tau, MAB_Matrix, best_arm)
            res_b = Sim_MAB_Beliefs(alphas, tau, MAB_Matrix, best_arm)
            # Add data for each strategy to current simulation data
            acc_rewards_cp.append(res_cp[0])
            acc_rewards_b.append(res_b[0])
            rate_of_best_rewards_cp.append(res_cp[1])
            rate_of_best_rewards_b.append(res_b[1])
            acc_regret_cp.append(res_cp[2])
            acc_regret_b.append(res_b[2])
        # Add data for each simulation to total data set
        all_acc_rewards_cp.append(acc_rewards_cp)
        all_acc_rewards_b.append(acc_rewards_b)
        all_rate_of_best_rewards_cp.append(rate_of_best_rewards_cp)
        all_rate_of_best_rewards_b.append(rate_of_best_rewards_b)
        all_acc_regret_cp.append(acc_regret_cp)
        all_acc_regret_b.append(acc_regret_b)
        # Update Timer
        count += 1
        # print(f'{count*100/num_sims}%')
    # Average data over all simulations to obtain total average acc reward, rate of best reward and total acc regret
    avg_acc_rewards_cp = vertical_avg(all_acc_rewards_cp)
    avg_acc_rewards_b = vertical_avg(all_acc_rewards_b)
    avg_rate_of_best_rewards_cp = vertical_avg(all_rate_of_best_rewards_cp)
    avg_rate_of_best_rewards_b = vertical_avg(all_rate_of_best_rewards_b)
    avg_acc_regret_cp = vertical_avg(all_acc_regret_cp)
    avg_acc_regret_b = vertical_avg(all_acc_regret_b)

    return avg_acc_regret_b, avg_acc_rewards_b, avg_rate_of_best_rewards_b, avg_acc_regret_cp, avg_acc_rewards_cp, avg_rate_of_best_rewards_cp


def Single_MAB_Multiple_Sims_Regret_Only(MAB_Matrix, best_arm, num_sims, taus, alphas):
    count = 0
    # Initialize total average acc reward, rate of best reward and total acc regret
    all_acc_regret_cp = []
    all_acc_regret_b = []
    # For each simulation:
    for i in range(num_sims):
        # Initialize acc reward, rate of best reward and total acc regret
        acc_regret_cp = []
        acc_regret_b = []
        # For each strategy (tau):
        for tau in taus:
            # Complete simulation of team playing Multi Armed Bandit
            res_cp = Sim_MAB_CP_Regret_Only(alphas, tau, MAB_Matrix, best_arm)
            res_b = Sim_MAB_Beliefs_Regret_Only(alphas, tau, MAB_Matrix, best_arm)
            # Add data for each strategy to current simulation data
            acc_regret_cp.append(res_cp)
            acc_regret_b.append(res_b)
        # Add data for each simulation to total data set
        all_acc_regret_cp.append(acc_regret_cp)
        all_acc_regret_b.append(acc_regret_b)
        # Update Timer
        count += 1
    # Average data over all simulations to obtain total average acc reward, rate of best reward and total acc regret
    avg_acc_regret_cp = np.mean(np.array(all_acc_regret_cp), axis=0)
    avg_acc_regret_b = np.mean(np.array(all_acc_regret_b), axis=0)
    return avg_acc_regret_b, avg_acc_regret_cp


def Single_MAB_Multiple_Sims_Reward_Only(MAB_Matrix, best_arm, num_sims, taus, alphas):
    count = 0
    # Initialize total average acc reward, rate of best reward and total acc regret
    all_acc_reward_cp = []
    all_acc_reward_b = []
    # For each simulation:
    for i in range(num_sims):
        # Initialize acc reward, rate of best reward and total acc regret
        acc_reward_cp = []
        acc_reward_b = []
        # For each strategy (tau):
        for tau in taus:
            # Complete simulation of team playing Multi Armed Bandit
            res_cp = Sim_MAB_CP_Reward_Only(alphas, tau, MAB_Matrix, best_arm)
            res_b = Sim_MAB_Beliefs_Reward_Only(alphas, tau, MAB_Matrix, best_arm)
            # Add data for each strategy to current simulation data
            acc_reward_cp.append(res_cp)
            acc_reward_b.append(res_b)
        # Add data for each simulation to total data set
        all_acc_reward_cp.append(acc_reward_cp)
        all_acc_reward_b.append(acc_reward_b)
        # Update Timer
        count += 1
    # Average data over all simulations to obtain total average acc reward, rate of best reward and total acc regret
    avg_acc_reward_cp = np.mean(np.array(all_acc_reward_cp), axis=0)
    avg_acc_reward_b = np.mean(np.array(all_acc_reward_b), axis=0)
    return avg_acc_reward_b, avg_acc_reward_cp
