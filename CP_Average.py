from CP_Team_Generation import create_team
from Helpers import vertical_avg
from Draw_Results import plot_results_constant_tau
from Combine_Information import generate_team_choice_prob
from Generate_MAB_Matrix import generate_MAB_Matrix
import numpy as np
import random


def play_Multiple_MABs_Avg_CP(number_of_trials, number_of_rounds, num_MABs,
                              probabilities_of_success, tau, alphas, num_sims):
    # Initialize total average acc reward, rate of best reward and total acc regret over all MABs:
    final_acc_rewards = []
    final_rate_of_best_rewards = []
    final_acc_regret = []
    MABs = []
    # For each different MAB problem:
    number_of_arms = len(probabilities_of_success)
    for i in range(num_MABs):
        print(f"MAB problem: {i+1}")
        # Generate probs of success for MAB Matrix:
        specific_probs_of_success = []
        for j in range(len(probabilities_of_success)):
            specific_probs_of_success.append(random.uniform(probabilities_of_success[j] - 0.05,
                                                            probabilities_of_success[j] + 0.05))
        MABs.append(specific_probs_of_success)
        # Generate MAB Matrix:
        MAB_Matrix = generate_MAB_Matrix(number_of_trials=number_of_trials,
                                         probabilities_of_success=specific_probs_of_success,
                                         number_of_rounds=number_of_rounds)
        # Initialize average acc reward, average rate of best reward and average acc regret for single MAB problem:
        all_acc_rewards = []
        all_rate_of_best_rewards = []
        all_acc_regret = []
        # Simulate single MAB problem multiple times:
        for _ in range(num_sims):
            # Initialize acc reward, rate of best reward and total acc regret:
            acc_rewards = []
            rate_of_best_rewards = []
            acc_regret = []
            # For each composition (alphas):
            for alpha in alphas:
                # Complete simulation of team playing Multi Armed Bandit:
                team = create_team(alpha, tau, number_of_arms)
                res = play_single_MAB_Matrix_Avg_CP(MAB_Matrix=MAB_Matrix,
                                                    team=team,
                                                    probabilities_of_success=probabilities_of_success,
                                                    number_of_rounds=number_of_rounds)
                # Add data for each strategy to current simulation data:
                acc_rewards.append(res[0])
                rate_of_best_rewards.append(res[1])
                acc_regret.append(res[2])
            # Add data for each simulation to total data set:
            all_acc_rewards.append(acc_rewards)
            all_rate_of_best_rewards.append(rate_of_best_rewards)
            all_acc_regret.append(acc_regret)
        # Average data over all simulations to obtain total average acc reward, rate of best reward and total acc regret
        avg_acc_rewards = vertical_avg(all_acc_rewards)
        avg_rate_of_best_rewards = vertical_avg(all_rate_of_best_rewards)
        avg_acc_regret = vertical_avg(all_acc_regret)
        # Add averaged data to final dataset:
        final_acc_rewards.append(avg_acc_rewards)
        final_rate_of_best_rewards.append(avg_rate_of_best_rewards)
        final_acc_regret.append(avg_acc_regret)
    # Average data over all different MAB problems:
    final_acc_rewards = vertical_avg(final_acc_rewards)
    final_rate_of_best_rewards = vertical_avg(final_rate_of_best_rewards)
    final_acc_regret = vertical_avg(final_acc_regret)
    # Plot results:
    print(f"Values for MABs: {MABs}")
    plot_results_constant_tau(number_of_rounds, alphas, final_acc_rewards, final_rate_of_best_rewards,
                              final_acc_regret, tau, num_sims, 0)


def play_single_MAB_Matrix_Avg_CP(MAB_Matrix, team, probabilities_of_success, number_of_rounds):
    # Initialise results to empty lists
    total_reward = 0
    total_regret = 0
    accumulated_reward = []
    accumulated_regret = []
    rate_of_best_arm = []
    number_of_best_arm_pulls = 0
    round = 0
    best_arm = probabilities_of_success.index(max(probabilities_of_success))  # Only one max arm!
    number_of_arms = len(probabilities_of_success)
    # Loop through the rounds:
    for i in range(number_of_rounds):
        # Convert each individual member's beliefs to choice probabilities (softmax)
        for member in team:
            member.get_choice_probabilities()
        # Combine individual choice probabilities into team choice probabilities
        team_choice_prob = generate_team_choice_prob(team)
        # Generate which arm is chosen out of the probabilities
        choice = np.random.choice(list(range(number_of_arms)), p=team_choice_prob)
        # Play a round depending on the choice
        reward = MAB_Matrix[i][choice]
        round += 1
        total_reward += reward
        accumulated_reward.append(total_reward)
        total_regret += max(MAB_Matrix[i]) - reward
        accumulated_regret.append(total_regret)
        if choice == best_arm:
            number_of_best_arm_pulls += 1
        rate_of_best_arm.append(number_of_best_arm_pulls / round)
        # Update beliefs of all team members
        for member in team:
            member.update_beliefs(choice, reward)
    return accumulated_reward, rate_of_best_arm, accumulated_regret
