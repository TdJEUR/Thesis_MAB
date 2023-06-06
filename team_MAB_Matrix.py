from Old_Create_Team_and_MAB import create_team, create_MAB
from Combine_Information import generate_team_choice_prob
import numpy as np


def team_MAB_Matrix(alphas, tau, MAB_Matrix, best_arm):
    """ Simulate a team consisting of members defined by alphas (list containing alpha
    for each member) playing a MAB with rewards corresponding to true_arm_rewards (list
    containing reward of each arm) for a certain number_of_rounds (integer). The exploration-
    exploitation strategy the team takes is defined by tau (float) """
    number_of_arms = len(MAB_Matrix[0])
    number_of_rounds = len(MAB_Matrix)
    total_reward = 0
    total_regret = 0
    number_of_best_arm_pulls = 0
    accumulated_reward = []
    accumulated_regret = []
    rate_of_best_arm = []
    # Create MAB model
    # mab = create_MAB(true_arm_rewards, true_arm_stds)
    # Create Team model
    team = create_team(alphas, tau, number_of_arms)
    # Loop through the rounds:
    round = 0
    for i in range(number_of_rounds):
        round += 1
        # Generate individual choice probabilities
        for j, member in enumerate(team, 1):
            member.get_choice_probabilities()
        # Generate team choice probabilities
        team_choice_prob = generate_team_choice_prob(team)
        # Generate which arm is chosen out of the probabilities
        choice = np.random.choice(list(range(number_of_arms)), p=team_choice_prob)
        # Play a round depending on the choice
        reward = MAB_Matrix[i][choice]
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
    # final_cd = mab.get_choices_distribution()
    # print(f"Distribution of picked arms: {final_cd}")
    return accumulated_reward, rate_of_best_arm, accumulated_regret
