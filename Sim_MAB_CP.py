from Create_Team_and_MAB import create_team
from Combine_Information import generate_team_choice_prob
import numpy as np


def Sim_MAB_CP(alphas, tau, MAB_Matrix, best_arm):
    number_of_arms = len(MAB_Matrix[0])
    number_of_rounds = len(MAB_Matrix)
    total_reward = 0
    total_regret = 0
    number_of_best_arm_pulls = 0
    accumulated_reward = []
    accumulated_regret = []
    rate_of_best_arm = []
    # Create Team
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
            # print(f'Old beliefs: {member.belief}')
            member.update_beliefs(choice, reward)
            # print(f'New beliefs: {member.belief} \n')
    return accumulated_reward, rate_of_best_arm, accumulated_regret
