from Combine_Information import generate_team_belief
from Create_Team_and_MAB import create_team
import numpy as np
from Helpers import softmax_belief_to_prob


def Sim_MAB_Beliefs(alphas, tau, MAB_Matrix, best_arm):
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
        # Combine beliefs
        team_belief = generate_team_belief(team)
        # Convert Team beliefs to Team CP:
        team_choice_prob = softmax_belief_to_prob(team_belief, tau)
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
    return accumulated_reward, rate_of_best_arm, accumulated_regret


def Sim_MAB_Beliefs_Regret_Only(alphas, tau, MAB_Matrix, best_arm):
    number_of_arms = len(MAB_Matrix[0])
    number_of_rounds = len(MAB_Matrix)
    total_regret = 0
    accumulated_regret = []
    # Create Team
    team = create_team(alphas, tau, number_of_arms)
    # Loop through the rounds:
    round = 0
    for i in range(number_of_rounds):
        round += 1
        # Combine beliefs
        team_belief = generate_team_belief(team)
        # Convert Team beliefs to Team CP:
        team_choice_prob = softmax_belief_to_prob(team_belief, tau)
        # Generate which arm is chosen out of the probabilities
        choice = np.random.choice(list(range(number_of_arms)), p=team_choice_prob)
        # Play a round depending on the choice
        reward = MAB_Matrix[i][choice]
        total_regret += max(MAB_Matrix[i]) - reward
        accumulated_regret.append(total_regret)
        # Update beliefs of all team members
        for member in team:
            member.update_beliefs(choice, reward)
    return accumulated_regret


def Sim_MAB_Beliefs_Reward_Only(alphas, tau, MAB_Matrix, best_arm):
    number_of_arms = len(MAB_Matrix[0])
    number_of_rounds = len(MAB_Matrix)
    total_reward = 0
    accumulated_reward = []
    # Create Team
    team = create_team(alphas, tau, number_of_arms)
    # Loop through the rounds:
    round = 0
    for i in range(number_of_rounds):
        round += 1
        # Combine beliefs
        team_belief = generate_team_belief(team)
        # Convert Team beliefs to Team CP:
        team_choice_prob = softmax_belief_to_prob(team_belief, tau)
        # Generate which arm is chosen out of the probabilities
        choice = np.random.choice(list(range(number_of_arms)), p=team_choice_prob)
        # Play a round depending on the choice
        reward = MAB_Matrix[i][choice]
        total_reward += reward
        accumulated_reward.append(total_reward)
        # Update beliefs of all team members
        for member in team:
            member.update_beliefs(choice, reward)
    return accumulated_reward