from Create_Team_and_MAB import create_team, create_MAB
from Team_Choice_Probability import generate_team_choice_prob
import numpy as np


def team_MAB(alphas, tau, true_arm_rewards, true_arm_stds, number_of_rounds):
    """ Simulate a team consisting of members defined by alphas (list containing alpha
    for each member) playing a MAB with rewards corresponding to true_arm_rewards (list
    containing reward of each arm) for a certain number_of_rounds (integer). The exploration-
    exploitation strategy the team takes is defined by tau (float) """
    number_of_arms = len(true_arm_rewards)
    # Create MAB model
    mab = create_MAB(true_arm_rewards, true_arm_stds)
    # Create Team model
    team = create_team(alphas, tau, number_of_arms)
    # Loop through the rounds:
    for i in range(1, number_of_rounds+1):
        # Generate individual choice probabilities
        for j, member in enumerate(team, 1):
            member.get_choice_probabilities()
        # Generate team choice probabilities
        team_choice_prob = generate_team_choice_prob(team)
        # Generate which arm is chosen out of the probabilities
        choice = np.random.choice(list(range(number_of_arms)), p=team_choice_prob)
        # Play a round depending on the choice
        reward = mab.play_round(choice)
        # Update beliefs of all team members
        for member in team:
            member.update_beliefs(choice, reward)
    # final_cd = mab.get_choices_distribution()
    # print(f"Distribution of picked arms: {final_cd}")
    return mab.accumulated_rewards, mab.rate_of_best_reward, mab.accumulated_regret
