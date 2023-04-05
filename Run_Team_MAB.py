from Team_Based_MAB import create_team, create_MAB, generate_team_choice_prob
import matplotlib.pyplot as plt


# ----------------------------------------------------------------------------------------------------------------------
# Team Based MAB
# ----------------------------------------------------------------------------------------------------------------------

# MAB variables
true_arm_rewards = [0.4, 0.8, 0.4]
number_of_arms = len(true_arm_rewards)
number_of_rounds = 100
# true_arm_stds = []

# Team variables
alphas = [0.5, 0.5, 0.5, 0.5, 0.5]
team_size = len(alphas)
tau = 10

if __name__ == '__main__':
    team = create_team(alphas, number_of_arms)
    for member in team:
        print(member.alpha, member.belief, member.choice_probabilities)
    mab = create_MAB(true_arm_rewards)
    generate_team_choice_prob(team)

