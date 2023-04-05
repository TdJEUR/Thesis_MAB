from Team_Based_MAB import team_MAB
# import matplotlib.pyplot as plt


# ----------------------------------------------------------------------------------------------------------------------
# Team Based MAB
# ----------------------------------------------------------------------------------------------------------------------

# MAB variables
true_arm_rewards = [0.1, 0.8, 0.1]
# number_of_arms = len(true_arm_rewards)
number_of_rounds = 50
# true_arm_stds = []

# Team variables
alphas = [0.3, 0.4, 0.5, 0.6]
tau = 0.5

if __name__ == '__main__':
    team_MAB(alphas, tau, true_arm_rewards, number_of_rounds)
