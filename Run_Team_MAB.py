from Team_Based_MAB import team_MAB
import numpy as np
import matplotlib.pyplot as plt


# ----------------------------------------------------------------------------------------------------------------------
# Team Based MAB
# ----------------------------------------------------------------------------------------------------------------------

# MAB variables
true_arm_rewards = [0.1, 0.8, 0.1]
# number_of_arms = len(true_arm_rewards)
number_of_rounds = 50
# true_arm_stds = []

# Team variables
alphas = [0.5, 0.5, 0.5, 0.5]
# tau = 0.9

if __name__ == '__main__':
    acc_rewards = []
    for i in np.linspace(start=0.01, stop=3, num=5):
        acc_rewards.append(team_MAB(alphas, i, true_arm_rewards, number_of_rounds))

    # set the x and y axis labels
    plt.xlabel('Round number')
    plt.ylabel('Reward gained')
    # set the title of the plot
    plt.title('Reward gained over time')
    # plot each y dataset
    for i in range(len(acc_rewards)):
        plt.plot(range(number_of_rounds), acc_rewards[i], label=f'Tau {i + 1}')
    # show the legend
    plt.legend()
    # display the plot
    plt.show()
