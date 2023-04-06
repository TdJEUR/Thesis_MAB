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


def simulate_different_taus():
    # Run simulation with constant alpha and different values of tau
    acc_rewards = []
    rate_of_best_rewards = []
    acc_regret = []
    for i in np.linspace(0.01, 2, num=5):
        res = team_MAB(alphas, i, true_arm_rewards, number_of_rounds)
        acc_rewards.append(res[0])
        rate_of_best_rewards.append(res[1])
        acc_regret.append(res[2])
    # Plot results
    fig, axs = plt.subplots(nrows=2, ncols=2, figsize=(10, 8))
    # fig.subplots_adjust(hspace=0.4, wspace=0.3)
    axs[0, 0].plot(range(number_of_rounds), acc_rewards[0], label='Tau: 0.01')
    axs[0, 0].plot(range(number_of_rounds), acc_rewards[1], label='Tau: 0.5')
    axs[0, 0].plot(range(number_of_rounds), acc_rewards[2], label='Tau: 1.0')
    axs[0, 0].plot(range(number_of_rounds), acc_rewards[3], label='Tau: 1.5')
    axs[0, 0].plot(range(number_of_rounds), acc_rewards[4], label='Tau: 2.0')
    axs[0, 0].set_xlabel('Round number')
    axs[0, 0].set_ylabel('Accumulated reward')
    axs[0, 0].set_title('Accumulated reward over time')
    axs[0, 0].legend()
    axs[0, 1].plot(range(number_of_rounds), rate_of_best_rewards[0], label='Tau: 0.01')
    axs[0, 1].plot(range(number_of_rounds), rate_of_best_rewards[1], label='Tau: 0.5')
    axs[0, 1].plot(range(number_of_rounds), rate_of_best_rewards[2], label='Tau: 1.0')
    axs[0, 1].plot(range(number_of_rounds), rate_of_best_rewards[3], label='Tau: 1.5')
    axs[0, 1].plot(range(number_of_rounds), rate_of_best_rewards[4], label='Tau: 2.0')
    axs[0, 1].set_xlabel('Round number')
    axs[0, 1].set_ylabel('Rate of best rewards')
    axs[0, 1].set_title('Rate of best rewards over time')
    axs[0, 1].legend()
    axs[1, 0].plot(range(number_of_rounds), acc_regret[0], label='Tau: 0.01')
    axs[1, 0].plot(range(number_of_rounds), acc_regret[1], label='Tau: 0.5')
    axs[1, 0].plot(range(number_of_rounds), acc_regret[2], label='Tau: 1.0')
    axs[1, 0].plot(range(number_of_rounds), acc_regret[3], label='Tau: 1.5')
    axs[1, 0].plot(range(number_of_rounds), acc_regret[4], label='Tau: 2.0')
    axs[1, 0].set_xlabel('Round number')
    axs[1, 0].set_ylabel('Accumulated regret')
    axs[1, 0].set_title('Accumulated regret over time')
    axs[1, 0].legend()
    fig.tight_layout()
    plt.show()


simulate_different_taus()