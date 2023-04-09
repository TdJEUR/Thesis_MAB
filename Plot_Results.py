import matplotlib.pyplot as plt


def plot_results(number_of_rounds, tau, acc_rewards, rate_of_best_rewards, acc_regret):
    """ Create plots of the accumulated reward, accumulated regret and
    rate of choosing the best reward against the round number for
    different values of tau """
    fig, axs = plt.subplots(nrows=2, ncols=2, figsize=(10, 8))
    for i, tau in enumerate(tau):
        axs[0, 0].plot(range(number_of_rounds), acc_rewards[i], label=f'Tau: {tau}')
        axs[0, 0].set_xlabel('Round number')
        axs[0, 0].set_ylabel('Accumulated reward')
        axs[0, 0].set_title('Accumulated reward over time')
        axs[0, 0].legend()
        axs[0, 1].plot(range(number_of_rounds), rate_of_best_rewards[i], label=f'Tau: {tau}')
        axs[0, 1].set_xlabel('Round number')
        axs[0, 1].set_ylabel('Rate of best rewards')
        axs[0, 1].set_title('Rate of best rewards over time')
        axs[0, 1].legend()
        axs[1, 0].plot(range(number_of_rounds), acc_regret[i], label=f'Tau: {tau}')
        axs[1, 0].set_xlabel('Round number')
        axs[1, 0].set_ylabel('Accumulated regret')
        axs[1, 0].set_title('Accumulated regret over time')
        axs[1, 0].legend()
    fig.tight_layout()
    plt.show()
