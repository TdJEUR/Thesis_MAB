import matplotlib.pyplot as plt


def plot_results(number_of_rounds, acc_rewards, rate_of_best_rewards, acc_regret):
    """ Create plots of the accumulated reward, accumulated regret and
    rate of choosing the best reward against the round number for
    different values of tau """
    fig, axs = plt.subplots(nrows=2, ncols=2, figsize=(10, 8))
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
