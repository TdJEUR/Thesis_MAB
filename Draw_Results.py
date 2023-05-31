import matplotlib.pyplot as plt


def plot_results_constant_alphas(number_of_rounds, tau, acc_rewards, rate_of_best_rewards, acc_regret, alpha, num_sims):
    """ Create plots of the accumulated reward, accumulated regret and
    rate of choosing the best reward against the round number for
    different values of tau """
    # Create a new figure and a set of subplots for the plots
    fig, axs = plt.subplots(nrows=2, ncols=2, figsize=(10, 8))
    # Loop through each value of tau and create plots for each metric
    for i, tau in enumerate(tau):
        axs[0, 0].plot(range(number_of_rounds), acc_rewards[i], label=f'Tau: {round(tau, 2)}')
        axs[0, 0].set_xlabel('Round number')
        axs[0, 0].set_ylabel('Accumulated reward')
        axs[0, 0].set_title(f'Accumulated reward over time | Alpha={alpha}, {num_sims} Simulations')
        axs[0, 0].legend()
        axs[0, 1].plot(range(number_of_rounds), rate_of_best_rewards[i], label=f'Tau: {round(tau, 2)}')
        axs[0, 1].set_xlabel('Round number')
        axs[0, 1].set_ylabel('Rate of best rewards')
        axs[0, 1].set_title(f'Rate of best rewards over time | Alpha={alpha}, {num_sims} Simulations')
        axs[0, 1].legend()
        axs[1, 0].plot(range(number_of_rounds), acc_regret[i], label=f'Tau: {round(tau, 2)}')
        axs[1, 0].set_xlabel('Round number')
        axs[1, 0].set_ylabel('Accumulated regret')
        axs[1, 0].set_title(f'Accumulated regret over time | Alpha={alpha}, {num_sims} Simulations')
        axs[1, 0].legend()
    fig.tight_layout()
    plt.draw()


def plot_results_constant_tau(number_of_rounds, alphas, acc_rewards, rate_of_best_rewards, acc_regret, tau, num_sims, method):
    """ Create plots of the accumulated reward, accumulated regret and
    rate of choosing the best reward against the round number for
    different compositions of alphas """
    # Create a new figure and a set of subplots for the plots
    fig, axs = plt.subplots(nrows=2, ncols=2, figsize=(10, 8))
    if method == 0:
        fig.suptitle(f'Averaging Choice Probabilities')
    else:
        fig.suptitle(f'Averaging Beliefs')
    # Loop through each value of tau and create plots for each metric
    for i, alpha in enumerate(alphas):
        axs[0, 0].plot(range(number_of_rounds), acc_rewards[i], label=f'Alphas: {alphas[i]}')
        axs[0, 0].set_xlabel('Round number')
        axs[0, 0].set_ylabel('Accumulated reward')
        axs[0, 0].set_title(f'Accumulated reward over time | Tau={tau}, {num_sims} Simulations')
        axs[0, 0].legend()
        axs[0, 1].plot(range(number_of_rounds), rate_of_best_rewards[i], label=f'Alphas: {alphas[i]}')
        axs[0, 1].set_xlabel('Round number')
        axs[0, 1].set_ylabel('Rate of best rewards')
        axs[0, 1].set_title(f'Rate of best rewards over time | Tau={tau}, {num_sims} Simulations')
        axs[0, 1].legend()
        axs[1, 0].plot(range(number_of_rounds), acc_regret[i], label=f'Alphas: {alphas[i]}')
        axs[1, 0].set_xlabel('Round number')
        axs[1, 0].set_ylabel('Accumulated regret')
        axs[1, 0].set_title(f'Accumulated regret over time | Tau={tau}, {num_sims} Simulations')
        axs[1, 0].legend()
    fig.tight_layout()
    plt.draw()
