import numpy as np
import itertools
import math
import matplotlib.pyplot as plt


def plot_results_constant_alphas(number_of_rounds, tau, acc_rewards, rate_of_best_rewards, acc_regret, alpha, num_mabs, method):
    """ Create plots of the accumulated reward, accumulated regret and
    rate of choosing the best reward against the round number for
    different values of tau """
    # Create a new figure and a set of subplots for the plots
    fig, axs = plt.subplots(nrows=2, ncols=2, figsize=(10, 8))
    if method == 0:
        fig.suptitle(f'Averaging Choice Probabilities')
    else:
        fig.suptitle(f'Averaging Beliefs')
    # Loop through each value of tau and create plots for each metric
    for i, tau in enumerate(tau):
        axs[0, 0].plot(range(number_of_rounds), acc_rewards[0][i], label=f'Tau: {round(tau, 2)}')
        axs[0, 0].set_xlabel('Round number')
        axs[0, 0].set_ylabel('Accumulated reward')
        axs[0, 0].set_title(f'Accumulated reward over time | Alpha={alpha}, {num_mabs} Simulations')
        axs[0, 0].legend()
        axs[0, 1].plot(range(number_of_rounds), rate_of_best_rewards[0][i], label=f'Tau: {round(tau, 2)}')
        axs[0, 1].set_xlabel('Round number')
        axs[0, 1].set_ylabel('Rate of best rewards')
        axs[0, 1].set_title(f'Rate of best rewards over time | Alpha={alpha}, {num_mabs} Simulations')
        axs[0, 1].legend()
        axs[1, 0].plot(range(number_of_rounds), acc_regret[0][i], label=f'Tau: {round(tau, 2)}')
        axs[1, 0].set_xlabel('Round number')
        axs[1, 0].set_ylabel('Accumulated regret')
        axs[1, 0].set_title(f'Accumulated regret over time | Alpha={alpha}, {num_mabs} Simulations')
        axs[1, 0].legend()
    fig.tight_layout()
    plt.draw()


def generate_MAB_Matrix(number_of_trials, probabilities_of_success, number_of_rounds):
    number_of_arms = len(number_of_trials)
    best_arm = np.argmax(probabilities_of_success)
    MAB_Matrix = np.random.binomial(n=number_of_trials,
                                    p=probabilities_of_success,
                                    size=[number_of_rounds, number_of_arms])
    return MAB_Matrix, best_arm


def diversity(float_list):
    sorted_list = sorted(float_list)
    differences = [abs(sorted_list[i] - sorted_list[i + 1]) for i in range(len(sorted_list) - 1)]
    max_difference = max(differences)
    if max_difference == 0:
        normalized_differences = [0] * len(differences)
    else:
        normalized_differences = [diff / max_difference for diff in differences]
    weighted_sum = sum([(1 - diff) * diff for diff in normalized_differences])
    if weighted_sum > 0:
        entropy = -math.log2(weighted_sum)
    else:
        entropy = float('1000')  # Assign a high value when weighted sum is zero or negative
    return entropy


def calculate_diversity(vector):
    unique_elements = len(set(vector))
    if unique_elements == 1:
        return 0
    elif unique_elements == 2:
        return 0.25
    elif unique_elements == 3:
        return 0.5
    elif unique_elements == 4:
        return 0.75
    elif unique_elements == 5:
        return 1


def generate_combinations(size, x, y, dt):
    # Calculate the number of steps based on the boundaries and step size
    num_steps = int((y - x) / dt) + 1
    # Create a list of all possible values for each entry in the vector
    value_range = [x + i * dt for i in range(num_steps)]
    # Generate all possible combinations using itertools.product
    combinations = list(itertools.combinations_with_replacement(value_range, size))
    # Convert each combination to a list
    combinations = [sorted(list(comb)) for comb in combinations]
    # Remove duplicate combinations
    unique_combinations = list(set(map(tuple, combinations)))
    # Sort combinations based on the smallest alpha value in each combination
    sorted_combinations = sorted(unique_combinations, key=lambda comb: min(comb))

    return sorted_combinations


def softmax_belief_to_prob(beliefs, tau):
    """ Implementation of the Softmax choice rule: Input beliefs and output choice probabilities """
    # choice_probabilities = []
    # # Calculate the denominator of the Softmax formula
    # denominator = sum(np.exp(i/(tau/len(beliefs))) for i in beliefs)
    # # Loop through each belief and calculate its corresponding choice probability
    # for belief in beliefs:
    #     choice_probabilities.append(round((np.exp(belief/(tau/len(beliefs)))/denominator), 10))
    # return choice_probabilities
    z = sum([math.exp(v/tau) for v in beliefs])
    probs = [math.exp(v/tau)/z for v in beliefs]
    return probs


def vertical_avg(lst):
    """Calculates the vertical average of a three-dimensional list. Given a three-dimensional
    list `lst` of shape (num_sub_lists, num_sub_sub_lists, n), where n represents the number
    of elements in each sublist, this function returns a new two-dimensional list where each
    element is the average of the corresponding vertical elements from all sub lists in `lst` """
    result = []
    # Calculate the number of sub-lists and sub-sub-lists in the input list
    num_sub_lists = len(lst)
    num_sub_sub_lists = len(lst[0])
    # Loop through each sub-sub-list in the input list
    for i in range(num_sub_sub_lists):
        sub_result = []
        # Loop through each element in the first sub-sub-list to get the length of the sub-sub-list
        for j in range(len(lst[0][0])):
            total = 0
            # Loop through each sub-list and add the corresponding element to the total
            for k in range(num_sub_lists):
                total += lst[k][i][j]
            # Calculate the average of the corresponding vertical elements and append to the sub-result list
            sub_result.append(total / num_sub_lists)
        # Append the sub-result list to the result list
        result.append(sub_result)
    return result


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
