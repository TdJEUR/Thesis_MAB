import numpy as np


def generate_MAB_Matrix(number_of_trials, probabilities_of_success, number_of_rounds):
    number_of_arms = len(number_of_trials)
    MAB_Matrix = np.random.binomial(n=number_of_trials,
                                    p=probabilities_of_success,
                                    size=[number_of_rounds, number_of_arms])
    return MAB_Matrix
