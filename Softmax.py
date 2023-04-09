import numpy as np


def softmax_belief_to_prob(beliefs, tau):
    """ Implementation of the Softmax choice rule: Input the
    player's current beliefs and output their choice probabilities """
    choice_probabilities = []
    # Calculate the denominator of the Softmax formula
    denominator = sum(np.exp(i/(tau/len(beliefs))) for i in beliefs)
    # Loop through each belief and calculate its corresponding choice probability
    for belief in beliefs:
        choice_probabilities.append(round((np.exp(belief/(tau/len(beliefs)))/denominator), 10))
    return choice_probabilities
