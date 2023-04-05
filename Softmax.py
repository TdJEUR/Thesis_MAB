import numpy as np


def softmax_belief_to_prob(beliefs, tau):
    # SOFTMAX CHOICE RULE:
    # Input the player's current beliefs and output their choice probabilities
    choice_probabilities = []
    denominator = sum(np.exp(i/(tau/len(beliefs))) for i in beliefs)
    for belief in beliefs:
        choice_probabilities.append(round((np.exp(belief/(tau/len(beliefs)))/denominator), 10))
    print(f"Beliefs: {beliefs}\nChoice Probabilities: {choice_probabilities}")
    return choice_probabilities
