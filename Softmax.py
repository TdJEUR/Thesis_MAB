import numpy as np


# def softmax_belief_to_prob(beliefs, tau):
#     # SOFTMAX CHOICE RULE:
#     # Input the player's current beliefs and output their choice probabilities
#     choice_probabilities = []
#     denominator = sum(np.exp(i/(tau/len(beliefs))) for i in beliefs)
#     for belief in beliefs:
#         choice_probabilities.append(round((np.exp(belief/(tau/len(beliefs)))/denominator), 10))
#     return choice_probabilities

# def softmax_belief_to_prob(beliefs, tau):
#     # SOFTMAX CHOICE RULE:
#     # Input the player's current beliefs and output their choice probabilities
#     choice_probabilities = []
#     denominator = sum(np.exp(i/(tau/len(beliefs))) for i in beliefs)
#     print("Denominator:", denominator)
#     for belief in beliefs:
#         exp_term = np.exp(belief/(tau/len(beliefs)))
#         print("Exponent term:", exp_term)
#         choice_probabilities.append(round((exp_term/denominator), 10))
#     print("Choice probabilities:", choice_probabilities)
#     return choice_probabilities

def softmax_belief_to_prob(beliefs, tau):
    # SOFTMAX CHOICE RULE:
    # Input the player's current beliefs and output their choice probabilities
    max_belief = max(beliefs)
    exp_beliefs = [np.exp(belief/(tau/len(beliefs))-max_belief) for belief in beliefs]
    print(f"exp_beliefs: {exp_beliefs}")
    denominator = sum(exp_beliefs)
    print(f"denominator: {denominator}")
    choice_probabilities = [round(exp_belief/denominator, 10) for exp_belief in exp_beliefs]
    return choice_probabilities
