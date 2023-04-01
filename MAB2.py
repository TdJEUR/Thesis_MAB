import numpy as np


true_arm_rewards = [0.5, 0.9, 0.5]
# true_arm_stds = []
beliefs = [0.7, 0.8, 0.7]
number_of_rounds = 10
tau = 0.1
pulls_per_arm = {i: 0 for i in range(1, len(true_arm_rewards)+1)}


def beliefs_to_choice_probabilities(beliefs):
    # SOFTMAX CHOICE RULE:
    # Input the player's current beliefs and output their choice probabilities
    choice_probabilities = []
    denominator = sum(np.exp(i/(tau/len(beliefs))) for i in beliefs)
    for i in beliefs:
        choice_probabilities.append(round((np.exp(i/(tau/len(beliefs)))/denominator), 10))
    print(f"Beliefs: {beliefs}\nTau: {tau}\nChoice Probabilities: {choice_probabilities}")
    return choice_probabilities


def choose_from_beliefs(choice_probabilities):
    # SELECT ARM:
    # Choose which arm to play according to the player's choice probabilities
    choice = np.random.choice(list(range(1, len(true_arm_rewards)+1)), p=choice_probabilities)
    # print(f"Picked arm: {choice}")
    return choice


def update_beliefs(beliefs, reward):
    pass


if __name__ == '__main__':
    choice_probs = beliefs_to_choice_probabilities(beliefs)
    choices = []
    for i in range(number_of_rounds):
        choices.append(choose_from_beliefs(choice_probs))
    choices = [(elem, choices.count(elem)) for elem in set(choices)]
    print(f"Distribution of picked arms: {dict(choices)}")
