import numpy as np
import math


def softmax_belief_to_prob(beliefs, tau):
    """ Implementation of the Softmax choice rule: Input beliefs and output choice probabilities """
    if tau == 0:
        arr = np.array(beliefs)
        mask = np.isclose(arr, np.max(arr), atol=1e-8)
        max_indices = np.where(mask)[0]
        selected_index = np.random.choice(max_indices) if max_indices.size > 0 else None
        output = np.zeros_like(beliefs)
        if selected_index is not None:
            output[selected_index] = 1
        return output.tolist()
    z = sum([math.exp(v/tau) for v in beliefs])
    probs = [math.exp(v/tau)/z for v in beliefs]
    return probs


# Example usage
my_list = [1.5, 2.7, 4.1, 2.7, 3.9, 4.1, 2.2]
index = softmax_belief_to_prob(my_list, 0)
print("Index of the largest float:", index)