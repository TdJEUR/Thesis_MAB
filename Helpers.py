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
