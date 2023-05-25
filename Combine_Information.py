

def generate_team_choice_prob(team):
    """" Combine the individual choice probabilities of the team members
    in team to a single team choice probability """
    # Extract the individual choice probabilities of each team member
    probabilities = [member.choice_probabilities for member in team]
    # Set weights for each team member's choice probabilities
    weights = [1.0 / len(probabilities)] * len(probabilities)
    # Initialize an empty list to store the combined probability
    combined_prob = [0.0] * len(probabilities[0])
    # Loop through each index of the combined probability list
    for i in range(len(combined_prob)):
        # Loop through each team member's probability list
        for j in range(len(probabilities)):
            # Combine each team member's probability using the specified weights
            combined_prob[i] += weights[j] * probabilities[j][i]
    # Calculate the total probability by summing the combined probabilities
    total_prob = sum(combined_prob)
    # Normalize the combined probabilities
    return [p / total_prob for p in combined_prob]


def generate_team_belief(team):
    """" Combine the individual beliefs of the team members
    in team to a single team belief """
    # Extract the individual beliefs of each team member
    belief = [member.belief for member in team]
    # Set weights for each team member's belief
    weights = [1.0 / len(belief)] * len(belief)
    # Initialize an empty list to store the combined belief
    combined_belief = [0.0] * len(belief[0])
    # Loop through each index of the combined belief list
    for i in range(len(combined_belief)):
        # Loop through each team member's belief list
        for j in range(len(belief)):
            # Combine each team member's beliefs using the specified weights
            combined_belief[i] += weights[j] * belief[j][i]
    # Calculate the total belief by summing the combined beliefs
    total_belief = sum(combined_belief)
    # Normalize the combined belief
    return [belief / total_belief for belief in combined_belief]

#
# def plurality_voting(team):
#     # Extract the individual choice probabilities of each team member
#     probabilities = [member.choice_probabilities for member in team]
#     res = [0]*len(probabilities[0])
#     scores = [0]*len(probabilities[0])
#     for member in probabilities:
#         max_index = member.index(max(member))
#         scores[max_index] += 1
#     most_votes_index = scores.index(max(scores))
#     res[most_votes_index] += 1
#     return res
#
#
# def two_stage_voting(team):
#     # Extract the individual choice probabilities of each team member
#     pass
#
#
# def rotating_dictator(team):
#     # Extract the individual choice probabilities of each team member
#     probabilities = [member.choice_probabilities for member in team]
#     res = [0] * len(probabilities[0])
#     chosen_member = np.random.choice(probabilities)
#     most_votes_index = chosen_member.index(max(chosen_member))
#     res[most_votes_index] += 1
#     return res
