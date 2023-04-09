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

    # combined_prob = [1.0] * len(probabilities[0])
    # for p in probabilities:
    #     for i, option_prob in enumerate(p):
    #         combined_prob[i] *= option_prob
    # total_prob = sum(combined_prob)
    # return [p / total_prob for p in combined_prob]
