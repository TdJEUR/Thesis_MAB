
def generate_team_choice_prob(team):
    """" Combine the individual choice probabilities of the team members
    in team to a single team choice probability """
    probabilities = [member.choice_probabilities for member in team]

    # combined_prob = [1.0] * len(probabilities[0])
    # for p in probabilities:
    #     for i, option_prob in enumerate(p):
    #         combined_prob[i] *= option_prob
    # total_prob = sum(combined_prob)
    # return [p / total_prob for p in combined_prob]

    weights = [1.0 / len(probabilities)] * len(probabilities)
    assert len(probabilities) == len(weights)
    combined_prob = [0.0] * len(probabilities[0])
    for i in range(len(combined_prob)):
        for j in range(len(probabilities)):
            combined_prob[i] += weights[j] * probabilities[j][i]
    total_prob = sum(combined_prob)
    return [p / total_prob for p in combined_prob]
