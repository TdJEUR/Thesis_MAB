from Team_Based_MAB import create_team


team = create_team([0.5, 0.5, 0.5], 3)


def generate_team_choice_prob(probabilities):
    combined_prob = [1.0] * len(probabilities[0])
    for p in probabilities:
        for i, option_prob in enumerate(p):
            combined_prob[i] *= option_prob
    total_prob = sum(combined_prob)
    return [p / total_prob for p in combined_prob]


comb = generate_team_choice_prob([[0.2, 0.5, 0.3], [0.1, 0.8, 0.1]])
print(comb)
