
def calculate_diversity(vectors):
    diversity_scores = []
    for vector in vectors:
        total_elements = len(vector)
        unique_elements = len(set(vector))
        if unique_elements == 1:
            diversity_scores.append(0)
        elif unique_elements == 2:
            diversity_scores.append(0.25)
        elif unique_elements == 3:
            diversity_scores.append(0.5)
        elif unique_elements == 4:
            diversity_scores.append(0.75)
        elif unique_elements == 5:
            diversity_scores.append(1)
    return diversity_scores


vectors = [[0.1, 0.2, 0.3, 0.4], [0.1, 0.1, 0.1, 0.1], [0, 0, 0, 1], [0, 0, 0, 0.1]]
diversity_scores = calculate_diversity(vectors)
print(diversity_scores)
