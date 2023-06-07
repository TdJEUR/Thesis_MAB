from Helpers import generate_MAB_Matrix
from Sim_MAB_CP import Sim_MAB_CP
import matplotlib.pyplot as plt
import numpy as np


# Number of Arms: N
no_arms = 3
probabilities_of_success = [0.5, 0.8, 0.3]
# Number of Rounds: T
number_of_rounds = 500

# Input Team Variables
# Exploration-Exploitation Temperature: Tau
# taus = np.linspace(0.01, 0.5, 2)
taus = [0.4, 0.5, 0.6]
taus = [round(tau, 4) for tau in taus]
# Composition of Team Experiential Learning Mechanisms:
alphas = [0.05, 0.1]
number_of_trails = [1]*no_arms


MAB_Matrix, best_arm = generate_MAB_Matrix(number_of_trails, probabilities_of_success, number_of_rounds)

res_reg = {}
res_rew = {}
res_robr = {}

for tau in taus:
    accumulated_reward, rate_of_best_arm, accumulated_regret = Sim_MAB_CP(alphas, tau, MAB_Matrix, best_arm)
    res_reg[tau] = accumulated_regret
    res_rew[tau] = accumulated_reward
    res_robr[tau] = rate_of_best_arm

# Plotting the results
for tau, output_list in res_reg.items():
    plt.plot(range(number_of_rounds), output_list, label=tau)

# Set plot labels and legend
plt.xlabel('Rounds')
plt.ylabel('Reg')
plt.legend()

# Show the plot
plt.show()
