from Beliefs_Average import play_Multiple_MABs_Avg_B
from CP_Average import play_Multiple_MABs_Avg_CP
import matplotlib.pyplot as plt


# Input MAB Variables
number_of_trials = [1, 1, 1, 1]
number_of_rounds = 200
probabilities_of_success = [0.5, 0.8, 0.3, 0.5]
no_arms = len(number_of_trials)

# Input Team Variables
tau = 0.1
alphas = [[0.05, 0.05, 0.05, 0.05],
          [0.05, 0.05, 0.05, 0.1],
          [0.05, 0.05, 0.1, 0.1],
          [0.05, 0.1, 0.1, 0.1],
          [0.1, 0.1, 0.1, 0.1]]

# Input number of MABs tested
num_MABs = 5

# Input number of simulations per MAB
num_sims = 150


play_Multiple_MABs_Avg_B(number_of_trials=number_of_trials,
                         number_of_rounds=number_of_rounds,
                         num_MABs=num_MABs,
                         probabilities_of_success=probabilities_of_success,
                         tau=tau,
                         alphas=alphas,
                         num_sims=num_sims)

print('Finished Beliefs')

play_Multiple_MABs_Avg_CP(number_of_trials=number_of_trials,
                          number_of_rounds=number_of_rounds,
                          num_MABs=num_MABs,
                          probabilities_of_success=probabilities_of_success,
                          tau=tau,
                          alphas=alphas,
                          num_sims=num_sims)

print('Finished CPs')
plt.show()