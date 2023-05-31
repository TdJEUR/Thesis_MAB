from Beliefs_Average import simulate_MAB_Avg_B
from CP_Average import simulate_MAB_Avg_CP
from Generate_MAB_Matrix import generate_MAB_Matrix
from Helpers import generate_combinations
import matplotlib.pyplot as plt
import pandas as pd


# Input MAB Variables
number_of_trials = [1, 1, 1, 1, 1, 1, 1]
number_of_rounds = 100
probabilities_of_success = [0.5, 0.8, 0.3, 0.5, 0.1, 0.1, 0.1]
no_arms = len(number_of_trials)

# Input Team Variables
tau = 0.1
# Option to generate all possible team combinations:
# alphas = generate_combinations(size=3, x=0.01, y=0.15, dt=0.05)
# Option to input teams manually:
alphas = [[0.1, 0.1, 0.1, 0.1],
          [0.1, 0.08, 0.05, 0.05],
          [0.15, 0.08, 0.05, 0.05],
          [0.15, 0.1, 0.08, 0.05]]

# Input number of simulations per MAB
num_sims = 100


# Generate MAB Matrix:
MAB_Matrix = generate_MAB_Matrix(number_of_trials=number_of_trials,
                                 probabilities_of_success=probabilities_of_success,
                                 number_of_rounds=number_of_rounds)

# Generate data for Averaging of Beliefs
df_rew_B, df_reg_B, df_ROBR_B = simulate_MAB_Avg_B(MAB_Matrix=MAB_Matrix,
                                                   probabilities_of_success=probabilities_of_success,
                                                   tau=tau,
                                                   alphas=alphas,
                                                   num_sims=num_sims)

# Generate data for Averaging of Choice Probabilities
df_rew_CP, df_reg_CP, df_ROBR_CP = simulate_MAB_Avg_CP(MAB_Matrix=MAB_Matrix,
                                                       probabilities_of_success=probabilities_of_success,
                                                       tau=tau,
                                                       alphas=alphas,
                                                       num_sims=num_sims)

# Combine data into single Dataframe
df_reg = pd.merge(df_reg_B, df_reg_CP, on="Alphas")
df_rew = pd.merge(df_rew_B, df_rew_CP, on="Alphas")
df_ROBR = pd.merge(df_ROBR_B, df_ROBR_CP, on="Alphas")
df_final = pd.merge(df_reg, df_rew, on="Alphas")
df_final = pd.merge(df_final, df_ROBR, on="Alphas")

# Show Dataframe
print(df_final.to_string())

# Show Graphs
# plt.show()
