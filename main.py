from Run_Various_MABs import Sim_Matrix_X_Times
from Helpers import generate_combinations
from Statistical_Analysis import statistical_analysis
from Draw_Results import plot_results_constant_tau
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np


# --------------------------------------------------------------------------------------------
# Input MAB Variables
# Number of Arms: N
no_arms = 5
# Number of Rounds: T
number_of_rounds = 100
# Number of Different MAB Matrices:
num_mabs = 5
# Number of simulations per MAB Matrix:
num_sims = 1
# --------------------------------------------------------------------------------------------
# Input Team Variables
# Exploration-Exploitation Temperature: Tau
tau = 0.1
# Composition of Team Experiential Learning Mechanisms:
# Option to generate all possible combinations:
alphas = generate_combinations(size=3, x=0.01, y=0.2, dt=0.05)
# Option to input teams manually:
# alphas = [[0.1, 0.1, 0.1, 0.1],
#           [0.1, 0.08, 0.05, 0.05],
#           [0.15, 0.08, 0.05, 0.05],
#           [0.15, 0.1, 0.08, 0.05]]
# --------------------------------------------------------------------------------------------

# Round Alphas
alphas = [[round(alpha, 4) for alpha in sublist] for sublist in alphas]

# Initialise list of dataframes for data on each MAB Matrix:
dataframes = []

# Generate various MAB Matrices (Different Arms) from same distribution of possible arms:
for i in range(num_mabs):
    # Initialise result dfs:
    df_rew_B = pd.DataFrame({'Alphas': [], 'Acc_Reward(Beliefs)': []})
    df_rew_CP = pd.DataFrame({'Alphas': [], 'Acc_Reward(CP)': []})
    df_reg_B = pd.DataFrame({'Alphas': [], 'Acc_Reg(Beliefs)': []})
    df_reg_CP = pd.DataFrame({'Alphas': [], 'Acc_Reg(CP)': []})
    df_ROBR_B = pd.DataFrame({'Alphas': [], 'ROBR(Beliefs)': []})
    df_ROBR_CP = pd.DataFrame({'Alphas': [], 'ROBR(CP)': []})

    # Generate Single MAB Matrix and Simulate teams on this MAB Matrix
    avg_acc_regret_B, avg_acc_rewards_B, avg_rate_of_best_rewards_B, avg_acc_regret_CP, avg_acc_rewards_CP, \
        avg_rate_of_best_rewards_CP = Sim_Matrix_X_Times(no_arms,
                                                         number_of_rounds,
                                                         num_sims,
                                                         tau,
                                                         alphas)

    # Enter Data from Single MAB Matrix into dfs
    for j, alpha in enumerate(alphas):
        df_reg_B.loc[len(df_reg_B)] = [str(alpha), avg_acc_regret_B[j][-1]]
        df_rew_B.loc[len(df_rew_B)] = [str(alpha), avg_acc_rewards_B[j][-1]]
        df_ROBR_B.loc[len(df_ROBR_B)] = [str(alpha), avg_rate_of_best_rewards_B[j][-1]]
        df_reg_CP.loc[len(df_reg_CP)] = [str(alpha), avg_acc_regret_CP[j][-1]]
        df_rew_CP.loc[len(df_rew_CP)] = [str(alpha), avg_acc_rewards_CP[j][-1]]
        df_ROBR_CP.loc[len(df_ROBR_CP)] = [str(alpha), avg_rate_of_best_rewards_CP[j][-1]]

    # Merge into single df reflecting the results of the MAB Matrix:
    df_reg = pd.merge(df_reg_B, df_reg_CP, on="Alphas")
    df_rew = pd.merge(df_rew_B, df_rew_CP, on="Alphas")
    df_ROBR = pd.merge(df_ROBR_B, df_ROBR_CP, on="Alphas")
    df_merged = pd.merge(df_reg, df_rew, on="Alphas")
    df_merged = pd.merge(df_merged, df_ROBR, on="Alphas")

    # Add data from Single MAB Matrix to list of dfs:
    dataframes.append(df_merged)
    print(f"Finished Simulating MAB Matrix: {i + 1}/{num_mabs}")
    print(df_merged.to_string())

# Generate df with averages of all MAB Matrix dfs
# Concatenate DataFrames
concatenated_df = pd.concat(dataframes)
# Group by 'team' column and calculate column averages
df_final = concatenated_df.groupby('Alphas').mean().reset_index()

# Perform Statistical Analysis
statistical_analysis(df_final)

# Show Graphs
# plt.show()
