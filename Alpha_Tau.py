import numpy as np
from Helpers import diversity, generate_combinations, calculate_diversity, generate_MAB_Matrix, \
    plot_results_constant_alphas
import pandas as pd
from Multiple_Sims import Single_MAB_Multiple_Sims
import matplotlib.pyplot as plt


path = "C:/Users/tommo/Downloads/Thesis_MAB/Data"

# --------------------------------------------------------------------------------------------
# Input MAB Variables
# Number of Arms: N
no_arms = 10
# probabilities_of_success = [0.5, 0.8, 0.3, 0.5]
probabilities_of_success = 0
# Number of Rounds: T
number_of_rounds = 500
# Number of Different MAB Matrices:
num_mabs = 100
# Number of simulations per MAB Matrix:
num_sims = 1
# --------------------------------------------------------------------------------------------
# Input Team Variables
# Exploration-Exploitation Temperature: Tau
taus = np.linspace(0.01, 1, 6)
taus = [round(tau, 4) for tau in taus]
# Composition of Team Experiential Learning Mechanisms:
# Option to generate all possible combinations:
# alphas = generate_combinations(size=5, x=0.01, y=0.22, dt=0.05)
# Option to input teams manually:
# alphas = [[0.1, 0.1, 0.1, 0.1],
#           [0.1, 0.08, 0.05, 0.05],
#           [0.15, 0.08, 0.05, 0.05],
#           [0.15, 0.1, 0.08, 0.05]]
alphas = [[0.05, 0.05, 0.05, 0.05, 0.05]]
# --------------------------------------------------------------------------------------------

# Define helper variables
number_of_trials = [1] * no_arms
total_calc = len(alphas)*num_mabs
count = 0

# Round Alphas
alphas = [[round(alpha, 4) for alpha in sublist] for sublist in alphas]

# Initialise final results for df and graphs
mab_results_B = []
mab_results_CP = []
all_acc_rewards_cp = []
all_acc_rewards_b = []
all_rate_of_best_rewards_cp = []
all_rate_of_best_rewards_b = []
all_acc_regret_cp = []
all_acc_regret_b = []

# For each different MAB:
for mab in range(num_mabs):
    # Generate MAB:
    if probabilities_of_success == 0:
        MAB_Matrix, best_arm = generate_MAB_Matrix(number_of_trials, np.random.beta(2, 2, no_arms), number_of_rounds)
    else:
        MAB_Matrix, best_arm = generate_MAB_Matrix(number_of_trials, probabilities_of_success, number_of_rounds)
    # Create dfs for specific MAB:
    columns = ['Alpha', 'Avg_Alpha', 'Diversity']
    for tau in taus:
        columns.append(f'Tau={tau}')
    df_B = pd.DataFrame(columns=columns)
    df_CP = pd.DataFrame(columns=columns)
    # Store lists containing
    MAB_acc_rewards_cp = []
    MAB_acc_rewards_b = []
    MAB_rate_of_best_rewards_cp = []
    MAB_rate_of_best_rewards_b = []
    MAB_acc_regret_cp = []
    MAB_acc_regret_b = []
    # For each alpha:
    for k, alpha in enumerate(alphas):
        # Create a row in the dfs:
        alpha_row_B = [f'{alpha}', sum(alpha)/len(alpha), calculate_diversity(alpha)]
        alpha_row_CP = [f'{alpha}', sum(alpha)/len(alpha), calculate_diversity(alpha)]
        # Calculate regret for every value of tau
        avg_acc_regret_B, avg_acc_rewards_B, avg_rate_of_best_rewards_B,\
        avg_acc_regret_CP, avg_acc_rewards_CP, avg_rate_of_best_rewards_CP = \
            Single_MAB_Multiple_Sims(MAB_Matrix, best_arm, num_sims, taus, alpha)
        # Add list of results for each alpha: alpha_1: [[regrets (tau_1)], [regrets (tau_2)], ... ] ,
        #                                     alpha_2: [[regrets (tau_1)], [regrets (tau_2)], ... ] , ...
        # to the list of results for the MAB
        MAB_acc_rewards_cp.append(avg_acc_rewards_CP)
        MAB_acc_rewards_b.append(avg_acc_rewards_B)
        MAB_rate_of_best_rewards_cp.append(avg_rate_of_best_rewards_CP)
        MAB_rate_of_best_rewards_b.append(avg_rate_of_best_rewards_B)
        MAB_acc_regret_cp.append(avg_acc_regret_CP)
        MAB_acc_regret_b.append(avg_acc_regret_B)
        # Sort regret values into the row for the df:
        for j, tau in enumerate(taus):
            regret_B = avg_acc_regret_B[j][-1]
            regret_CP = avg_acc_regret_CP[j][-1]
            alpha_row_B.append(regret_B)
            alpha_row_CP.append(regret_CP)
        # Add row to dfs:
        df_B.loc[len(df_B)] = alpha_row_B
        df_CP.loc[len(df_CP)] = alpha_row_CP
        count += 1
        print(f'Finished all taus for alpha {k+1}/{len(alphas)} | '
              f'MAB {mab+1}/{num_mabs} | '
              f'{count*100/total_calc}%')
    # Add all results for all alphas to the total results containing blocks for all MABs:
    all_acc_rewards_cp.append(MAB_acc_rewards_cp)
    all_acc_rewards_b.append(MAB_acc_rewards_b)
    all_rate_of_best_rewards_cp.append(MAB_rate_of_best_rewards_cp)
    all_rate_of_best_rewards_b.append(MAB_rate_of_best_rewards_b)
    all_acc_regret_cp.append(MAB_acc_regret_cp)
    all_acc_regret_b.append(MAB_acc_regret_b)
    # Add all results for each MAB to total results:
    df_B = df_B.sort_values('Diversity')
    df_CP = df_CP.sort_values('Diversity')
    mab_results_B.append(df_B)
    mab_results_CP.append(df_CP)
# Average all MABs into single df
concatenated_df_B = pd.concat(mab_results_B)
concatenated_df_CP = pd.concat(mab_results_CP)
df_B_final = concatenated_df_B.groupby('Alpha').mean()
df_CP_final = concatenated_df_CP.groupby('Alpha').mean()
df_B_final = df_B_final.sort_values('Diversity').reset_index()
df_CP_final = df_CP_final.sort_values('Diversity').reset_index()
# Average all MABs to graph
avg_acc_rewards_cp = np.mean(np.array(all_acc_rewards_cp), axis=0)
avg_acc_rewards_b = np.mean(np.array(all_acc_rewards_b), axis=0)
avg_rate_of_best_rewards_cp = np.mean(np.array(all_rate_of_best_rewards_cp), axis=0)
avg_rate_of_best_rewards_b = np.mean(np.array(all_rate_of_best_rewards_b), axis=0)
avg_acc_regret_cp = np.mean(np.array(all_acc_regret_cp), axis=0)
avg_acc_regret_b = np.mean(np.array(all_acc_regret_b), axis=0)

# Plot results (Single team)
if len(alphas) == 1:
    plot_results_constant_alphas(number_of_rounds, taus, avg_acc_rewards_cp, avg_rate_of_best_rewards_cp,
                                 avg_acc_regret_cp, alphas, num_sims, 0)
    plot_results_constant_alphas(number_of_rounds, taus, avg_acc_rewards_b, avg_rate_of_best_rewards_b,
                                 avg_acc_regret_b, alphas, num_sims, 1)

# df_B_final.to_excel(f'C:/Users/tommo/Downloads/Thesis_MAB/Data/df_B_0.5_{num_mabs}mabs_{num_sims}sims_x.xlsx', index=False, engine='openpyxl')
# df_CP_final.to_excel(f'C:/Users/tommo/Downloads/Thesis_MAB/Data/df_CP_0.5_{num_mabs}mabs_{num_sims}sims_x.xlsx', index=False, engine='openpyxl')

plt.show()

# Perform Statistical Analysis
stat_df = pd.DataFrame

