import numpy as np
from scipy.stats import ttest_ind, mannwhitneyu, skew
from Helpers import generate_combinations, calculate_diversity, generate_MAB_Matrix, calc_skew
import pandas as pd
from Multiple_Sims import Single_MAB_Multiple_Sims, Single_MAB_Multiple_Sims_Reward_Only
import time

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
num_mabs = 50
# Number of simulations per MAB Matrix:
num_sims = 1
# --------------------------------------------------------------------------------------------
# Input Team Variables
# Exploration-Exploitation Temperature: Tau
taus = np.linspace(0, 0.05, 10)
taus = [round(tau, 5) for tau in taus]
# Composition of Team Experiential Learning Mechanisms - Generate all possible combinations:
alphas = generate_combinations(size=5, x=0.01, y=1.0, dt=0.45)
# --------------------------------------------------------------------------------------------

# Define helper variables
number_of_trials = [1] * no_arms
total_calc = len(alphas)*num_mabs
total_calc_left = total_calc
count = 0

# Round Alphas
alphas = [[round(alpha, 4) for alpha in sublist] for sublist in alphas]

# Initialise final results for df and graphs
mab_results_B = []
mab_results_CP = []
all_acc_rewards_cp = []
all_acc_rewards_b = []

# For each different MAB:
for mab in range(num_mabs):
    # Generate MAB:
    if probabilities_of_success == 0:
        MAB_Matrix, best_arm = generate_MAB_Matrix(number_of_trials, np.random.beta(2, 2, no_arms), number_of_rounds)
    else:
        MAB_Matrix, best_arm = generate_MAB_Matrix(number_of_trials, probabilities_of_success, number_of_rounds)
    # Create dfs for specific MAB:
    columns = ['Alpha', 'Avg_Alpha', 'STD', 'Skew']
    for tau in taus:
        columns.append(f'Tau={tau}')
    df_B = pd.DataFrame(columns=columns)
    df_CP = pd.DataFrame(columns=columns)
    # For each alpha:
    for k, alpha in enumerate(alphas):
        start_time = time.time()
        # Create a row in the dfs:
        alpha_row_B = [f'{alpha}', sum(alpha)/len(alpha), np.std(alpha), calc_skew(alpha)]
        alpha_row_CP = [f'{alpha}', sum(alpha)/len(alpha), np.std(alpha), calc_skew(alpha)]
        # Calculate regret for every value of tau
        avg_acc_rewards_B, avg_acc_rewards_CP = Single_MAB_Multiple_Sims_Reward_Only(MAB_Matrix, best_arm, num_sims, taus, alpha)
        # Add list of results for each alpha: alpha_1: [[regrets (tau_1)], [regrets (tau_2)], ... ] ,
        #                                     alpha_2: [[regrets (tau_1)], [regrets (tau_2)], ... ] , ...
        # to the list of results for the MAB
        # Sort regret values into the row for the df:
        for j, tau in enumerate(taus):
            reward_B = avg_acc_rewards_B[j][-1]
            reward_CP = avg_acc_rewards_CP[j][-1]
            alpha_row_B.append(reward_B)
            alpha_row_CP.append(reward_CP)
        # Add row to dfs:
        df_B.loc[len(df_B)] = alpha_row_B
        df_CP.loc[len(df_CP)] = alpha_row_CP
        count += 1
        total_calc_left -= 1
        iteration_duration = time.time() - start_time
        print(f'Finished all taus for alpha {k+1}/{len(alphas)} | '
              f'MAB {mab+1}/{num_mabs} | '
              f'{count*100/total_calc}% | {(total_calc_left*iteration_duration)/60/60} hours')
    # Add all results for each MAB to total results:
    mab_results_B.append(df_B)
    mab_results_CP.append(df_CP)
# Average all MABs into single df
concatenated_df_B = pd.concat(mab_results_B)
concatenated_df_CP = pd.concat(mab_results_CP)
df_B_final = concatenated_df_B.groupby('Alpha').mean()
df_CP_final = concatenated_df_CP.groupby('Alpha').mean()
res = []
for tau in taus:
    res.append(f'Tau={tau}')
print(df_B_final.to_string())
# averaged_df = df_B_final.groupby('Avg_Alpha')[res].mean().reset_index()
# merged_df = pd.merge(df_B_final[['Alpha', 'Avg_Alpha']], averaged_df, on='Avg_Alpha')

# Save results
# merged_df.to_excel(f'C:/Users/tommo/Downloads/Thesis_MAB/Data/Merged_{num_mabs}mabs_{num_sims}sims_10arms_test3.xlsx', engine='openpyxl')
df_B_final.to_excel(f'C:/Users/tommo/Downloads/Thesis_MAB/Data/df_B_{num_mabs}mabs_{num_sims}sims_10arms_largeA2.xlsx', engine='openpyxl')
df_CP_final.to_excel(f'C:/Users/tommo/Downloads/Thesis_MAB/Data/df_CP_{num_mabs}mabs_{num_sims}sims_10arms_largeA2.xlsx', engine='openpyxl')

# Perform Statistical Analysis
# for tau in taus:
#     print('Tau:', tau)
#     c1 = df_CP_final[f'Tau={tau}']
#     c2 = df_B_final[f'Tau={tau}']
#     t_statistic, p_value = ttest_ind(c1, c2)
#     print(f"T-Statistic: {t_statistic}, p-value: {p_value}")
#     if p_value < 0.05:
#         print("T-Statistic: The columns are statistically different.")
#     else:
#         print("T-Statistic: The columns are not statistically different.")
#     statistic, p_value = mannwhitneyu(c1, c2)
#     print(f"Mann-Whitney U statistic: {statistic}, p-value: {p_value}")
#     if p_value < 0.05:
#         print("Mann-Whitney U statistic: The columns are statistically different.\n")
#     else:
#         print("Mann-Whitney U statistic: The columns are not statistically different.\n")
