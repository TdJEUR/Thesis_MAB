import numpy as np
import timeit
from Helpers import diversity, generate_combinations, calculate_diversity
import pandas as pd


path = "C:/Users/tommo/Downloads/Thesis_MAB/Data"
path2 = "C:/Users/tommo/PycharmProjects/Thesis_MAB"
# --------------------------------------------------------------------------------------------
# Input MAB Variables
# Number of Arms: N
from Run_Various_MABs import Sim_Matrix_X_Times

no_arms = 5
# Number of Rounds: T
number_of_rounds = 100
# Number of Different MAB Matrices:
num_mabs = 50
# Number of simulations per MAB Matrix:
num_sims = 50
# --------------------------------------------------------------------------------------------
# Input Team Variables
# Exploration-Exploitation Temperature: Tau
taus = np.linspace(0.01, 0.5, 20)
taus = [round(tau, 4) for tau in taus]
# Composition of Team Experiential Learning Mechanisms:
# Option to generate all possible combinations:
alphas = generate_combinations(size=5, x=0.01, y=0.22, dt=0.05)
# Option to input teams manually:
# alphas = [[0.1, 0.1, 0.1, 0.1],
#           [0.1, 0.08, 0.05, 0.05],
#           [0.15, 0.08, 0.05, 0.05],
#           [0.15, 0.1, 0.08, 0.05]]
# --------------------------------------------------------------------------------------------


# Round Alphas
alphas = [[round(alpha, 4) for alpha in sublist] for sublist in alphas]

diversities_dict = {}
for alpha in alphas:
    diversities_dict[f'{alpha}'] = calculate_diversity(alpha)

mab_results_B = []
mab_results_CP = []

for mab in range(num_mabs):

    df_B = pd.DataFrame(list(diversities_dict.items()), columns=['Alpha', 'Diversity'])
    df_CP = pd.DataFrame(list(diversities_dict.items()), columns=['Alpha', 'Diversity'])

    # For each tau:
    for k, tau in enumerate(taus):
        tau_row_B = []
        tau_row_CP = []
        avg_acc_regret_B, _, _, avg_acc_regret_CP,_ ,_ = Sim_Matrix_X_Times(no_arms,
                                                                            number_of_rounds,
                                                                            num_sims,
                                                                            tau,
                                                                            alphas)
        for j, alpha in enumerate(alphas):
            diversity = diversities_dict[str(alpha)]
            regret_B = avg_acc_regret_B[j][-1]
            regret_CP = avg_acc_regret_CP[j][-1]
            tau_row_B.append(regret_B)
            tau_row_CP.append(regret_CP)
        df_B[f'Tau={tau}'] = tau_row_B
        df_CP[f'Tau={tau}'] = tau_row_CP
        print(f'Finished all alphas for tau {k+1}/{len(taus)} | '
              f'MAB {mab+1}/{num_mabs} | '
              f'{(100*(mab)/(num_mabs))+(100*(mab)/(num_mabs)*((k)/(len(taus))))}%')

    df_B = df_B.sort_values('Diversity')
    df_CP = df_CP.sort_values('Diversity')

    mab_results_B.append(df_B)
    mab_results_CP.append(df_CP)

# Concatenate DataFrames
concatenated_df_B = pd.concat(mab_results_B)
concatenated_df_CP = pd.concat(mab_results_CP)
print(concatenated_df_CP)
# Group by 'team' column and calculate column averages
df_B_final = concatenated_df_B.groupby('Alpha').mean()
df_CP_final = concatenated_df_CP.groupby('Alpha').mean()

df_B_final = df_B_final.sort_values('Diversity').reset_index()
df_CP_final = df_CP_final.sort_values('Diversity').reset_index()

df_B_final.to_excel('C:/Users/tommo/Downloads/Thesis_MAB/Data/df_B2.xlsx', index=False, engine='openpyxl')
df_CP_final.to_excel('C:/Users/tommo/Downloads/Thesis_MAB/Data/df_CP2.xlsx', index=False, engine='openpyxl')

print(df_B_final.to_string())
