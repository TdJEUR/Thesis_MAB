import numpy as np
from Helpers import generate_MAB_Matrix
from Multiple_Sims import Single_MAB_Multiple_Sims_Regret_Only
import matplotlib.pyplot as plt


# --------------------------------------------------------------------------------------------
# Input MAB Variables
# Number of Arms: N
no_arms = 10
# probabilities_of_success = [0.5, 0.8, 0.3, 0.5]
probabilities_of_success = 0
# Number of Rounds: T
number_of_rounds = 500
# Number of Different MAB Matrices:
num_mabs = 1000
# Number of simulations per MAB Matrix:
num_sims = 1
# --------------------------------------------------------------------------------------------
# Input Team Variables
# Exploration-Exploitation Temperature: Tau
taus = np.linspace(0, 0.05, 20)
taus = [round(tau, 5) for tau in taus]
# Composition of Team Experiential Learning Mechanisms - Input teams manually:
alphas = [[0.01, 0.01, 0.01, 0.01],
          [0.02, 0.02, 0.02, 0.02],
          [0.05, 0.05, 0.05, 0.05],
          [0.1, 0.1, 0.1, 0.1],
          [0.15, 0.15, 0.15, 0.15]]
# --------------------------------------------------------------------------------------------

# Define helper variables
number_of_trials = [1] * no_arms
total_calc = len(alphas)*num_mabs
count = 0

# Round Alphas
alphas = [[round(alpha, 4) for alpha in sublist] for sublist in alphas]

# Initialise final results for graphs:
performances_b = []
performances_cp = []

# For each different MAB:
for mab in range(num_mabs):
    # Initialise results for MAB:
    performance_b = {}
    performance_cp = {}
    # Generate MAB:
    if probabilities_of_success == 0:
        MAB_Matrix, best_arm = generate_MAB_Matrix(number_of_trials, np.random.beta(2, 2, no_arms), number_of_rounds)
    else:
        MAB_Matrix, best_arm = generate_MAB_Matrix(number_of_trials, probabilities_of_success, number_of_rounds)
    # For each alpha:
    for k, alpha in enumerate(alphas):
        # Initialise results for alpha:
        performance_per_tau_b = []
        performance_per_tau_cp = []
        # Calculate regret for every value of tau:
        avg_acc_regret_B, avg_acc_regret_CP, = Single_MAB_Multiple_Sims_Regret_Only(MAB_Matrix, best_arm, num_sims, taus, alpha)
        # Add list of results for each alpha: alpha_1: [[regrets (tau_1)], [regrets (tau_2)], ... ] ,
        #                                     alpha_2: [[regrets (tau_1)], [regrets (tau_2)], ... ] , ...
        # to the list of results for the MAB
        for j, tau in enumerate(taus):
            regret_B = avg_acc_regret_B[j][-1]
            performance_per_tau_b.append(avg_acc_regret_B[j][-1])
            regret_CP = avg_acc_regret_CP[j][-1]
            performance_per_tau_cp.append(avg_acc_regret_CP[j][-1])
        performance_b[f'{alpha}'] = performance_per_tau_b
        performance_cp[f'{alpha}'] = performance_per_tau_cp
        count += 1
        print(f'Finished all taus for alpha {k+1}/{len(alphas)} | '
              f'MAB {mab+1}/{num_mabs} | '
              f'{count*100/total_calc}%')
    performances_b.append(performance_b)
    performances_cp.append(performance_cp)

# Average all MABs into single dictionary
merged_dict_b = {}
merged_dict_cp = {}
for mab in performances_b:
    for key, value in mab.items():
        if key in merged_dict_b:
            merged_dict_b[key].append(value)
        else:
            merged_dict_b[key] = [value]
for mab in performances_cp:
    for key, value in mab.items():
        if key in merged_dict_cp:
            merged_dict_cp[key].append(value)
        else:
            merged_dict_cp[key] = [value]
averaged_dict_b = {}
averaged_dict_cp = {}
for key, value in merged_dict_b.items():
    averaged_dict_b[key] = [[sum(i) // len(value)] for i in zip(*value)]
for key, value in merged_dict_cp.items():
    averaged_dict_cp[key] = [[sum(i) // len(value)] for i in zip(*value)]

# Plot results
fig, axs = plt.subplots(nrows=1, ncols=2, figsize=(10, 8))
for key, values in performances_b[0].items():
    x = taus
    y = values
    regression = np.polyfit(x, y, 1)
    regression_fn = np.poly1d(regression)
    axs[0].set_title('Avg Beliefs')
    axs[0].plot(x, y, label=key)
    axs[0].plot(x, regression_fn(x), label='Regression: ' + key)
    axs[0].set_xlabel('Tau')
    axs[0].set_ylabel('Accumulated Regret')
    axs[0].legend()
for key, values in performances_cp[0].items():
    x = taus
    y = values
    regression = np.polyfit(x, y, 1)
    regression_fn = np.poly1d(regression)
    axs[1].set_title('Avg CP')
    axs[1].plot(x, y, label=key)  # 'o' between y, label=key for points
    axs[1].plot(x, regression_fn(x), label='Regression: ' + key)
    axs[1].set_xlabel('Tau')
    axs[1].set_ylabel('Accumulated Regret')
    axs[1].legend()
fig.tight_layout()
plt.draw()

print(f'Number of Taus:', len(taus))
print(f'Number of MABS:', num_mabs)

# Show results
plt.show()
