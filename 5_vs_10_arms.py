import numpy as np
from Helpers import generate_MAB_Matrix
from Multiple_Sims import Single_MAB_Multiple_Sims_Regret_Only, Single_MAB_Multiple_Sims_Reward_Only
import matplotlib.pyplot as plt


# --------------------------------------------------------------------------------------------
# Input MAB Variables
# Number of Arms: N
no_arms = [3, 5, 10, 15]
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
taus = np.linspace(0, 0.5, 20)
taus = [round(tau, 5) for tau in taus]
# Composition of Team Experiential Learning Mechanisms:
# Option to input teams manually:
alphas = [[0.05, 0.05, 0.05, 0.05, 0.05]]
# --------------------------------------------------------------------------------------------

# CP
res_cp = []
for i in no_arms:
    # Define helper variables
    number_of_trials = [1] * i
    total_calc = len(alphas) * num_mabs
    count = 0
    # Initialise final results for graphs
    all_acc_reward_cp = []
    # For each different MAB:
    for mab in range(num_mabs):
        # Generate MAB:
        if probabilities_of_success == 0:
            MAB_Matrix, best_arm = generate_MAB_Matrix(number_of_trials, np.random.beta(2, 2, i), number_of_rounds)
        else:
            MAB_Matrix, best_arm = generate_MAB_Matrix(number_of_trials, probabilities_of_success, number_of_rounds)
        # Store lists containing:
        MAB_acc_reward_cp = []
        # For each alpha:
        for k, alpha in enumerate(alphas):
            # Calculate regret for every value of tau:
            avg_acc_reward_B, avg_acc_reward_CP, = Single_MAB_Multiple_Sims_Reward_Only(MAB_Matrix, best_arm, num_sims, taus, alpha)
            # Add list of results for each alpha: alpha_1: [[regrets (tau_1)], [regrets (tau_2)], ... ] ,
            #                                     alpha_2: [[regrets (tau_1)], [regrets (tau_2)], ... ] , ...
            # to the list of results for the MAB
            for l, tau in enumerate(taus):
                MAB_acc_reward_cp.append(avg_acc_reward_CP[l][-1])
            count += 1
            print(f'Finished all taus for alpha {k + 1}/{len(alphas)} | '
                  f'MAB {mab + 1}/{num_mabs} | '
                  f'{count * 100 / total_calc}%')
        # Add all results for all alphas to the total results containing blocks for all MABs:
        all_acc_reward_cp.append(MAB_acc_reward_cp)
    # Average all MABs to graph
    avg_acc_reward_cp = np.mean(np.array(all_acc_reward_cp), axis=0)
    # Add results to res:
    res_cp.append(avg_acc_reward_cp)

# B
res_b = []
for i in no_arms:
    # Define helper variables
    number_of_trials = [1] * i
    total_calc = len(alphas) * num_mabs
    count = 0
    # Initialise final results for graphs
    all_acc_reward_b = []
    # For each different MAB:
    for mab in range(num_mabs):
        # Generate MAB:
        if probabilities_of_success == 0:
            MAB_Matrix, best_arm = generate_MAB_Matrix(number_of_trials, np.random.beta(2, 2, i), number_of_rounds)
        else:
            MAB_Matrix, best_arm = generate_MAB_Matrix(number_of_trials, probabilities_of_success, number_of_rounds)
        # Store lists containing:
        MAB_acc_reward_b = []
        # For each alpha:
        for k, alpha in enumerate(alphas):
            # Calculate regret for every value of tau:
            avg_acc_reward_B, avg_acc_reward_CP, = Single_MAB_Multiple_Sims_Reward_Only(MAB_Matrix, best_arm, num_sims, taus, alpha)
            # Add list of results for each alpha: alpha_1: [[regrets (tau_1)], [regrets (tau_2)], ... ] ,
            #                                     alpha_2: [[regrets (tau_1)], [regrets (tau_2)], ... ] , ...
            # to the list of results for the MAB
            for l, tau in enumerate(taus):
                MAB_acc_reward_b.append(avg_acc_reward_B[l][-1])
            count += 1
            print(f'Finished all taus for alpha {k + 1}/{len(alphas)} | '
                  f'MAB {mab + 1}/{num_mabs} | '
                  f'{count * 100 / total_calc}%')
        # Add all results for all alphas to the total results containing blocks for all MABs:
        all_acc_reward_b.append(MAB_acc_reward_b)
    # Average all MABs to graph
    avg_acc_reward_b = np.mean(np.array(all_acc_reward_b), axis=0)
    # Add results to res:
    res_b.append(avg_acc_reward_b)


fig, axs = plt.subplots(nrows=1, ncols=2, figsize=(10, 8))
for m, arms in enumerate(no_arms):
    # Plot results (Single team)
    axs[0].plot(taus, res_b[m], label=f'{arms} Arms')
    axs[1].plot(taus, res_cp[m], label=f'{arms} Arms')
# Add labels and title to the plots
axs[0].set_xlabel('Tau')
axs[0].set_ylabel('Reward')
axs[0].set_title('Averaging Beliefs')
axs[0].legend()
# Add labels and title to the plot
axs[1].set_xlabel('Tau')
axs[1].set_ylabel('Reward')
axs[1].set_title('Averaging Choice Probabilities')
axs[1].legend()

plt.show()
