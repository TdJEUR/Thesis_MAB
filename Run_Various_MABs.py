import numpy as np
from Beliefs_Average import Sim_Matrix_X_Times_B
from CP_Average import Sim_Matrix_X_Times_CP
from Helpers import generate_MAB_Matrix


def Sim_Matrix_X_Times(no_arms, number_of_rounds, num_sims, tau, alphas):

    # Generate MAB Matrix:
    MAB_Matrix, best_arm = generate_MAB_Matrix(number_of_trials=[1] * no_arms,
                                               probabilities_of_success=np.random.beta(2, 2, no_arms),
                                               number_of_rounds=number_of_rounds)

    # Generate data for Averaging of Beliefs:
    avg_acc_regret_B, avg_acc_rewards_B, avg_rate_of_best_rewards_B = Sim_Matrix_X_Times_B(MAB_Matrix=MAB_Matrix,
                                                                                           best_arm=best_arm,
                                                                                           tau=tau,
                                                                                           alphas=alphas,
                                                                                           num_sims=num_sims)

    # Generate data for Averaging of Choice Probabilities:
    avg_acc_regret_CP, avg_acc_rewards_CP, avg_rate_of_best_rewards_CP = Sim_Matrix_X_Times_CP(MAB_Matrix=MAB_Matrix,
                                                                                               best_arm=best_arm,
                                                                                               tau=tau,
                                                                                               alphas=alphas,
                                                                                               num_sims=num_sims)

    return avg_acc_regret_B, avg_acc_rewards_B, avg_rate_of_best_rewards_B, avg_acc_regret_CP, avg_acc_rewards_CP, \
           avg_rate_of_best_rewards_CP
