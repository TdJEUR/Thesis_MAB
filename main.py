from SoftMaxMAB import SoftmaxMAB
import matplotlib.pyplot as plt

true_arm_rewards = [0.4, 0.8, 0.4, 0.3]
# true_arm_stds = []
number_of_rounds = 100
tau = 10
alpha = 0.5

if __name__ == '__main__':

    bandit = SoftmaxMAB(true_arm_rewards, alpha, tau)
    final_beliefs, total_reward, accumulated_reward = bandit.play(number_of_rounds)
    plt.plot(range(number_of_rounds), accumulated_reward)
    plt.xlabel("Round")
    plt.ylabel("Accumulated Reward")
    plt.title(f"Accumulated Reward per Round, Tau = {tau}")
    plt.show()
