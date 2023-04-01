import numpy as np


class MultiArmedBandit:
    def __init__(self, num_arms, mean_rewards, std_dev_rewards, tau):
        self.num_arms = num_arms
        self.mean_rewards = mean_rewards
        self.std_dev_rewards = std_dev_rewards
        self.tau = tau

        # Initialize the true mean rewards for each arm
        self.true_mean_rewards = np.array(mean_rewards)

        # Initialize the estimated mean rewards for each arm
        self.estimated_mean_rewards = np.zeros(num_arms)

        # Initialize the number of times each arm has been played
        self.arm_counts = np.zeros(num_arms)

        # Initialize the total reward obtained so far
        self.total_reward = 0

    def softmax(self, q):
        # Compute the softmax probabilities
        exp_q = np.exp(q / self.tau)
        return exp_q / np.sum(exp_q)

    def play(self, num_rounds):
        # Play the bandit for num_rounds
        for t in range(num_rounds):
            # Compute the softmax probabilities for selecting each arm
            softmax_probabilities = self.softmax(self.estimated_mean_rewards)

            # Choose an arm based on the softmax probabilities
            arm = np.random.choice(np.arange(self.num_arms), p=softmax_probabilities)

            # Get the reward for the chosen arm
            reward = np.random.normal(self.mean_rewards[arm], self.std_dev_rewards[arm])

            # Update the estimated mean reward for the chosen arm
            self.arm_counts[arm] += 1
            alpha = 1 / self.arm_counts[arm]
            self.estimated_mean_rewards[arm] += alpha * (reward - self.estimated_mean_rewards[arm])

            # Update the total reward obtained so far
            self.total_reward += reward

        # Return the true mean rewards, estimated mean rewards, and total reward obtained
        return self.true_mean_rewards, self.estimated_mean_rewards, self.total_reward
