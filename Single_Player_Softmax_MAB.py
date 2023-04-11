import matplotlib.pyplot as plt
import numpy as np


""" Simulate a single player playing a MAB with rewards for each arm 
corresponding to true_arm_rewards (list) for a certain number_of_rounds 
(int). The player's verbosity is defined by alpha (float) and their 
exploration-exploitation strategy is defined by tau (float). To use this
script simply input the desired values below and press run """


# MAB Values:
true_arm_rewards = [0.4, 0.8, 0.4, 0.3]
number_of_rounds = 50
# true_arm_stds = []

# PLayer Values
tau = 0.5
alpha = 0.5


class SoftmaxMAB:

    def __init__(self, true_arm_rewards, alpha, tau):
        self.pulls_per_arm = {i: 0 for i in range(1, len(true_arm_rewards)+1)}
        self.beliefs = np.zeros(len(true_arm_rewards))
        self.choices = []
        self.accumulated_rewards = []
        self.total_reward = 0
        self.true_arm_rewards = true_arm_rewards
        self.alpha = alpha
        self.tau = tau

    def beliefs_to_choice_probabilities(self):
        # SOFTMAX CHOICE RULE:
        # Input the player's current beliefs and output their choice probabilities
        choice_probabilities = []
        denominator = sum(np.exp(i/(self.tau/len(self.beliefs))) for i in self.beliefs)
        for belief in self.beliefs:
            choice_probabilities.append(round((np.exp(belief/(self.tau/len(self.beliefs)))/denominator), 10))
        print(f"Beliefs: {self.beliefs}\nChoice Probabilities: {choice_probabilities}")
        return choice_probabilities

    def choose_from_beliefs(self, choice_probabilities):
        # SELECT ARM, GET REWARD, UPDATE BELIEFS:
        # Choose which arm to play according to the player's choice probabilities
        choice = np.random.choice(list(range(len(self.true_arm_rewards))), p=choice_probabilities)
        print(f"Python index of picked arm: {choice}")
        # Get reward from corresponding arm
        reward = self.true_arm_rewards[choice]
        # Update list of all choices
        self.choices.append(choice)
        # Update total reward and list of accumulated reward
        self.total_reward += reward
        self.accumulated_rewards.append(self.total_reward)
        # Update beliefs
        self.beliefs[choice] += self.alpha*(reward-self.beliefs[choice])

    def play(self, number_of_rounds):
        for i in range(number_of_rounds):
            cp = self.beliefs_to_choice_probabilities()
            self.choose_from_beliefs(cp)
        choices_ = [(elem, self.choices.count(elem)) for elem in set(self.choices)]
        print(f"Beliefs: {self.beliefs}\nTotal Reward: {self.total_reward}")
        print(f"Distribution of picked arms: {dict(choices_)}")
        return self.beliefs, self.total_reward, self.accumulated_rewards


if __name__ == '__main__':
    bandit = SoftmaxMAB(true_arm_rewards, alpha, tau)
    final_beliefs, total_reward, accumulated_reward = bandit.play(number_of_rounds)
    plt.plot(range(number_of_rounds), accumulated_reward)
    plt.xlabel("Round")
    plt.ylabel("Accumulated Reward")
    plt.title(f"Accumulated Reward per Round, Tau = {tau}")
    plt.show()
