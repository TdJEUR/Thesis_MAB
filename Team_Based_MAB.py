import numpy as np
from Softmax import softmax_belief_to_prob


# Team Based Classes and Functions

class TeamMember:

    def __init__(self, alpha, no_arms):
        self.alpha = alpha
        self.belief = [0]*no_arms
        self.choice_probabilities = [1/no_arms]*no_arms

    def get_choice_probabilities(self, tau):
        self.choice_probabilities = softmax_belief_to_prob(self.belief, tau)

    def update_beliefs(self, choice, reward):
        self.belief[choice] += self.alpha * (reward-self.belief[choice])


def create_team(alphas, no_arms):
    # Input list of alphas for all players and the number of arms
    # Output list of all
    team = []
    for alpha in alphas:
        team.append(TeamMember(alpha, no_arms))
    return team


def generate_team_choice_prob(team):
    choice_probs = [member.choice_probabilities for member in team]
    print(choice_probs)
    return choice_probs


# MAB based classes and functions

class MAB:

    def __init__(self, true_arm_rewards):
        self.pulls_per_arm = {i: 0 for i in range(1, len(true_arm_rewards)+1)}
        self.choices = []
        self.accumulated_rewards = []
        self.total_reward = 0
        self.true_arm_rewards = true_arm_rewards

    def play_round(self, choice):
        # Get reward from corresponding arm
        reward = self.true_arm_rewards[choice]
        # Update list of all choices
        self.choices.append(choice)
        # Update total reward and list of accumulated reward
        self.total_reward += reward
        self.accumulated_rewards.append(self.total_reward)
        return reward

    def get_choices_distribution(self):
        choices_ = [(elem, self.choices.count(elem)) for elem in set(self.choices)]
        return dict(choices_)


def create_MAB(true_arm_rewards):
    return MAB(true_arm_rewards)





