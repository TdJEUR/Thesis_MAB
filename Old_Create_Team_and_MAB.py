from Helpers import softmax_belief_to_prob
import numpy as np


def create_team(alphas, tau, no_arms):
    """Create a list of TeamMember objects. The verbosity of each member is
    determined by alphas (list containing alpha for each member) and the team
    exploration-exploitation strategy is determined by tau (float). The number
    of arms of the MAB to be played must also be specified using no_arms (int)"""
    team = []
    for alpha in alphas:
        team.append(TeamMember(alpha, no_arms, tau))
    return team


def create_MAB(true_arm_rewards, true_arm_stds):
    """ Create an MAB object that simulates a MAB with rewards corresponding to
    true_arm_rewards (list containing reward of each arm) """
    return MAB(true_arm_rewards, true_arm_stds)


class TeamMember:
    """ A TeamMember has attributes alpha (verbosity), tau (exploration-exploitation
    strategy), and no_arms (number of arms of the MAB that the member will play). A
    TeamMember can convert their belief to a choice probability and update their belief
    by incorporating the obtained reward associated with a choice """

    def __init__(self, alpha, no_arms, tau):
        """ Initialize team member alpha (float), tau (float), belief (all options
        are equal when initializing) and choice_probabilities (all options are
        equal when initializing) """
        self.alpha = alpha
        self.belief = [0]*no_arms
        self.choice_probabilities = [1/no_arms]*no_arms
        self.tau = tau

    def get_choice_probabilities(self):
        """ Convert member's beliefs to choice probabilities using Softmax """
        self.choice_probabilities = softmax_belief_to_prob(self.belief, self.tau)

    def update_beliefs(self, choice, reward):
        """ Update beliefs by incorporating the obtained reward associated with
        the member's choice in accordance with exponential smoothing"""
        self.belief[choice] = (self.alpha*reward) + ((1-self.alpha)*self.belief[choice])
        #self.belief[choice] += self.alpha * (reward - self.belief[choice])


class MAB:

    def __init__(self, true_arm_rewards, true_arm_stds):
        """ Initialize a MAB by inputting the true rewards of each arm
        in true_arm_rewards (list containing reward of each arm) """
        self.true_arm_rewards = true_arm_rewards
        self.true_arm_stds = true_arm_stds
        self.choices = []
        self.accumulated_rewards = []
        self.accumulated_regret = []
        self.rate_of_best_reward = []
        self.total_reward = 0
        self.total_regret = 0
        self.number_of_best_arm_pulls = 0
        self.round = 0

    def play_round(self, choice):
        """ Play a single round on the MAB. Input a choice (python index
        of the chosen arm). A corresponding reward will then be generated.
        The function updates the round number, total reward and list of accumulated
        reward for each round, total regret and list of accumulated regret
        for each round, list of rate of choosing the best reward and list
        of all choices """
        # Update round number
        self.round += 1
        # Obtain a reward from the MAB depending on the chosen arm
        mean = self.true_arm_rewards[choice]
        std = self.true_arm_stds[choice]
        reward = np.random.normal(loc=mean, scale=std)
        self.total_reward += reward
        self.accumulated_rewards.append(self.total_reward)
        # Update regret
        self.total_regret += max(self.true_arm_rewards)-reward
        self.accumulated_regret.append(self.total_regret)
        # Update list of rate_of_best_reward (keep track of % best choice made)
        if mean == max(self.true_arm_rewards):
            self.number_of_best_arm_pulls += 1
        self.rate_of_best_reward.append(self.number_of_best_arm_pulls/self.round)
        # Update list of all choices (keep track of distribution of choices)
        self.choices.append(choice)
        return reward

    def get_choices_distribution(self):
        """ Return the distribution of how many times each arm was chosen """
        choices_ = [(elem, self.choices.count(elem)) for elem in set(self.choices)]
        return dict(choices_)
