from Softmax import softmax_belief_to_prob
import numpy as np


# Team Based Classes and Functions

class TeamMember:

    def __init__(self, alpha, no_arms, tau):
        # Initialize team member alpha and initial belief (all options are equal)
        self.alpha = alpha
        self.belief = [0]*no_arms
        self.choice_probabilities = [1/no_arms]*no_arms
        self.tau = tau

    def get_choice_probabilities(self):
        # Convert beliefs to choice probabilities using the Softmax algorithm
        self.choice_probabilities = softmax_belief_to_prob(self.belief, self.tau)

    def update_beliefs(self, choice, reward):
        # Update beliefs due to last obtained reward using exponential smoothing
        # self.belief[choice] += ((1-self.alpha)*self.belief[choice])+(self.alpha*reward)
        self.belief[choice] += self.alpha * (reward - self.belief[choice])
        # print(f"Belief: {self.belief}")


def create_team(alphas, tau, no_arms):
    # Create list of team member objects
    team = []
    for alpha in alphas:
        team.append(TeamMember(alpha, no_arms, tau))
    return team


def generate_team_choice_prob(team):
    # Combine the individual choice probabilities of team members to a single
    # team choice probability. assumes that the options are independent of
    # each other and that the probabilities are accurate
    probabilities = [member.choice_probabilities for member in team]
    # combined_prob = [1.0] * len(probabilities[0])
    # for p in probabilities:
    #     for i, option_prob in enumerate(p):
    #         combined_prob[i] *= option_prob
    # total_prob = sum(combined_prob)
    # return [p / total_prob for p in combined_prob]

    weights = [1.0 / len(probabilities)] * len(probabilities)
    assert len(probabilities) == len(weights)
    combined_prob = [0.0] * len(probabilities[0])
    for i in range(len(combined_prob)):
        for j in range(len(probabilities)):
            combined_prob[i] += weights[j] * probabilities[j][i]
    total_prob = sum(combined_prob)
    return [p / total_prob for p in combined_prob]


# MAB based classes and functions

class MAB:

    def __init__(self, true_arm_rewards):
        # Initialize the MAB by inputting the true rewards of each arm
        # self.pulls_per_arm = {i: 0 for i in range(1, len(true_arm_rewards)+1)}
        self.true_arm_rewards = true_arm_rewards
        self.choices = []
        self.accumulated_rewards = []
        self.accumulated_regret = []
        self.rate_of_best_reward = []
        self.total_reward = 0
        self.total_regret = 0
        self.number_of_best_arm_pulls = 0
        self.round = 0

    def play_round(self, choice):
        # Update round number
        self.round += 1
        # Obtain a reward from the MAB depending on the chosen arm
        reward = self.true_arm_rewards[choice]
        print(f"Reward obtained: {reward}")
        self.total_reward += reward
        self.accumulated_rewards.append(self.total_reward)
        # Update regret
        self.total_regret += max(self.true_arm_rewards)-reward
        self.accumulated_regret.append(self.total_regret)
        # Update list of rate_of_best_reward (keep track of % best choice made)
        if reward == max(self.true_arm_rewards):
            self.number_of_best_arm_pulls += 1
        self.rate_of_best_reward.append(self.number_of_best_arm_pulls/self.round)
        # Update list of all choices (keep track of distribution of choices)
        self.choices.append(choice)
        return reward

    def get_choices_distribution(self):
        # Return the distribution of arms chosen
        choices_ = [(elem, self.choices.count(elem)) for elem in set(self.choices)]
        return dict(choices_)


def create_MAB(true_arm_rewards):
    # Create MAB object
    return MAB(true_arm_rewards)


def team_MAB(alphas, tau, true_arm_rewards, number_of_rounds):
    number_of_arms = len(true_arm_rewards)
    # Create MAB model
    mab = create_MAB(true_arm_rewards)
    # Create Team model
    team = create_team(alphas, tau, number_of_arms)
    # At each new round:
    for i in range(1, number_of_rounds+1):
        print(f"\nStarting round {i}")
        # Generate individual choice probabilities
        print("Generating individual choice probabilities for each team member:")
        for j, member in enumerate(team, 1):
            member.get_choice_probabilities()
            print(f"Member {j}\nbelief: {member.belief}\nchoice prob: {member.choice_probabilities}")
        # Generate team choice probabilities
        team_choice_prob = generate_team_choice_prob(team)
        print(f"Team choice prob: {team_choice_prob}")
        # Generate which arm is chosen out of the probabilities
        choice = np.random.choice(list(range(number_of_arms)), p=team_choice_prob)
        # Play a round depending on the choice
        reward = mab.play_round(choice)
        # cd = mab.get_choices_distribution()
        # print(f"Python index of picked arm: {choice}")
        # print(f"Distribution of picked arms: {cd}")
        # Update beliefs of all team members
        for member in team:
            member.update_beliefs(choice, reward)
            # print(f"Updated belief: {member.belief}")
    final_cd = mab.get_choices_distribution()
    print(f"Distribution of picked arms: {final_cd}")
    return mab.accumulated_rewards, mab.rate_of_best_reward, mab.accumulated_regret
