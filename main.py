from MAB import MultiArmedBandit


if __name__ == '__main__':
    num_arms = 5
    num_rounds = 1000
    mean_rewards = [1, 2, 3, 4, 5]
    std_dev_rewards = [0, 3, 2, 1, 4]
    tau = 0.6

    bandit = MultiArmedBandit(num_arms, mean_rewards, std_dev_rewards, tau)
    true_mean_rewards, estimated_mean_rewards, total_reward = bandit.play(num_rounds)

    print("True mean rewards:", true_mean_rewards)
    print("Estimated mean rewards:", estimated_mean_rewards)
    print("Total reward obtained:", total_reward)

