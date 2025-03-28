"""
  Run this file at first, in order to see what is it printng. Instead of the print() use the respective log level
"""
############################### LOGGER
from abc import ABC, abstractmethod
from loguru import logger
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import random


class Bandit(ABC):
    """ """
    ##==== DO NOT REMOVE ANYTHING FROM THIS CLASS ====##

    @abstractmethod
    def __init__(self, p):
        pass

    @abstractmethod
    def __repr__(self):
        pass

    @abstractmethod
    def pull(self):
        """ """
        pass

    @abstractmethod
    def update(self):
        """ """
        pass

    @abstractmethod
    def experiment(self):
        """ """
        pass

    @abstractmethod
    def report(self):
        """ """
        # store data in csv
        # print average reward (use f strings to make it informative)
        # print average regret (use f strings to make it informative)
        pass

#--------------------------------------#


class Visualization():
    """ """

    def plot1(self, eg_rewards, ts_rewards):
        """

        Args:
          eg_rewards: 
          ts_rewards: 

        Returns:

        """
        plt.figure()
        plt.plot(eg_rewards, label='Epsilon Greedy')
        plt.plot(ts_rewards, label='Thompson Sampling')
        plt.title('Cumulative Rewards')
        plt.xlabel('Trials')
        plt.ylabel('Cumulative Reward')
        plt.legend()
        plt.grid()
        plt.savefig('plot1_rewards.png')
        plt.show()

    def plot2(self, eg_rewards, ts_rewards, p):
        """

        Args:
          eg_rewards: 
          ts_rewards: 
          p: 

        Returns:

        """
        max_p = max(p)
        eg_regret = [max_p * (i+1) - r for i, r in enumerate(eg_rewards)]
        ts_regret = [max_p * (i+1) - r for i, r in enumerate(ts_rewards)]

        plt.figure()
        plt.plot(eg_regret, label='Epsilon Greedy Regret')
        plt.plot(ts_regret, label='Thompson Sampling Regret')
        plt.title('Cumulative Regrets')
        plt.xlabel('Trials')
        plt.ylabel('Cumulative Regret')
        plt.legend()
        plt.grid()
        plt.savefig('plot2_regrets.png')
        plt.show()

#--------------------------------------#

class EpsilonGreedy(Bandit):
    """ """
    def __init__(self, p, epsilon=1.0):
        self.p = p
        self.n = len(p)
        self.epsilon = epsilon
        self.counts = np.zeros(self.n)
        self.values = np.zeros(self.n)
        self.total_reward = 0
        self.rewards = []
        self.choices = []
        self.algorithm = 'EpsilonGreedy'

    def __repr__(self):
        return f"EpsilonGreedy(epsilon={self.epsilon})"

    def pull(self):
        """ """
        if random.random() < self.epsilon:
            return np.random.randint(self.n)
        else:
            return np.argmax(self.values)

    def update(self, chosen_arm, reward):
        """

        Args:
          chosen_arm: 
          reward: 

        Returns:

        """
        self.counts[chosen_arm] += 1
        n = self.counts[chosen_arm]
        value = self.values[chosen_arm]
        self.values[chosen_arm] = ((n - 1) * value + reward) / n
        self.total_reward += reward
        self.rewards.append(self.total_reward)
        self.choices.append((chosen_arm, reward))

    def experiment(self, trials=20000):
        """

        Args:
          trials:  (Default value = 20000)

        Returns:

        """
        for t in range(1, trials + 1):
            self.epsilon = 1 / t
            arm = self.pull()
            reward = np.random.normal(self.p[arm], 1)
            self.update(arm, reward)

    def report(self):
        """ """
        df = pd.DataFrame(self.choices, columns=['Bandit', 'Reward'])
        df['Algorithm'] = self.algorithm
        df.to_csv('epsilon_greedy_results.csv', index=False)
        avg_reward = self.total_reward / len(self.choices)
        regret = np.sum(np.max(self.p) - np.array([self.p[choice[0]] for choice in self.choices]))
        logger.info(f"EpsilonGreedy - Average Reward: {avg_reward:.4f}")
        logger.info(f"EpsilonGreedy - Total Regret: {regret:.4f}")

#--------------------------------------#

class ThompsonSampling(Bandit):
    """ """
    def __init__(self, p):
        self.p = p
        self.n = len(p)
        self.alpha = np.ones(self.n)
        self.beta = np.ones(self.n)
        self.total_reward = 0
        self.rewards = []
        self.choices = []
        self.algorithm = 'ThompsonSampling'

    def __repr__(self):
        return "ThompsonSampling()"

    def pull(self):
        """ """
        samples = [np.random.beta(self.alpha[i], self.beta[i]) for i in range(self.n)]
        return np.argmax(samples)

    def update(self, chosen_arm, reward):
        """

        Args:
          chosen_arm: 
          reward: 

        Returns:

        """
        if reward >= 0:
            self.alpha[chosen_arm] += reward
        else:
            self.beta[chosen_arm] += abs(reward)
        self.total_reward += reward
        self.rewards.append(self.total_reward)
        self.choices.append((chosen_arm, reward))

    def experiment(self, trials=20000):
        """

        Args:
          trials:  (Default value = 20000)

        Returns:

        """
        for _ in range(trials):
            arm = self.pull()
            reward = np.random.normal(self.p[arm], 1)
            self.update(arm, reward)

    def report(self):
        """ """
        df = pd.DataFrame(self.choices, columns=['Bandit', 'Reward'])
        df['Algorithm'] = self.algorithm
        df.to_csv('thompson_sampling_results.csv', index=False)
        avg_reward = self.total_reward / len(self.choices)
        regret = np.sum(np.max(self.p) - np.array([self.p[choice[0]] for choice in self.choices]))
        logger.info(f"ThompsonSampling - Average Reward: {avg_reward:.4f}")
        logger.info(f"ThompsonSampling - Total Regret: {regret:.4f}")




def comparison():
    """ """
    p = [1, 2, 3, 4]

    eg = EpsilonGreedy(p)
    ts = ThompsonSampling(p)

    eg.experiment()
    ts.experiment()

    eg.report()
    ts.report()

    vis = Visualization()
    vis.plot1(eg.rewards, ts.rewards)
    vis.plot2(eg.rewards, ts.rewards, p)

if __name__=='__main__':
    comparison()
    logger.debug("debug message")
    logger.info("info message")
    logger.warning("warning message")
    logger.error("error message")
    logger.critical("critical message")
