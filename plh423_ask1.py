# import modules

import numpy as np
import matplotlib.pyplot as plt

class Bandit:

    def __init__(self, a_i, b_i):
        self.a = a_i
        self.b = b_i
        self.mu_i = (self.b + self.a)/2
        self.pulls = 0

    def pull(self):
        self.pulls += 1
        return np.random.uniform(self.a, self.b)

    def getpulls(self):
        return self.pulls



class MABagent:

    def __init__(self, cur_env):
        self.env_name = ['default_Env', 'Env2', 'Env3']
        if cur_env in self.env_name:
            self.cur_env = cur_env
        else:
            self.cur_env = 'default_Env'

        if self.cur_env == 'default_Env':
            self.k = 10                             # number of arms
            self.T = 1000                           # horizon
            self.epsilon_decay = (self.k * np.log(self.T)) ** (-1/3) / self.T ** (-1/3)             # exponential decay rate
        elif self.cur_env == 'Env2':
            self.k = 10  # number of arms
            self.T = 3000  # horizon
            self.epsilon_decay = (self.k * np.log(self.T)) ** (-1/3) / self.T ** (-1/3)             # exponential decay rate
        else:
            self.k = 30  # number of arms
            self.T = 5000  # horizon
            self.epsilon_decay = (self.k * np.log(self.T)) ** (-1/3) / self.T ** (-1/3)  # exponential decay rate



        self.N = 20  # explore rounds
        self.epsilon = 1.0                      # exploration rate
        self.epsilon_min = 0.01                 # minimum exploration rate
        self.e_greedy = False
        self.ucb = False

        self.bandit_score = np.zeros((self.k,))  # total score of each arm
        self.bandit_ucb = np.zeros((self.k,))    # upper confidence bound of each arm
        # self.pulls = np.zeros((self.k,))         # num of arm pulls
        self.inst_score = np.zeros((self.T,))    # reward for round t
        self.best_score = np.zeros((self.T,))    # cumulative reward of best arm for round t
        self.alg_score = np.zeros((self.T,))     # cumulative reward for round t
        self.regret = np.zeros((self.T,))        # regret for round t

    def reset(self):
        self.bandit_score = np.zeros((self.k,))  # total score of each arm
        self.pulls = np.zeros((self.k,))         # num of arm pulls
        self.inst_score = np.zeros((self.T,))    # reward for round t
        self.alg_score = np.zeros((self.T,))     # cumulative reward for round t
        self.regret = np.zeros((self.T,))        # regret for round t

    def calculate_regret(self, best):
        for i in range(self.T):
            if i > 0:
                self.alg_score[i] = self.alg_score[i - 1] + self.inst_score[i]  # vector keeping track of cummulative explore-then-eploit reward at all times
            else:
                self.alg_score[i] = self.inst_score[i]
            if i > 0:
                self.best_score[i] = self.best_score[i - 1] + best  # vector keeping track of t*optimal reward (cummulative reward)
            else:
                self.best_score[i] = best

            self.regret[i] = (self.best_score[i] - self.alg_score[i]) / (i + 1)   # regret per iteration at round t

    def grapher(self, regret_greedy, regret_ucb):
        plt.plot(np.arange(1, self.T + 1), regret_greedy)
        plt.title("e-Greedy Performance [T = %d, k = %d]" % (self.T, self.k))
        plt.xlabel("Round T")
        plt.ylabel("Regret")
        plt.show()

        plt.plot(np.arange(1, self.T + 1), regret_ucb)
        plt.title("UCB Performance [T = %d, k = %d]" % (self.T, self.k))
        plt.xlabel("Round T")
        plt.ylabel("Regret")
        plt.show()

        plt.plot(np.arange(1, self.T + 1), regret_greedy, color='r', label='e-Greedy')
        plt.plot(np.arange(1, self.T + 1), regret_ucb, color='b', label='UCB')
        plt.title("e-Greedy and UCB common plot [T = %d, k = %d]" % (self.T, self.k))
        plt.xlabel("Round T")
        plt.ylabel("Regret")

        plt.legend()
        plt.show()

    def reduce_epsilon(self):
        if self.epsilon > self.epsilon_min:
            self.epsilon *= (1 - self.epsilon_decay)

    def act(self):
        # decision if greedy
        if self.e_greedy:
            if self.epsilon <= np.random.rand():
                self.reduce_epsilon()
                return np.argmax(self.bandit_score)                      # return id of best arm
            else:
                self.reduce_epsilon()
                return np.random.choice(self.k)                          # return id of random arm
        # decision if ucb
        if self.ucb:
            return np.argmax(self.bandit_ucb)                          # return id of arm with highest ucb


    def begin(self, bandit):
        # run each arm once
        for j in range(self.k):
            score = bandit[j].pull()                  # get a reward for arm j
            self.inst_score[j] = self.best_score[j] = score             # record reward of algorithm at that instant
            self.bandit_score[j] += score                               # update the total score of arm j
            # self.pulls[j] += 1                                          # update how many times each arm was pulled
            # if we run ucb calculate ucb of each arm for the first rounds played
            if self.ucb:
                self.bandit_ucb[j] = self.bandit_score[j] + np.sqrt(2*np.log(j + 1) / bandit[j].getpulls())
        # run the algorithm
        for i in range(self.k, self.T):
            arm = self.act()                                            # choose arm
            score = bandit[arm].pull()                # play arm
            # print('best arm: %d ' % arm)
            self.inst_score[i] = score
            # self.pulls[arm] += 1                                        # update how many times arm was pulled
            self.bandit_score[arm] = ((self.bandit_score[arm] * bandit[arm].getpulls() - 1) +
                                      score) / bandit[arm].getpulls()          # update the total score of best arm
            # if we run ucb also calculate the upper confidence bound of arm
            if self.ucb:
                self.bandit_ucb[arm] = self.bandit_score[arm] + np.sqrt(2*np.log(i + 1) / bandit[arm].getpulls())

    def run(self):
        # create the enviroment
        # bandit = np.random.random((self.k,))                            # generate success prob. for each arm
        # print('best arm = %d' % np.argmax(bandit))

        # create the enviroment asked: reward that is uniformly distributed in [a_i, b_i].
        bandit = []
        for i in range(self.k):
            a = np.random.uniform(0, 1)
            b = np.random.uniform(0, 1)
            while b == a:
                b = np.random.uniform(0, 1)
            if b < a:
                temp = b
                b = a
                a = temp
            bandit.append(Bandit(a, b))
            print('arm = %d: range = (%f,%f) : ' % (i, a, b))

        best = np.amax([obj.mu_i for obj in bandit]) # best arm is the arm with the bigest mu = a+b/2
        self.e_greedy = True
        # self.ucb = True
        self.begin(bandit)
        self.calculate_regret(best)
        regret_greedy = self.regret

        self.reset()
        self.e_greedy = False
        self.ucb = True
        self.begin(bandit)
        self.calculate_regret(best)
        regret_ucb = self.regret
        self.grapher(regret_greedy, regret_ucb)


if __name__ == '__main__':
    agent = MABagent('Env1')
    agent.run()
    agent = MABagent('Env2')
    agent.run()
    agent = MABagent('Env3')
    agent.run()


