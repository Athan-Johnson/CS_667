import numpy as np
import sys
import copy
import time
import random
from randAgent import randomAgent
from epsGreedyAgent import epsGreedyAgent
from UCBAgent import UCBAgent
from thompsonAgent import thompsonAgent
import argparse
# code I added
import matplotlib.pyplot as plt
# ends here
######################################################
AGENTS_MAP = {'randomAgent' : randomAgent,
               'epsGreedyAgent' : epsGreedyAgent,
              'UCBAgent': UCBAgent,
              'thompsonAgent': thompsonAgent  }
                
class bandit: 
    def __init__(self, file):
        f = open(file, "r")
        lines = f.readlines()
        for i in range(len(lines)):
            lines[i] = lines[i].rstrip("\n")
        self.arms = []
        for i in range(1, len(lines)):
            self.arms.append(float(lines[i]))
        
    def pull_arm(self, arm):
        prob = self.arms[arm]

        randNum = random.random()
        if randNum <= prob:
            return 1
        else:
            return 0
    def getNumArms(self):
        return len(self.arms)
        
 


parser = argparse.ArgumentParser(description='Define bandit problem and agents.')
parser.add_argument('--input', choices=['input/test0.txt', 'input/test1.txt'], default='input/test1.txt', help='The input file, can be input/test0.txt or input/test1.txt')
parser.add_argument('--agent', choices=AGENTS_MAP.keys(), default='randomAgent', help='The bandit AI. Can be randomAgent, epsGreedyAgent, UCBAgent, or thompsonAgent')
parser.add_argument('--num_plays', type=int, default = 10000, help='The number of pulls an agent has.')
args = parser.parse_args()

testBandit = bandit(args.input)
agent = AGENTS_MAP[args.agent]()
history = []
cumulative_reward = 0

# code I added
cumulative_regret = [0]
test0data = [0.5, 0.6, 0.4, 0.3, 0.45, 0.7, 0.65, 0.43, 0.55, 0.45]
test1data = [0.2, 0.2, 0.2, 0.2, 0.9, 0.2, 0.2, 0.2, 0.2, 0.2]
# ends here


for numRuns in range(args.num_plays):
    testArm = agent.recommendArm(testBandit, history)
    reward = testBandit.pull_arm(testArm)
    cumulative_reward += reward
    history.append((testArm, reward))

    # code I added
    if args.input == 'input/test0.txt':
        cumulative_regret.append(cumulative_regret[-1] + 0.7 - test0data[testArm])
    else:
        cumulative_regret.append(cumulative_regret[-1] + 0.9 - test1data[testArm])
    #ends here

# code I added
range = range(len(cumulative_regret))

plt.plot(range, cumulative_regret)
plt.show()
# ends here

print(cumulative_reward)
    
