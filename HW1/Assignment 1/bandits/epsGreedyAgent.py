
import numpy as np
import sys
import copy
import time
import random
import argparse
######################################################

class epsGreedyAgent: 
    def __init__(self, bandit):
        self.name = "Eric the Epsilon Greedy Agent"
        
        # code I added
        self.epsilon = 0.1
        self.armJackpots = [0] * bandit.getNumArms()
        self.armPulls = [0] * bandit.getNumArms()
        # ends here
    
    def recommendArm(self, bandit, history):
        #Hey, your code goes here!
        numArms = bandit.getNumArms()
        
        if history:
            self.armPulls[history[-1][0]] += 1
            self.armJackpots[history[-1][0]] += history[-1][1]
        
        # random arm is pulled
        if (random.uniform(0.0, 1.0) <= self.epsilon):
            return random.choice(range(numArms))
        else: # greedy arm is pulled
            # calculate probabilities
            rewards = [0] * 10
            bestReward = 0
            bestArm = 0
            for i in range(10):
                if self.armPulls[i] > 0:
                    rewards[i] = self.armJackpots[i] / self.armPulls[i]
                    if rewards[i] > bestReward:
                        bestArm = i
                        bestReward = rewards[i]
                            
        return bestArm
        
        return False
