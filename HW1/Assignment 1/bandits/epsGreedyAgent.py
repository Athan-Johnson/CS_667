
import numpy as np
import sys
import copy
import time
import random
import argparse
######################################################

class epsGreedyAgent: 
    def __init__(self):
        self.name = "Eric the Epsilon Greedy Agent"
        
        # code I added
        self.epsilon = 0.1
        self.armJackpots = [0] * 10 # I know this is hardcoded but it speeds up efficiency
        self.armAppearances = [0] * 10 # otherwise I have to recompute this each time or pass the bandit in init
        # passing the bandit means I must modify all other agents
        # overall it runs faster to hardcode this than use history
        # ends here
    
    def recommendArm(self, bandit, history):
        #Hey, your code goes here!
        numArms = bandit.getNumArms()
        
        if history:
            self.armAppearances[history[-1][0]] += 1
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
                if self.armAppearances[i] > 0:
                    rewards[i] = self.armJackpots[i] / self.armAppearances[i]
                    if rewards[i] > bestReward:
                        bestArm = i
                        bestReward = rewards[i]
                            
        print(bestArm)
        return bestArm
        
        return False
