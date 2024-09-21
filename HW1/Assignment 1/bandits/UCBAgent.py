
import numpy as np
import sys
import copy
import time
import random
import argparse

class UCBAgent:
    def __init__(self, bandit):
        self.name = "Uma the UCB Agent"
        
        # code I added
        self.armJackpots = [0] * bandit.getNumArms()
        self.armPulls = [0] * bandit.getNumArms()
        # ends here
    
    def recommendArm(self, bandit, history):
        #Hey, your code goes here!
        
        # update our number of arm pulls and jackpots with the last pull results
        if history:
            self.armPulls[history[-1][0]] += 1
            if history[-1][1] == 1:
                self.armJackpots[history[-1][0]] += 1
        
        # we start this algorithm by pulling each arm once, to avoid dividing by zero
        if len(history) < bandit.getNumArms():
            return len(history)
        
        t = len(history) + 1
        
        bestArm = 0
        bestValue = 0
        for arm in range(len(self.armPulls)): # go through each arm
            u = np.sqrt((2 * np.log(t)) / self.armPulls[arm]) # calculate u
            value = (self.armJackpots[arm] / self.armPulls[arm]) + u # calculate the mean plus the u
            if value > bestValue:
                bestArm = arm
                bestValue = value
        
        return bestArm
        
        return False
