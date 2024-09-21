
import numpy as np
import sys
import copy
import time
import random
import argparse

class thompsonAgent: 
    def __init__(self, bandit):
        self.name = "Terry the Thompson Sampling Agent"
        
        # code I added
        self.armRewards = [1] * bandit.getNumArms() # these must be initialized to one
        self.armFails = [1] * bandit.getNumArms()
        # ends here
    
    def recommendArm(self, bandit, history):
        #Hey, your code goes here!
        
        if history:
            if history[-1][1] == 1:
                self.armRewards[history[-1][0]] += 1
            else:
                self.armFails[history[-1][0]] += 1
                
        # get a random value from each distribution
        samples = [0] * bandit.getNumArms()
        samples = np.random.beta(self.armRewards, self.armFails)
        
        return np.argmax(samples)
        
        return False
