import numpy as np
from utils import *

class SequentiAlg:
    def __init__(self, bandit, name="Sequential", params={'init': -np.inf}):
        self.bandit = bandit
        self.name = name
        self.params = params
        self.reset()
    
    def reset(self, horizon=None):
        self.all_selected = False
        self.means = np.zeros(self.bandit.nbr_arms)
        self.nbr_pulls = np.zeros(self.bandit.nbr_arms, int)
        self.indices = np.zeros(self.bandit.nbr_arms) + self.params['init']
        self.time = 0
        if self.params is not None:
            if 'set_constants_func' in self.params.keys():
                self.params['set_constants_func'](self.params, bandit, horizon)
    
    def __repr__(self):
        res = f"{self.name} algorithm - time step = {self.time}\n"
        for i in range(self.bandit.nbr_arms):
            res += "  "
            res += str(self.bandit.arms[i])
            res += " : "
            res += f"est. mean = {self.means[i]:.3f} - "
            res += f"nbr. pulls = {self.nbr_pulls[i]}\n"
        return res
    
    def __str__(self):
        res = f"{self.name} algorithm - time step = {self.time}\n"
        for i in range(self.bandit.nbr_arms):
            res += "  "
            res += str(self.bandit.arms[i])
            res += " : "
            res += f"est. mean = {self.means[i]:.3f} - "
            res += f"nbr. pulls = {self.nbr_pulls[i]}\n"
        return res
    
    def update_statistics(self, arm, r):
        self.time += 1
        self.means[arm] = (self.means[arm]*self.nbr_pulls[arm]+r)/(self.nbr_pulls[arm]+1)
        self.nbr_pulls[arm] += 1
        if not self.all_selected:
            self.all_selected = np.all(self.nbr_pulls > 0)
        self.compute_indices()
    
    def pull(self, arm):
        r = self.bandit.pull(arm)
        self.update_statistics(arm, r)
        return r
    
    def compute_indices(self):
        if self.all_selected:
            # probably faster computation is possible
            self.indices = np.random.rand(self.bandit.nbr_arms)
        else:
            # probably slower computation
            self.indices = np.random.randn(self.bandit.nbr_arms)
        
    def choose_an_arm(self):
        return randamin(self.indices)
        
    def play(self):
        arm = self.choose_an_arm()
        r = self.pull(arm)
        return arm, r

    def fit(self, horizon, reset=True, experiment=None):
        if reset:
            self.reset(horizon)
        if experiment is not None:
            for t in range(horizon):
                arm, r = self.play()
                experiment(arm, r, t)
        else:
            for _ in range(horizon):
                arm, r = self.play()
