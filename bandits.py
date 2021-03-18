import numpy as np
from .utils import klGaussian, klBernoulli

########################################
#                 Arms                 #
########################################
class Normal:
    def __init__(self, mean, std=1.):
        self.mean = mean
        self.std = std
        self.name = "Normal"

    def __repr__(self):
        return f"{self.name} (mean={self.mean:.3f}, std={self.std:.3f})"

    def sample(self):
        return np.random.normal(loc=self.mean, scale=self.std)

class Bernoulli:
    def __init__(self, mean):
        assert mean >= 0 and mean <= 1, f"The mean of a Bernoulli should between 0 and 1: mean={mean}"
        self.mean = mean
        self.std = mean * (1 - mean)

    def __repr__(self):
        return f"{self.name} (mean={self.mean:.3f})"

    def sample(self):
        return np.random.binomial(1, self.mean)


########################################
#               Bandit                 #
########################################
class Bandit:
    def __init__(self, arms, structure="unknown", kullback=None, complexity=None):
        self.arms = arms
        self.nbr_arms = len(arms)
        self.structure = structure
        self.kullback = kullback
        self.complexity_func = complexity
        
        # Preprocessing of useful statistics
        
        ## Expected value of arms
        ### Bandits community vocabulary
        self.rewards = np.array([arms[i].mean for i in range(self.nbr_arms)])
        ### Probability community vocabulary
        self.means = self.rewards
        
        ## Best arm index (one of) and expected value (unique)
        self.best_arm = np.argmax(self.rewards)
        self.best_reward = np.max(self.rewards)
        self.best_mean = self.best_reward
        
        ## Regret/suboptimality gap of arms
        self.regrets = self.best_reward - self.rewards

    def __str__(self):
        return f"Bandit({self.arms})"

    def __repr__(self):
        return f"Bandit({self.arms})"
    
    # Bandits community vocabulary
    def pull(self, arm):
        return self.arms[arm].sample()
    
    # Probability community vocabulary
    def sample(self, idx):
        return self.pull(idx)
    
    def complexity(self, kullback=None):
        if self.complexity_func is not None:
            return self.complexity_func(self)
        elif self.kullback is not None:
            kullback = self.kullback
        else:
            assert (kullback is not None), "Kullback Leibler divergence should be specified"
        suboptimal_arms = np.where(self.regrets != 0.)[0]
        term_1 = self.regrets[suboptimal_arms]
        term_2 = kullback(self.rewards[suboptimal_arms], self.best_reward)
        c = sum(term_1 / term_2)
        return c


class NormalBandit(Bandit):
    def __init__(self, means, stds=None, structure="unknown", complexity=None):
        assert len(means) > 0, "means should not be empty"
        if stds is not None:
            assert len(means) == len(stds), \
            f"Lengths should match: len(means)={len(means)} - len(stds)={len(stds)}"
            arms = [Normal(m, s) for m,s in zip(means, stds)]
        else:
            arms = [Normal(m) for m in means]
        Bandit.__init__(self, arms, structure=structure, kullback=klGaussian, complexity=None)
        
class BernoulliBandit(Bandit):
    def __init__(self, means, structure="unknown", complexity=None):
        assert len(means) > 0, "means should not be empty"
        assert np.all(means >= 0) and np.all(means <= 1), \
        "Bernoulli mean should be between 0 and 1:\n(means={means})"
        arms = [Bernoulli(m) for m in means]
        Bandit.__init__(self, arms, structure=structure, kullback=klBernoulli, complexity=None)
