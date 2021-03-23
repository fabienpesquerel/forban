import numpy as np
from datetime import datetime as date
import matplotlib.pyplot as plt
from matplotlib import colors as mapcol



########################################
#              Plot Bandit             #
########################################
def plot_bandit(bandit):
    plt.figure(figsize=(12,7))
    plt.xticks(range(0,bandit.nbr_arms), range(0,bandit.nbr_arms))
    plt.plot(range(0,bandit.nbr_arms), bandit.rewards)
    plt.scatter(range(0,bandit.nbr_arms), bandit.rewards)
    plt.show()

########################################
#           KL Divergences             #
########################################
def klBernoulli(mean_1, mean_2, eps=1e-15):
    """Kullback-Leibler divergence for Bernoulli distributions."""
    x = np.minimum(np.maximum(mean_1, eps), 1-eps)
    y = np.minimum(np.maximum(mean_2, eps), 1-eps)
    return x*np.log(x/y) + (1-x)*np.log((1-x)/(1-y))

def klGaussian(mean_1, mean_2, sig2=1.):
    """Kullback-Leibler divergence for Gaussian distributions."""
    return ((mean_1-mean_2)**2)/(2*sig2)


########################################
#         Argument selectors           #
########################################

def randamax(V, T=None, I=None):
    """
    V: array of values
    T: array used to break ties
    I: array of indices from which we should return an amax
    """
    if I is None:
        idxs = np.where(V == np.amax(V))[0]
        if T is None:
            idx = np.random.choice(idxs)
        else:
            assert len(V) == len(T), f"Lengths should match: len(V)={len(V)} - len(T)={len(T)}"
            t_idxs = np.where(T[idxs] == np.amin(T[idxs]))[0]
            t_idxs = np.random.choice(t_idxs)
            idx = idxs[t_idxs]
    else:
        idxs = np.where(V[I] == np.amax(V[I]))[0]
        if T is None:
            idx = I[np.random.choice(idxs)]
        else:
            assert len(V) == len(T), f"Lengths should match: len(V)={len(V)} - len(T)={len(T)}"
            t = T[I]
            t_idxs = np.where(t[idxs] == np.amin(t[idxs]))[0]
            t_idxs = np.random.choice(t_idxs)
            idx = I[idxs[t_idxs]]
    return idx

def randamin(V, T=None, I=None):
    """
    V: array of values
    T: array used to break ties
    I: array of indices from which we should return an amax
    """
    if I is None:
        idxs = np.where(V == np.amin(V))[0]
        if T is None:
            idx = np.random.choice(idxs)
        else:
            assert len(V) == len(T), f"Lengths should match: len(V)={len(V)} - len(T)={len(T)}"
            t_idxs = np.where(T[idxs] == np.amin(T[idxs]))[0]
            t_idxs = np.random.choice(t_idxs)
            idx = idxs[t_idxs]
    else:
        idxs = np.where(V[I] == np.amin(V[I]))[0]
        if T is None:
            idx = I[np.random.choice(idxs)]
        else:
            assert len(V) == len(T), f"Lengths should match: len(V)={len(V)} - len(T)={len(T)}"
            t = T[I]
            t_idxs = np.where(t[idxs] == np.amin(t[idxs]))[0]
            t_idxs = np.random.choice(t_idxs)
            idx = I[idxs[t_idxs]]
    return idx


########################################
#             Experiments              #
########################################

class Experiment:
    def __init__(self, sequential_algorithms, bandit,
                 statistics={'mean':True, 'std':True, 'quantile':False, 'pulls':False},
                 quantile_ticks=[0.1, 0.9, 0.5],
                 kullback=None, complexity=False):
        assert len(sequential_algorithms) > 0
        self.algorithms = sequential_algorithms
        self.bandit = bandit
        self.nbr_algo = len(sequential_algorithms)
        self.algo_idx = 0
        self.statistics = {}
        self.regret = None
        self.path = None
        self.horizon = None
        self.nbr_exp = None
        self.complexity = None
        if complexity:
            if kullback is not None:
                self.complexity = bandit.complexity(kullback)
            else:
                self.complexity = bandit.complexity()
        for algo_idx, algo in enumerate(self.algorithms):
            self.statistics[algo_idx] = {'name': algo.name}
            for s in statistics.items():
                if s[1]:
                    self.statistics[algo_idx][s[0]] = None
        stats = self.statistics[0].keys()
        self.stats = stats
        self.regret_flag = ('mean' in stats) or ('std' in stats) or ('quantile' in stats)
        self.quantile_ticks = quantile_ticks
        self.path_flag = 'pulls' in stats
        
    def __call__(self, arm, r, t):
        if self.regret_flag:
            self.regret[t] = self.bandit.regrets[arm]
        if self.path_flag:
            self.path[t] = arm

    def run(self, nbr_exp=500, horizon=50):
        self.nbr_exp = nbr_exp
        self.horizon = horizon
        for algo_idx, algo in enumerate(self.algorithms):
            if self.regret_flag:
                regret = np.zeros((nbr_exp, horizon))
            if self.path_flag:
                path = np.zeros((nbr_exp, horizon), int)
            for i in range(nbr_exp):
                if self.regret_flag:
                    self.regret = regret[i]
                if self.path_flag:
                    self.path = path[i]
                
                algo.fit(horizon, experiment=self)
                
                if self.regret_flag:
                    regret[i] = np.cumsum(self.regret)
                if self.path_flag:
                    path[i] = self.path
            
            for k in self.stats:
                if k == 'mean':
                    self.statistics[algo_idx][k] = np.mean(regret, 0)
                elif k == 'std':
                    self.statistics[algo_idx][k] = np.std(regret, 0)
                elif k == 'pulls':
                    self.statistics[algo_idx][k] = path
                elif k == 'quantile':
                    self.statistics[algo_idx][k] = np.quantile(regret, q=self.quantile_ticks, axis=0)
                    
    def plot(self, save=False, complexity=True, dimensions=[(0,1)], functions=[]):
        ticks = np.arange(self.horizon)
        exp_info = f"Horizon = {self.horizon} - Nbr. of experiments = {self.nbr_exp}"
        if 'mean' in self.stats:
            plt.figure(figsize=(12,6))
            if 'std' in self.stats:
                title = "Mean cumulative regret and standard deviation tube\n"
                title += exp_info
                plt.title(title)
                for algo_stats in self.statistics.values():
                    name = algo_stats['name']
                    mean = algo_stats['mean']
                    std = algo_stats['std']
                    plt.plot(ticks, mean, label = name + f" - R = {mean[-1]:.2f}")
                    plt.fill_between(ticks, np.maximum(0, mean-std), mean+std, alpha=0.3)
                
            else:
                title = "Mean cumulative regret\n"
                title += exp_info
                plt.title(title)
                for algo_stats in self.statistics.values():
                        name = algo_stats['name']
                        mean = algo_stats['mean']
                        plt.plot(ticks, mean, label = name + f" - R = {mean[-1]:.2f}")
            
            if complexity and (self.complexity is not None):
                plt.plot(ticks, self.complexity*np.log(ticks+1), label="Regret lower Bound")
            
            if len(functions) > 0:
                for f, n in functions:
                    plt.plot(ticks, f(ticks), label=n)
            plt.legend()
            plt.show()
            if save:
                plt.savefig(f"./images/exp_mean_std_{date.now().strftime('%d_%m_%Y_%H_%M_%S')}",
                            bbox_inches='tight')
            plt.close()
            
        elif 'std' in self.stats:
            plt.figure(figsize=(12,6))
            title = "Standard deviation of the regret\n"
            title += exp_info
            plt.title(title)
            for algo_stats in self.statistics.values():
                name = algo_stats['name']
                std = algo_stats['std']
                plt.plot(ticks, std, label = name + f" - R = {std[-1]:.2f}")                
            if len(functions) > 0:
                for f, n in functions:
                    plt.plot(ticks, f(ticks), label=n)
            plt.legend()
            plt.show()
            if save:
                plt.savefig(f"./images/exp_std_{date.now().strftime('%d_%m_%Y_%H_%M_%S')}",
                            bbox_inches='tight')
            plt.close()
        
        if 'quantile' in self.stats:
            plt.figure(figsize=(12,6))
            title = f"Median cumulative regret and quantile {self.quantile_ticks[:-1]} tube\n"
            title += exp_info
            plt.title(title)
            for algo_stats in self.statistics.values():
                name = algo_stats['name']
                quantile = algo_stats['quantile']
                # By convention, the last quantile is the median
                plt.plot(ticks, quantile[-1], label = name + f" - R = {quantile[-1][-1]:.2f}")
                nbr_quantile = len(quantile)
                ptr_1 = 0
                ptr_2 = nbr_quantile-2
                while ptr_1 < ptr_2:
                    plt.fill_between(ticks, quantile[ptr_1], quantile[ptr_2], alpha=0.3)
                    ptr_1 += 1
                    ptr_2 -= 1
            if complexity and (self.complexity is not None):
                plt.plot(ticks, self.complexity*np.log(ticks+1), label="Regret lower Bound")
            if len(functions) > 0:
                for f, n in functions:
                    plt.plot(ticks, f(ticks), label=n)
            plt.legend()
            plt.show()
            if save:
                plt.savefig(f"./images/exp_quantile_{date.now().strftime('%d_%m_%Y_%H_%M_%S')}",
                            bbox_inches='tight')
            plt.close()
            
        if 'pulls' in self.stats:
            # basis = np.eye(self.bandit.nbr_arms, dtype=int)
            basis = np.eye(2, dtype=int)
            colors = list(mapcol.TABLEAU_COLORS.values())
            len_colors = len(colors)
            for dims in dimensions:
                plt.figure(figsize=(10,10))
                title = f"Sampling strategy as a random walk - abs.: arm {dims[0]}, ord.: arm {dims[1]}\n"
                title += exp_info
                plt.title(title)
                color_ctr = 0
                alpha = (self.bandit.nbr_arms+0.1*np.log(float(self.horizon)))/self.nbr_exp
                for algo_stats in self.statistics.values():
                    name = algo_stats['name']
                    paths = algo_stats['pulls']
                    for i, p in enumerate(paths):
                        path = np.zeros((2, self.horizon+1), int)
                        for t, arm in enumerate(p):
                            if arm == dims[0]:
                                path[:,t+1] = path[:,t] + basis[0]
                            elif arm == dims[1]:
                                path[:,t+1] = path[:,t] + basis[1]
                        if i == 0:
                            plt.plot(path[0], path[1], alpha=alpha, color=colors[color_ctr], label=f"{name}")
                        else:
                            plt.plot(path[0], path[1], alpha=alpha, color=colors[color_ctr])
                    color_ctr += 1
                    color_ctr = color_ctr % len_colors
                
                leg = plt.legend()
                for l in leg.get_lines():
                    l.set_alpha(1)
                plt.show()
                if save:
                    plt.savefig(f"./images/rw_dim_{dims[0]}_{dims[1]}_{date.now().strftime('%d_%m_%Y_%H_%M_%S')}",
                                bbox_inches='tight')
                plt.close()
