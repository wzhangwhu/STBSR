import copy
import numpy as np

class Plateau:
    """Same original thresholds, with honest component names; no epoch selection."""
    def __init__(self, rule, weights):
        self.rule, self.weights = copy.deepcopy(rule), copy.deepcopy(weights)
        self.checks, self.passed, self.converged = [], 0, False

    def __call__(self, history):
        n, c = len(history), self.rule
        if n < c['minimum_epoch'] or n % c['check_every']:
            return False
        a, b = history[-2*c['window']:-c['window']], history[-c['window']:]
        assert len(a) == len(b) == c['window']
        mean = lambda rows, key: float(np.mean([r[key] for r in rows]))
        total = max(abs(mean(b, 'Loss')), 1e-12)
        change = abs(mean(a, 'Loss') - mean(b, 'Loss')) / total
        cv = float(np.std([r['Loss'] for r in b])) / total
        comp = sum(self.weights[k] * abs(mean(a, col) - mean(b, col))
                   for k, col in [('rec', 'Reconstruction'), ('pair', 'Pair'), ('ccr', 'CCR')]) / total
        good = change <= c['relative_mean_change'] and cv <= c['relative_std'] and comp <= c['component_change']
        self.passed = self.passed + 1 if good else 0
        self.converged = self.passed >= c['consecutive_checks']
        self.checks.append(dict(epoch=n, relative_mean_change=change, relative_std=cv,
                                weighted_component_change=comp, passed=bool(good), consecutive=self.passed))
        return self.converged
