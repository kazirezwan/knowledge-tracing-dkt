"""Bayesian Knowledge Tracing inference and a simple grid-search fit."""
from __future__ import annotations

import itertools

import numpy as np

from .simulate import BKTParams


def predict(skills: np.ndarray, correct: np.ndarray, params: BKTParams) -> np.ndarray:
    """P(correct) for each step, using only the student's earlier answers."""
    n, T = skills.shape
    out = np.zeros((n, T))
    for s in range(n):
        p_know = params.p_init.copy()
        for t in range(T):
            k, c = skills[s, t], correct[s, t]
            g, sl, l = params.p_guess[k], params.p_slip[k], params.p_learn[k]
            out[s, t] = p_know[k] * (1 - sl) + (1 - p_know[k]) * g
            if c:
                post = p_know[k] * (1 - sl) / (p_know[k] * (1 - sl) + (1 - p_know[k]) * g)
            else:
                post = p_know[k] * sl / (p_know[k] * sl + (1 - p_know[k]) * (1 - g))
            p_know[k] = post + (1 - post) * l
    return out


def fit_grid(skills: np.ndarray, correct: np.ndarray, n_skills: int) -> BKTParams:
    """Fit each skill independently by maximising log-likelihood over a grid."""
    grid = dict(p_init=[0.05, 0.15, 0.3, 0.5], p_learn=[0.05, 0.1, 0.2, 0.3],
                p_guess=[0.1, 0.2, 0.3], p_slip=[0.05, 0.1, 0.2])
    best = {key: np.zeros(n_skills) for key in grid}
    for k in range(n_skills):
        seqs = [correct[s][skills[s] == k] for s in range(len(skills))]
        seqs = [q for q in seqs if len(q)]
        best_ll = -np.inf
        for pi, pl, pg, ps in itertools.product(*grid.values()):
            ll = 0.0
            for q in seqs:
                pk = pi
                for c in q:
                    pc = np.clip(pk * (1 - ps) + (1 - pk) * pg, 1e-6, 1 - 1e-6)
                    ll += np.log(pc if c else 1 - pc)
                    post = pk * (1 - ps) / pc if c else pk * ps / (1 - pc)
                    pk = post + (1 - post) * pl
            if ll > best_ll:
                best_ll, choice = ll, (pi, pl, pg, ps)
        for key, v in zip(grid, choice):
            best[key][k] = v
    return BKTParams(**best)
