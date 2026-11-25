"""Simulate students answering questions with Bayesian Knowledge Tracing.

Each skill k has four BKT parameters:
    p_init  : probability the student already knows the skill
    p_learn : probability of learning it after each practice attempt
    p_guess : probability of answering correctly without knowing it
    p_slip  : probability of answering wrongly despite knowing it
Questions follow a loose curriculum: early questions favour easy skills.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class BKTParams:
    p_init: np.ndarray
    p_learn: np.ndarray
    p_guess: np.ndarray
    p_slip: np.ndarray

    @property
    def n_skills(self) -> int:
        return len(self.p_init)


def random_params(n_skills: int, rng: np.random.Generator) -> BKTParams:
    return BKTParams(
        p_init=rng.uniform(0.05, 0.4, n_skills),
        p_learn=rng.uniform(0.05, 0.3, n_skills),
        p_guess=rng.uniform(0.1, 0.3, n_skills),
        p_slip=rng.uniform(0.05, 0.15, n_skills),
    )


def simulate(n_students: int, seq_len: int, params: BKTParams, seed: int = 0):
    """Return arrays skills[n, T] (int) and correct[n, T] (0/1)."""
    rng = np.random.default_rng(seed)
    K = params.n_skills
    skills = np.zeros((n_students, seq_len), dtype=np.int64)
    correct = np.zeros((n_students, seq_len), dtype=np.int64)
    for s in range(n_students):
        known = rng.random(K) < params.p_init
        ability = rng.normal(0, 0.5)  # some students learn faster
        for t in range(seq_len):
            centre = (t / seq_len) * (K - 1)  # curriculum moves through skills
            w = np.exp(-0.5 * ((np.arange(K) - centre) / 2.0) ** 2)
            k = rng.choice(K, p=w / w.sum())
            p_correct = (1 - params.p_slip[k]) if known[k] else params.p_guess[k]
            c = rng.random() < p_correct
            skills[s, t], correct[s, t] = k, int(c)
            p_learn = np.clip(params.p_learn[k] * np.exp(ability), 0, 0.9)
            if not known[k] and rng.random() < p_learn:
                known[k] = True
    return skills, correct
