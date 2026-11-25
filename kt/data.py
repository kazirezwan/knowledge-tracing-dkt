"""Load real interaction logs (e.g. ASSISTments 2009) into fixed-length sequences."""
from __future__ import annotations

import numpy as np
import pandas as pd


def load_interactions(path: str, user="user_id", skill="skill_id", correct="correct",
                      order="order_id", seq_len: int = 50, min_len: int = 5):
    """Return (skills, correct, n_skills) with sequences cut into windows of seq_len."""
    df = pd.read_csv(path, usecols=[user, skill, correct, order]).dropna()
    df[skill] = df[skill].astype("category").cat.codes
    df = df.sort_values([user, order])
    S, C = [], []
    for _, g in df.groupby(user):
        s, c = g[skill].to_numpy(), g[correct].astype(int).to_numpy()
        for i in range(0, len(s), seq_len):
            if len(s[i:i + seq_len]) == seq_len:
                S.append(s[i:i + seq_len]); C.append(c[i:i + seq_len])
    return np.array(S), np.array(C), int(df[skill].max()) + 1
