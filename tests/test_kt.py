import numpy as np
import torch

from kt import bkt, dkt
from kt.simulate import random_params, simulate


def test_simulation_shapes():
    p = random_params(5, np.random.default_rng(0))
    S, C = simulate(20, 15, p, seed=0)
    assert S.shape == C.shape == (20, 15)
    assert S.min() >= 0 and S.max() < 5 and set(np.unique(C)) <= {0, 1}


def test_bkt_probabilities_valid_and_respond_to_answers():
    p = random_params(3, np.random.default_rng(1))
    S = np.zeros((2, 6), dtype=int)
    C = np.array([[1, 1, 1, 1, 1, 1], [0, 0, 0, 0, 0, 0]])
    pred = bkt.predict(S, C, p)
    assert ((pred > 0) & (pred < 1)).all()
    assert pred[0, -1] > pred[1, -1]  # a run of correct answers raises P(correct)


def test_dkt_shapes_and_no_future_leak():
    torch.manual_seed(0)
    m = dkt.DKT(4, hidden=8, dropout=0.0)
    S = torch.randint(0, 4, (3, 10)); C = torch.randint(0, 2, (3, 10))
    out1 = dkt.next_step_logits(m, S, C)
    C2 = C.clone(); C2[:, -1] = 1 - C2[:, -1]  # change only the last answer
    out2 = dkt.next_step_logits(m, S, C2)
    assert out1.shape == (3, 9)
    assert torch.allclose(out1, out2)  # the last answer is never used as input


def test_dkt_learns_on_simulated_data():
    p = random_params(4, np.random.default_rng(2))
    S, C = simulate(200, 20, p, seed=3)
    m = dkt.DKT(4, hidden=16)
    hist = dkt.train(m, S, C, epochs=5)
    assert hist[-1] < hist[0]
