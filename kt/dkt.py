"""Deep Knowledge Tracing (Piech et al., 2015) in PyTorch."""
from __future__ import annotations

import numpy as np
import torch
from torch import nn


class DKT(nn.Module):
    def __init__(self, n_skills: int, hidden: int = 64, dropout: float = 0.2):
        super().__init__()
        self.n_skills = n_skills
        self.rnn = nn.LSTM(2 * n_skills, hidden, batch_first=True)
        self.drop = nn.Dropout(dropout)
        self.out = nn.Linear(hidden, n_skills)

    def forward(self, skills: torch.Tensor, correct: torch.Tensor) -> torch.Tensor:
        """Logits for P(correct on each skill) *after* seeing step t."""
        x = nn.functional.one_hot(skills + self.n_skills * correct, 2 * self.n_skills).float()
        h, _ = self.rnn(x)
        return self.out(self.drop(h))


def next_step_logits(model: DKT, skills: torch.Tensor, correct: torch.Tensor) -> torch.Tensor:
    """Logit for the question actually asked at t+1, given steps <= t."""
    logits = model(skills[:, :-1], correct[:, :-1])
    return logits.gather(2, skills[:, 1:].unsqueeze(-1)).squeeze(-1)


def train(model: DKT, skills: np.ndarray, correct: np.ndarray, epochs: int = 20,
          batch: int = 64, lr: float = 3e-3, seed: int = 0) -> list[float]:
    torch.manual_seed(seed)
    S, C = torch.as_tensor(skills), torch.as_tensor(correct)
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    loss_fn = nn.BCEWithLogitsLoss()
    history = []
    for _ in range(epochs):
        model.train()
        perm = torch.randperm(len(S))
        total = 0.0
        for i in range(0, len(S), batch):
            idx = perm[i:i + batch]
            logit = next_step_logits(model, S[idx], C[idx])
            loss = loss_fn(logit, C[idx][:, 1:].float())
            opt.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            total += loss.item() * len(idx)
        history.append(total / len(S))
    return history


@torch.no_grad()
def predict(model: DKT, skills: np.ndarray, correct: np.ndarray) -> np.ndarray:
    """P(correct) for steps 1..T-1 (step 0 has no history)."""
    model.eval()
    logit = next_step_logits(model, torch.as_tensor(skills), torch.as_tensor(correct))
    return torch.sigmoid(logit).numpy()
