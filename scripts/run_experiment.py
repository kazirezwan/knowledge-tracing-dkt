"""Compare knowledge-tracing models on simulated students.

    python scripts/run_experiment.py
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
import matplotlib  # noqa: E402
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from sklearn.metrics import accuracy_score, roc_auc_score  # noqa: E402

from kt import bkt, dkt  # noqa: E402
from kt.simulate import random_params, simulate  # noqa: E402

K, T, N_TRAIN, N_TEST = 12, 60, 2400, 600


def evaluate(p: np.ndarray, y: np.ndarray) -> dict:
    p, y = p.ravel(), y.ravel()
    return dict(auc=round(float(roc_auc_score(y, p)), 4), acc=round(float(accuracy_score(y, p > 0.5)), 4))


def main():
    rng = np.random.default_rng(42)
    true = random_params(K, rng)
    S_tr, C_tr = simulate(N_TRAIN, T, true, seed=1)
    S_te, C_te = simulate(N_TEST, T, true, seed=2)
    y = C_te[:, 1:]  # every model predicts steps 1..T-1
    results, t0 = {}, time.time()

    results["global_mean"] = evaluate(np.full(y.shape, C_tr.mean()), y)
    skill_rate = np.array([C_tr[S_tr == k].mean() for k in range(K)])
    results["per_skill_rate"] = evaluate(skill_rate[S_te[:, 1:]], y)

    # each student's own past accuracy on the skill, smoothed towards the skill rate
    p_hist = np.zeros(y.shape)
    for s in range(N_TEST):
        hits, tries = np.zeros(K), np.zeros(K)
        for t in range(1, T):
            k_prev, k = S_te[s, t - 1], S_te[s, t]
            hits[k_prev] += C_te[s, t - 1]; tries[k_prev] += 1
            p_hist[s, t - 1] = (hits[k] + 2 * skill_rate[k]) / (tries[k] + 2)
    results["student_skill_history"] = evaluate(p_hist, y)

    fitted = bkt.fit_grid(S_tr[:600], C_tr[:600], K)
    results["bkt_fitted"] = evaluate(bkt.predict(S_te, C_te, fitted)[:, 1:], y)
    results["bkt_true_params (oracle)"] = evaluate(bkt.predict(S_te, C_te, true)[:, 1:], y)
    print(f"baselines done ({time.time() - t0:.0f}s)")

    model = dkt.DKT(K, hidden=64)
    hist = dkt.train(model, S_tr, C_tr, epochs=25)
    results["dkt_lstm"] = evaluate(dkt.predict(model, S_te, C_te), y)
    print(f"dkt done ({time.time() - t0:.0f}s)")

    out = ROOT / "results"
    out.mkdir(exist_ok=True)
    meta = dict(n_skills=K, seq_len=T, n_train=N_TRAIN, n_test=N_TEST,
                test_correct_rate=round(float(y.mean()), 4))
    (out / "results.json").write_text(json.dumps(dict(setup=meta, test=results), indent=2))

    fig, ax = plt.subplots(1, 2, figsize=(11, 4))
    names = list(results)
    ax[0].barh(names, [results[n]["auc"] for n in names], color="#2e86c1")
    ax[0].set_xlim(0.5, max(r["auc"] for r in results.values()) + 0.03)
    ax[0].set_xlabel("next-step ROC-AUC (test students)")
    ax[1].plot(range(1, len(hist) + 1), hist, marker="o")
    ax[1].set_xlabel("epoch"); ax[1].set_ylabel("training BCE loss"); ax[1].set_title("DKT training")
    fig.tight_layout(); fig.savefig(out / "comparison.png", dpi=150)
    print(json.dumps(dict(setup=meta, test=results), indent=2))


if __name__ == "__main__":
    main()
