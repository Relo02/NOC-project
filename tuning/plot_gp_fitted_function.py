#!/usr/bin/env python3
import json
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

REPO_ROOT = Path(__file__).resolve().parents[1]
HISTORY_FILE = REPO_ROOT / 'bag_gp_tuning' / 'gp_bag_tuning_history.json'
OUT_DIR = REPO_ROOT / 'bag_gp_tuning'
OUT_DIR.mkdir(parents=True, exist_ok=True)

# Matches gp_bag_mpc_cost_optimization notebook bounds
PARAM_BOUNDS = {
    'Q_x': (50.0, 200.0),
    'Q_y': (50.0, 200.0),
    'Q_yaw': (0.0, 15.0),
    'Q_terminal': (50.0, 300.0),
    'R_vx': (0.1, 10.0),
    'R_vy': (0.1, 10.0),
    'R_omega': (0.1, 10.0),
    'R_jerk': (0.1, 5.0),
    'W_obs_sigmoid': (1.0, 500.0),
    'obs_alpha': (0.1, 15.0),
    'obs_r': (0.05, 1.0),
}
PARAM_NAMES = list(PARAM_BOUNDS.keys())


def minmax_scale(X, lo, hi):
    denom = np.where((hi - lo) > 1e-12, (hi - lo), 1.0)
    return (X - lo) / denom


def _matern52_kernel(Xa, Xb, length_scale=0.35, sigma_f=1.0):
    diff = Xa[:, None, :] - Xb[None, :, :]
    r = np.sqrt(np.sum(diff * diff, axis=2) + 1e-12) / max(length_scale, 1e-9)
    sqrt5_r = np.sqrt(5.0) * r
    return sigma_f * sigma_f * (1.0 + sqrt5_r + (5.0 / 3.0) * r * r) * np.exp(-sqrt5_r)


def gp_fit_predict(X_train, y_train, X_test, length_scale=0.35, sigma_f=1.0, sigma_n=0.10):
    K = _matern52_kernel(X_train, X_train, length_scale=length_scale, sigma_f=sigma_f)
    K += (sigma_n ** 2 + 1e-8) * np.eye(K.shape[0])

    L = np.linalg.cholesky(K)
    alpha = np.linalg.solve(L.T, np.linalg.solve(L, y_train))

    Ks = _matern52_kernel(X_train, X_test, length_scale=length_scale, sigma_f=sigma_f)
    mu = Ks.T @ alpha

    v = np.linalg.solve(L, Ks)
    Kss_diag = np.diag(_matern52_kernel(X_test, X_test, length_scale=length_scale, sigma_f=sigma_f))
    var = np.maximum(Kss_diag - np.sum(v * v, axis=0), 1e-12)
    std = np.sqrt(var)
    return mu, std


def main():
    if not HISTORY_FILE.exists():
        raise FileNotFoundError(HISTORY_FILE)

    data = json.loads(HISTORY_FILE.read_text())
    hist = data['history']

    X = np.array([[row[p] for p in PARAM_NAMES] for row in hist], dtype=float)
    y = np.array([row['objective'] for row in hist], dtype=float)

    lo = np.array([PARAM_BOUNDS[p][0] for p in PARAM_NAMES], dtype=float)
    hi = np.array([PARAM_BOUNDS[p][1] for p in PARAM_NAMES], dtype=float)

    Xn = minmax_scale(X, lo, hi)
    best_idx = int(np.argmin(y))
    best_x = X[best_idx]

    chosen = [p for p in ['Q_x', 'Q_y', 'W_obs_sigmoid'] if p in PARAM_NAMES]
    if not chosen:
        chosen = PARAM_NAMES[:3]

    fig, axes = plt.subplots(1, len(chosen), figsize=(5.2 * len(chosen), 3.8), squeeze=False)
    axes = axes.ravel()

    for ax, pname in zip(axes, chosen):
        pidx = PARAM_NAMES.index(pname)
        p_lo, p_hi = PARAM_BOUNDS[pname]

        if pname in {'Q_x', 'Q_y', 'W_obs_sigmoid', 'Q_terminal'} and p_lo > 0:
            xs = np.logspace(np.log10(p_lo), np.log10(p_hi), 500)
            ax.set_xscale('log')
        else:
            xs = np.linspace(p_lo, p_hi, 500)

        X_probe = np.tile(best_x.copy(), (len(xs), 1))
        X_probe[:, pidx] = xs
        X_probe_n = minmax_scale(X_probe, lo, hi)

        mu, std = gp_fit_predict(Xn, y, X_probe_n, length_scale=0.35, sigma_f=1.0, sigma_n=0.10)
        low95 = mu - 1.96 * std
        up95 = mu + 1.96 * std

        ax.fill_between(xs, low95, up95, color='royalblue', alpha=0.18, label='95% CI')
        ax.plot(xs, mu, color='blue', lw=2.0, label='GP mean')

        ax.scatter(X[:, pidx], y, marker='x', s=26, c='black', linewidths=1.1, alpha=0.9, label='samples')
        ax.scatter([X[0, pidx]], [y[0]], marker='*', s=120, c='#d6c800', edgecolors='black', linewidths=0.5, label='first sample')
        ax.scatter([best_x[pidx]], [np.min(y)], marker='X', s=95, c='red', edgecolors='black', linewidths=0.5, label='best sample')

        ax.set_title(f'GP posterior over {pname}', fontsize=11)
        ax.set_xlabel(pname)
        ax.set_ylabel('objective (lower is better)')
        ax.grid(True, alpha=0.25)

    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc='upper center', ncol=5, frameon=False, fontsize=9)
    fig.suptitle('Fitted Gaussian-Process Surrogate from BO Samples', fontsize=13, y=1.04)
    fig.tight_layout(rect=[0, 0, 1, 0.92])

    out_pdf = OUT_DIR / 'gp_fitted_function.pdf'
    out_png = OUT_DIR / 'gp_fitted_function.png'
    fig.savefig(out_pdf, bbox_inches='tight')
    fig.savefig(out_png, dpi=300, bbox_inches='tight')
    plt.close(fig)

    print('Saved:', out_pdf)
    print('Saved:', out_png)
    print('Best objective:', float(np.min(y)))


if __name__ == '__main__':
    main()
