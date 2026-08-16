"""
GSE114134 Within Cohort Classification (infants, n=59) — L2 / ridge variant

Given effect sizes of 3-4 percentage points, L2 is arguably the better-matched
assumption. If this is also null, the result holds under both.

Stability tracking is disabled here: L2 never zeros a coefficient, so
"selection frequency" is meaningless.

Writes to *_L2 filenames so it will not overwrite the L1 results.

This script was scaffolded by an AI agent rather than written by me directly, but every step was manually verified by me.
"""

import warnings
warnings.filterwarnings('ignore')

import numpy as np
import pandas as pd
import json
import time
from sklearn.model_selection import RepeatedStratifiedKFold
from sklearn.feature_selection import f_classif
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score

data_dir = '/home/ethan-xiao/food-allergy-biomarkers/data'

VAR_Q   = 0.80
TOP_N   = 500
C_VAL   = 1.0
N_SPLIT = 5
N_REP   = 20
N_PERM  = 200
SEED    = 42


def fold_funnel(Xtr_full, ytr, Xte_full, var_q=VAR_Q, top_n=TOP_N, C=C_VAL):
    """Variance filter -> univariate rank -> L2 ridge -> fitted model.

    Xtr_full / Xte_full are probes x samples. Everything derives from
    training rows only.
    """
    v = Xtr_full.var(axis=1)
    keep = Xtr_full.index[v > v.quantile(var_q)]

    _, p = f_classif(Xtr_full.loc[keep].T.values, ytr)
    top = pd.Series(p, index=keep).sort_values().index[:top_n]

    scaler = StandardScaler().fit(Xtr_full.loc[top].T.values)
    Xtr = scaler.transform(Xtr_full.loc[top].T.values)

    # no penalty= argument -> sklearn default is L2
    clf = LogisticRegression(C=C, max_iter=5000).fit(Xtr, ytr)

    Xte = scaler.transform(Xte_full.loc[top].T.values)
    return clf, Xte


def cv_run(Xfull, yfull, seed=SEED, n_splits=N_SPLIT, n_repeats=N_REP,
           progress=None):
    """Repeated stratified CV. Returns per-fold AUCs."""
    cv = RepeatedStratifiedKFold(n_splits=n_splits, n_repeats=n_repeats,
                                 random_state=seed)
    aucs = []
    Xt = Xfull.T

    for i, (tr_idx, te_idx) in enumerate(cv.split(Xt.values, yfull)):
        Xtr_full = Xfull.iloc[:, tr_idx]
        Xte_full = Xfull.iloc[:, te_idx]
        ytr, yte = yfull[tr_idx], yfull[te_idx]

        clf, Xte = fold_funnel(Xtr_full, ytr, Xte_full)
        aucs.append(roc_auc_score(yte, clf.predict_proba(Xte)[:, 1]))

        if progress and i % progress == 0:
            print(f"    fold {i}/{n_splits * n_repeats}", flush=True)

    return np.array(aucs)


if __name__ == '__main__':
    raw_a = pd.read_csv(f'{data_dir}/raw_114134.csv', index_col=0)
    meta  = pd.read_csv(f'{data_dir}/sample_info.csv')

    is_114134 = (meta['cohort'] == 'GSE114134').values
    y_all = (meta['allergy_status'] == 'allergic').astype(int).values
    ya = y_all[is_114134]

    assert raw_a.shape[1] == len(ya), "column count != label count"
    print("=== L2 / ridge variant ===")
    print(f"GSE114134: {raw_a.shape[0]:,} probes x {raw_a.shape[1]} samples")
    print(f"  allergic={ya.sum()}, control={(1-ya).sum()}, "
          f"majority baseline={max(ya.mean(), 1-ya.mean()):.3f}")

    # --- timing probe ---
    t0 = time.time()
    cv_run(raw_a, ya, n_repeats=1)
    per_fold = (time.time() - t0) / N_SPLIT
    total = per_fold * (N_SPLIT * N_REP + N_PERM * N_SPLIT * 2)
    print(f"\n{per_fold:.2f} sec/fold -> ~{total/60:.0f} min total\n",
          flush=True)

    # --- observed ---
    print("Observed CV:", flush=True)
    aucs = cv_run(raw_a, ya, progress=20)
    print(f"\nCV AUC: {aucs.mean():.3f} (sd {aucs.std():.3f}, "
          f"{len(aucs)} folds)")

    # --- permutation null: full funnel, shuffled labels ---
    print(f"\nPermutation null ({N_PERM} perms):", flush=True)
    rng = np.random.default_rng(SEED)
    null = []
    for i in range(N_PERM):
        y_perm = rng.permutation(ya)
        a = cv_run(raw_a, y_perm, seed=SEED + i, n_repeats=2)
        null.append(a.mean())
        if i % 20 == 0:
            print(f"    perm {i}/{N_PERM}", flush=True)
    null = np.array(null)
    p_val = (null >= aucs.mean()).mean()

    print(f"\nNull mean: {null.mean():.3f} (sd {null.std():.3f})")
    print(f"p = {p_val:.4f}")

    # --- checkpoint ---
    np.save(f'{data_dir}/within_cohort_null_L2.npy', null)

    with open(f'{data_dir}/within_cohort_results_L2.json', 'w') as f:
        json.dump({
            'cohort': 'GSE114134', 'n': int(raw_a.shape[1]),
            'n_allergic': int(ya.sum()), 'n_control': int((1 - ya).sum()),
            'var_quantile': VAR_Q, 'top_n': TOP_N, 'C': C_VAL,
            'solver': 'lbfgs', 'penalty': 'l2',
            'cv': f'{N_SPLIT}-fold x {N_REP} repeats',
            'n_perm': N_PERM, 'perm_cv': f'{N_SPLIT}-fold x 2 repeats',
            'cv_auc_mean': float(aucs.mean()), 'cv_auc_sd': float(aucs.std()),
            'null_mean': float(null.mean()), 'null_sd': float(null.std()),
            'p': float(p_val),
            'note': 'L2 robustness check on L1 result (AUC 0.538, p 0.34)',
        }, f, indent=2)

    print("\nSaved to within_cohort_results_L2.json")
