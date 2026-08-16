
import warnings
warnings.filterwarnings('ignore', category=FutureWarning)
warnings.filterwarnings('ignore', category=UserWarning)
"""
GSE114134 within cohort signature discovery (infants, n=59)

Everything runs on uncorrected M-values.
Nothing is computed on full cohort before splitting.

This script was scaffolded by an AI agent rather than written by me directly, but every step was manually verified by me.
"""

import numpy as np
import pandas as pd
import json
from collections import Counter
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
SEED    = 42


def fold_funnel(Xtr_full, ytr, Xte_full, var_q=VAR_Q, top_n=TOP_N, C=C_VAL):
    """Variance filter -> univariate rank -> L1 -> fitted model.

    Xtr_full / Xte_full are probes x samples. Everything derives from
    training rows only.
    """
    v = Xtr_full.var(axis=1)
    keep = Xtr_full.index[v > v.quantile(var_q)]

    _, p = f_classif(Xtr_full.loc[keep].T.values, ytr)
    top = pd.Series(p, index=keep).sort_values().index[:top_n]

    scaler = StandardScaler().fit(Xtr_full.loc[top].T.values)
    Xtr = scaler.transform(Xtr_full.loc[top].T.values)

    clf = LogisticRegression(penalty='l1', C=C, solver='liblinear',
                             max_iter=5000).fit(Xtr, ytr)

    selected = np.array(top)[clf.coef_[0] != 0]
    Xte = scaler.transform(Xte_full.loc[top].T.values)
    return clf, Xte, selected


def cv_run(Xfull, yfull, seed=SEED, n_splits=N_SPLIT, n_repeats=N_REP):
    """Repeated stratified CV. Returns per-fold AUCs and selection counts."""
    cv = RepeatedStratifiedKFold(n_splits=n_splits, n_repeats=n_repeats,
                                 random_state=seed)
    aucs, counts = [], Counter()
    Xt = Xfull.T  # samples x probes, for indexing by sample

    for tr_idx, te_idx in cv.split(Xt.values, yfull):
        Xtr_full = Xfull.iloc[:, tr_idx]
        Xte_full = Xfull.iloc[:, te_idx]
        ytr, yte = yfull[tr_idx], yfull[te_idx]

        clf, Xte, selected = fold_funnel(Xtr_full, ytr, Xte_full)
        aucs.append(roc_auc_score(yte, clf.predict_proba(Xte)[:, 1]))
        counts.update(selected)

    n_folds = n_splits * n_repeats
    freq = pd.Series(counts, dtype=float).sort_values(ascending=False) / n_folds
    return np.array(aucs), freq


if __name__ == '__main__':
    raw_a = pd.read_csv(f'{data_dir}/raw_114134.csv', index_col=0)
    meta  = pd.read_csv(f'{data_dir}/sample_info.csv')

    is_114134 = (meta['cohort'] == 'GSE114134').values
    y_all = (meta['allergy_status'] == 'allergic').astype(int).values
    ya = y_all[is_114134]

    assert raw_a.shape[1] == len(ya), "column count != label count"
    print(f"GSE114134: {raw_a.shape[0]:,} probes x {raw_a.shape[1]} samples")
    print(f"  allergic={ya.sum()}, control={(1-ya).sum()}, "
          f"majority baseline={max(ya.mean(), 1-ya.mean()):.3f}")

    # observed
    aucs, freq = cv_run(raw_a, ya)
    print(f"\nCV AUC: {aucs.mean():.3f} (sd {aucs.std():.3f}, "
          f"{len(aucs)} folds)")
    print(f"Probes ever selected: {len(freq)}")
    print(f"Max selection frequency: {freq.max():.3f}")
    print("\nTop 30 by selection frequency:")
    print(freq.head(30))

    # permutation null: full funnel, shuffled labels
    N_PERM = 200
    rng = np.random.default_rng(SEED)
    null = []
    for i in range(N_PERM):
        y_perm = rng.permutation(ya)
        a, _ = cv_run(raw_a, y_perm, seed=SEED + i, n_repeats=2)
        null.append(a.mean())
        if i % 20 == 0:
            print(f"  perm {i}")
    null = np.array(null)
    p_val = (null >= aucs.mean()).mean()

    print(f"\nNull mean: {null.mean():.3f} (sd {null.std():.3f})")
    print(f"p = {p_val:.4f}")

    # checkpoint
    np.save(f'{data_dir}/within_cohort_null.npy', null)
    freq.to_csv(f'{data_dir}/within_cohort_selection_freq.csv',
                header=['selection_frequency'])

    SIG_N = 20
    signature = freq.head(SIG_N)
    signature.to_csv(f'{data_dir}/signature_{SIG_N}cpg.csv',
                     header=['selection_frequency'])

    with open(f'{data_dir}/within_cohort_results.json', 'w') as f:
        json.dump({
            'cohort': 'GSE114134', 'n': int(raw_a.shape[1]),
            'n_allergic': int(ya.sum()), 'n_control': int((1 - ya).sum()),
            'var_quantile': VAR_Q, 'top_n': TOP_N, 'C': C_VAL,
            'cv': f'{N_SPLIT}-fold x {N_REP} repeats',
            'cv_auc_mean': float(aucs.mean()), 'cv_auc_sd': float(aucs.std()),
            'null_mean': float(null.mean()), 'null_sd': float(null.std()),
            'p': float(p_val),
            'n_probes_ever_selected': int(len(freq)),
            'max_selection_freq': float(freq.max()),
            'signature_size': SIG_N,
            'signature': signature.index.tolist(),
        }, f, indent=2)

    print(f"\nSaved. Signature of {SIG_N} CpGs in signature_{SIG_N}cpg.csv")
