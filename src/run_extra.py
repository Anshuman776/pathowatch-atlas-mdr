"""Extra analyses: adjusted odds ratios (unpenalised logistic regression with Wald CIs),
country-adjusted yearly trend per species, and intercept recalibration of the
HIC-trained model for LMIC / India."""
import sys, json, os
import numpy as np, pandas as pd
from scipy import stats
from scipy.special import expit, logit
from scipy.optimize import brentq
from sklearn.preprocessing import OrdinalEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import roc_auc_score, brier_score_loss

src, outdir = sys.argv[1], sys.argv[2]
SEED = 42
exec(open(os.path.join(os.path.dirname(__file__), 'common_prep.py')).read())  # defines d, inc, prep helpers

def irls(X, y, iters=50, tol=1e-8):
    beta = np.zeros(X.shape[1])
    for _ in range(iters):
        eta = X @ beta; p = expit(eta); w = p * (1 - p)
        H = X.T @ (X * w[:, None]) + 1e-8 * np.eye(X.shape[1])
        g = X.T @ (y - p)
        step = np.linalg.solve(H, g); beta += step
        if np.max(np.abs(step)) < tol: break
    cov = np.linalg.inv(H)
    return beta, np.sqrt(np.diag(cov))

def design(df, cats, refs, add_year=False):
    cols, names = [np.ones(len(df))], ['Intercept']
    for c in cats:
        levels = sorted(df[c].unique())
        for lv in levels:
            if lv == refs[c]: continue
            cnt = (df[c] == lv).sum()
            if cnt < 30: continue
            cols.append((df[c] == lv).astype(float).values); names.append(f'{c}={lv}')
    if add_year:
        cols.append((df['Year'] - 2010).astype(float).values); names.append('Year')
    return np.column_stack(cols), names

res = {}
# ---- adjusted odds ratios, all isolates ----
cats = ['Species', 'SourceGroup', 'Speciality', 'In / Out Patient', 'Age Group', 'Gender', 'Country']
refs = {'Species': 'Escherichia coli', 'SourceGroup': 'GU', 'Speciality': 'Medicine General',
        'In / Out Patient': 'Inpatient', 'Age Group': '19 to 64 Years', 'Gender': 'Male', 'Country': 'United States'}
X, names = design(d, cats, refs, add_year=True)
beta, se = irls(X, d['MDR'].values.astype(float))
z = stats.norm.ppf(0.975)
res['adjusted_or'] = {n: dict(or_=float(np.exp(b)), lo=float(np.exp(b - z * s)), hi=float(np.exp(b + z * s)),
                              p=float(2 * stats.norm.sf(abs(b / s))))
                      for n, b, s in zip(names, beta, se) if not n.startswith('Country=')}
res['adjusted_or_india'] = {n: dict(or_=float(np.exp(b)), lo=float(np.exp(b - z * s)), hi=float(np.exp(b + z * s)))
                            for n, b, s in zip(names, beta, se) if n == 'Country=India'}
res['or_refs'] = refs

# ---- country-adjusted yearly trend per species ----
trend = {}
for sp in sorted(d['Species'].unique()):
    sub = d[d['Species'] == sp]
    Xs, nm = design(sub, ['Country'], {'Country': 'United States'}, add_year=True)
    b, s = irls(Xs, sub['MDR'].values.astype(float))
    i = nm.index('Year')
    trend[sp] = dict(or_per_year=float(np.exp(b[i])), lo=float(np.exp(b[i] - z * s[i])), hi=float(np.exp(b[i] + z * s[i])))
res['adjusted_trend'] = trend

# ---- year x study composition ----
res['study_by_year'] = pd.crosstab(d['Year'], d['Study']).to_dict()

# ---- intercept recalibration of HIC-trained GBM ----
CATg = ['Species', 'SourceGroup', 'Speciality', 'In / Out Patient', 'Age Group', 'Gender']
def gbm():
    pre = ColumnTransformer([('cat', OrdinalEncoder(handle_unknown='use_encoded_value', unknown_value=-1,
                                                    encoded_missing_value=-1), CATg), ('num', 'passthrough', ['Year'])])
    clf = HistGradientBoostingClassifier(learning_rate=0.05, max_iter=600, max_leaf_nodes=31, l2_regularization=1.0,
                                         early_stopping=True, validation_fraction=0.1, random_state=SEED,
                                         categorical_features=list(range(len(CATg))))
    return Pipeline([('pre', pre), ('clf', clf)])
def X_(df):
    X = df[CATg + ['Year']].copy(); X['Year'] = X['Year'] - 2010; return X
train = d[d.Year <= 2014]; test = d[d.Year >= 2015]
m = gbm().fit(X_(train[train.Income == 'HIC']), train[train.Income == 'HIC']['MDR'])
def shift_for(p, target):
    lp = logit(np.clip(p, 1e-6, 1 - 1e-6))
    return brentq(lambda dlt: expit(lp + dlt).mean() - target, -10, 10)
recal = {}
for name, rc_df, te_df in [('LMIC', train[train.Income == 'LMIC'], test[test.Income == 'LMIC']),
                           ('UMIC', train[train.Income == 'UMIC'], test[test.Income == 'UMIC']),
                           ('India', train[train.Country == 'India'], test[test.Country == 'India'])]:
    p_rc = m.predict_proba(X_(rc_df))[:, 1]
    delta = shift_for(p_rc, rc_df['MDR'].mean())
    p_te = m.predict_proba(X_(te_df))[:, 1]
    p_new = expit(logit(np.clip(p_te, 1e-6, 1 - 1e-6)) + delta)
    y = te_df['MDR'].values
    recal[name] = dict(n_recal=int(len(rc_df)), recal_years=sorted(map(int, rc_df['Year'].unique())),
                       n_test=int(len(te_df)), observed=float(y.mean()),
                       pred_before=float(p_te.mean()), pred_after=float(p_new.mean()),
                       brier_before=float(brier_score_loss(y, p_te)), brier_after=float(brier_score_loss(y, p_new)),
                       auroc=float(roc_auc_score(y, p_te)), delta_logit=float(delta))
res['recalibration'] = recal

# reliability (deciles) for HIC-trained model on HIC and LMIC test sets
rel = {}
for name, te_df in [('HIC', test[test.Income == 'HIC']), ('LMIC', test[test.Income == 'LMIC'])]:
    p = m.predict_proba(X_(te_df))[:, 1]; y = te_df['MDR'].values
    q = pd.qcut(p, 5, labels=False, duplicates='drop')
    rel[name] = [dict(mean_pred=float(p[q == k].mean()), observed=float(y[q == k].mean()), n=int((q == k).sum()))
                 for k in sorted(np.unique(q))]
res['reliability'] = rel
json.dump(res, open(os.path.join(outdir, 'results_extra.json'), 'w'), indent=1)
print(json.dumps({k: v for k, v in res.items() if k in ('adjusted_trend', 'recalibration', 'adjusted_or_india')}, indent=1))
for k, v in res['adjusted_or'].items():
    print(f"{k:45s} {v['or_']:.2f} ({v['lo']:.2f}-{v['hi']:.2f})")
print(res['reliability'])
