"""Experiments for the PathoWatch ATLAS paper (classification component only).

Outputs JSON + CSV tables into the output directory. Deterministic (fixed seeds).
"""
import sys, json, time, os
import numpy as np, pandas as pd
from scipy import stats
from sklearn.preprocessing import OneHotEncoder, OrdinalEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
from sklearn.calibration import CalibratedClassifierCV
from sklearn.ensemble import RandomForestClassifier, HistGradientBoostingClassifier
from sklearn.metrics import roc_auc_score, average_precision_score, brier_score_loss, f1_score
from sklearn.inspection import permutation_importance

SEED = 42
src, outdir = sys.argv[1], sys.argv[2]
os.makedirs(outdir, exist_ok=True)
rng = np.random.default_rng(SEED)

d = pd.read_csv(src)
d['MDR'] = (d['cats_ns'] >= 3).astype(int)
fix = {'Genital/Urinary (GU)': 'GU', 'GastroIntestinal (GI)': 'GI', 'CardioVascular (CVS)': 'CVS',
       'Integumentary (INT)': 'INT', 'Central Nervous (CNS)': 'CNS', 'None Given': 'Unknown'}
d['SourceGroup'] = d['SourceGroup'].replace(fix)
for c in ['Gender', 'Age Group', 'Speciality', 'In / Out Patient', 'Country', 'SourceGroup']:
    d[c] = d[c].fillna('Unknown').astype(str)

# World Bank income classification, FY2019 (July 2018; based on 2017 GNI per capita)
LMIC = ['India', 'Pakistan', 'Philippines', 'Morocco', 'Tunisia', 'Kenya', 'Nigeria', 'Honduras',
        'El Salvador', 'Egypt', 'Vietnam', 'Indonesia', 'Nicaragua']
UMIC = ['Mexico', 'China', 'South Africa', 'Colombia', 'Russia', 'Venezuela', 'Turkey', 'Brazil',
        'Thailand', 'Romania', 'Malaysia', 'Guatemala', 'Bulgaria', 'Jordan', 'Serbia', 'Jamaica',
        'Lebanon', 'Namibia', 'Mauritius']
HIC = ['United States', 'Spain', 'France', 'Germany', 'Italy', 'Belgium', 'Portugal', 'Argentina',
       'Canada', 'Israel', 'Greece', 'Czech Republic', 'Hungary', 'United Kingdom', 'Australia',
       'Korea, South', 'Chile', 'Taiwan', 'Denmark', 'Poland', 'Netherlands', 'Sweden', 'Kuwait',
       'Ireland', 'Austria', 'Switzerland', 'Japan', 'Croatia', 'Finland', 'Panama', 'Hong Kong',
       'Singapore', 'Slovenia', 'Latvia', 'Saudi Arabia', 'Lithuania', 'Slovak Republic', 'Oman',
       'Norway', 'Puerto Rico', 'Estonia']
inc = {**{c: 'HIC' for c in HIC}, **{c: 'UMIC' for c in UMIC}, **{c: 'LMIC' for c in LMIC}}
missing = sorted(set(d['Country']) - set(inc))
d['Income'] = d['Country'].map(inc)
print('countries without income group:', missing, 'rows:', d['Income'].isna().sum())

CAT = ['Species', 'SourceGroup', 'Speciality', 'In / Out Patient', 'Age Group', 'Gender', 'Country']
NUM = ['Year']
train = d[d.Year <= 2014].copy(); test = d[d.Year >= 2015].copy()

def wilson(k, n, z=1.96):
    if n == 0: return (np.nan, np.nan)
    p = k / n; den = 1 + z * z / n
    c = (p + z * z / (2 * n)) / den; h = z * np.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / den
    return (c - h, c + h)

def boot_ci(y, p, fn, n=500):
    y = np.asarray(y); p = np.asarray(p); vals = []
    idx_all = np.arange(len(y))
    for _ in range(n):
        idx = rng.choice(idx_all, len(y), replace=True)
        if y[idx].min() == y[idx].max(): continue
        vals.append(fn(y[idx], p[idx]))
    return float(np.percentile(vals, 2.5)), float(np.percentile(vals, 97.5))

def make(model_name, cats, nums):
    if model_name in ('LR', 'LinearSVM', 'RF', 'Baseline'):
        pre = ColumnTransformer([('cat', OneHotEncoder(handle_unknown='ignore', min_frequency=20), cats),
                                 ('num', 'passthrough', nums)])
        # centre year so linear models are well conditioned
        if model_name in ('LR', 'Baseline'):
            clf = LogisticRegression(C=1.0, max_iter=3000)
        elif model_name == 'LinearSVM':
            clf = CalibratedClassifierCV(LinearSVC(C=0.1, max_iter=5000), method='sigmoid', cv=3)
        else:
            clf = RandomForestClassifier(n_estimators=300, min_samples_leaf=20, max_features='sqrt',
                                         n_jobs=2, random_state=SEED)
        return Pipeline([('pre', pre), ('clf', clf)])
    if model_name == 'GBM':
        pre = ColumnTransformer([('cat', OrdinalEncoder(handle_unknown='use_encoded_value', unknown_value=-1,
                                                        encoded_missing_value=-1), cats),
                                 ('num', 'passthrough', nums)])
        clf = HistGradientBoostingClassifier(learning_rate=0.05, max_iter=600, max_leaf_nodes=31,
                                             l2_regularization=1.0, early_stopping=True,
                                             validation_fraction=0.1, random_state=SEED,
                                             categorical_features=list(range(len(cats))))
        return Pipeline([('pre', pre), ('clf', clf)])

def prep(df):
    X = df[CAT + NUM].copy(); X['Year'] = X['Year'] - 2010
    return X

def best_threshold(y, p):
    ths = np.linspace(0.05, 0.95, 181)
    f1s = [f1_score(y, (p >= t).astype(int)) for t in ths]
    return float(ths[int(np.argmax(f1s))])

def evaluate(y, p, thr):
    yhat = (p >= thr).astype(int)
    tp = int(((yhat == 1) & (y == 1)).sum()); tn = int(((yhat == 0) & (y == 0)).sum())
    fp = int(((yhat == 1) & (y == 0)).sum()); fn = int(((yhat == 0) & (y == 1)).sum())
    r = dict(n=int(len(y)), prevalence=float(y.mean()), mean_pred=float(p.mean()),
             auroc=float(roc_auc_score(y, p)), auprc=float(average_precision_score(y, p)),
             brier=float(brier_score_loss(y, p)), f1=float(f1_score(y, yhat)),
             sens=tp / max(tp + fn, 1), spec=tn / max(tn + fp, 1), thr=thr)
    r['auroc_ci'] = boot_ci(y, p, roc_auc_score)
    r['auprc_ci'] = boot_ci(y, p, average_precision_score)
    return r

results = {'meta': {}}
y_tr, y_te = train['MDR'].values, test['MDR'].values
results['meta'].update(n_total=int(len(d)), n_train=int(len(train)), n_test=int(len(test)),
                       mdr_train=float(y_tr.mean()), mdr_test=float(y_te.mean()),
                       n_countries=int(d['Country'].nunique()))

# ---------------- Experiment 1: temporal validation, all models ----------------
exp1 = {}
inner_tr = train[train.Year <= 2013]; inner_va = train[train.Year == 2014]
specs = {'Baseline': (['Species'], ['Year']), 'LR': (CAT, NUM), 'LinearSVM': (CAT, NUM),
         'RF': (CAT, NUM), 'GBM': (CAT, NUM)}
preds = {}
for name, (cats, nums) in specs.items():
    t0 = time.time()
    Xi = prep(inner_tr)[cats + nums]; Xv = prep(inner_va)[cats + nums]
    m = make(name, cats, nums).fit(Xi, inner_tr['MDR'])
    thr = best_threshold(inner_va['MDR'].values, m.predict_proba(Xv)[:, 1])
    m = make(name, cats, nums).fit(prep(train)[cats + nums], y_tr)
    p = m.predict_proba(prep(test)[cats + nums])[:, 1]
    preds[name] = p
    exp1[name] = evaluate(y_te, p, thr)
    exp1[name]['fit_seconds'] = round(time.time() - t0, 1)
    if name == 'GBM': gbm_model = m
    if name == 'LR': lr_model = m
    print(name, {k: (round(v, 4) if isinstance(v, float) else v) for k, v in exp1[name].items()}, flush=True)
results['exp1_temporal'] = exp1

# per-species AUROC for GBM and LR
per_sp = {}
for sp in sorted(test['Species'].unique()):
    msk = (test['Species'] == sp).values
    per_sp[sp] = {mn: dict(n=int(msk.sum()), prevalence=float(y_te[msk].mean()),
                           auroc=float(roc_auc_score(y_te[msk], preds[mn][msk])),
                           auroc_ci=boot_ci(y_te[msk], preds[mn][msk], roc_auc_score, n=300))
                  for mn in ('LR', 'GBM')}
results['exp1_per_species'] = per_sp

# India subset of the temporal test set (pooled model)
msk = (test['Country'] == 'India').values
results['exp1_india'] = {mn: dict(n=int(msk.sum()), prevalence=float(y_te[msk].mean()),
                                  mean_pred=float(preds[mn][msk].mean()),
                                  auroc=float(roc_auc_score(y_te[msk], preds[mn][msk])),
                                  auroc_ci=boot_ci(y_te[msk], preds[mn][msk], roc_auc_score))
                         for mn in ('LR', 'GBM')}

# ---------------- Feature importance (GBM, permutation on test) ----------------
Xte = prep(test)[CAT + NUM]
pi = permutation_importance(gbm_model, Xte, y_te, scoring='roc_auc', n_repeats=5,
                            random_state=SEED, n_jobs=1)
results['importance'] = {f: dict(mean=float(pi.importances_mean[i]), sd=float(pi.importances_std[i]))
                         for i, f in enumerate(CAT + NUM)}

# LR odds ratios for selected categories
oh = lr_model.named_steps['pre']; coefs = lr_model.named_steps['clf'].coef_[0]
names = oh.get_feature_names_out()
orr = pd.Series(np.exp(coefs), index=names)
results['lr_odds_ratios'] = {k: float(v) for k, v in orr.items()}

# ---------------- Experiment 2: geographic transfer ----------------
CATg = [c for c in CAT if c != 'Country']
exp2 = {}
tests = {g: test[test['Income'] == g] for g in ('HIC', 'UMIC', 'LMIC')}
tests['India'] = test[test['Country'] == 'India']
for trname, trdf in {'HIC-trained': train[train['Income'] == 'HIC'], 'All-trained': train}.items():
    inner_tr = trdf[trdf.Year <= 2013]; inner_va = trdf[trdf.Year == 2014]
    m = make('GBM', CATg, NUM).fit(prep(inner_tr)[CATg + NUM], inner_tr['MDR'])
    thr = best_threshold(inner_va['MDR'].values, m.predict_proba(prep(inner_va)[CATg + NUM])[:, 1])
    m = make('GBM', CATg, NUM).fit(prep(trdf)[CATg + NUM], trdf['MDR'])
    exp2[trname] = {'n_train': int(len(trdf))}
    for g, tdf in tests.items():
        p = m.predict_proba(prep(tdf)[CATg + NUM])[:, 1]
        exp2[trname][g] = evaluate(tdf['MDR'].values, p, thr)
        print(trname, g, round(exp2[trname][g]['auroc'], 4), round(exp2[trname][g]['prevalence'], 3),
              round(exp2[trname][g]['mean_pred'], 3), flush=True)
results['exp2_transfer'] = exp2

# ---------------- Descriptive: MDR prevalence by year ----------------
def trend_test(sub):
    """Cochran-Armitage test for trend in proportions across years."""
    g = sub.groupby('Year')['MDR'].agg(['sum', 'size'])
    t = g.index.values.astype(float); k = g['sum'].values; n = g['size'].values
    N = n.sum(); pbar = k.sum() / N
    T = np.sum(t * (k - n * pbar))
    var = pbar * (1 - pbar) * (np.sum(n * t * t) - np.sum(n * t) ** 2 / N)
    z = T / np.sqrt(var)
    return float(z), float(2 * stats.norm.sf(abs(z)))

desc = {}
for sp in sorted(d['Species'].unique()):
    sub = d[d['Species'] == sp]
    yr = {}
    for y, g in sub.groupby('Year'):
        k, n = int(g['MDR'].sum()), int(len(g)); lo, hi = wilson(k, n)
        yr[int(y)] = dict(n=n, k=k, p=k / n, lo=lo, hi=hi)
    z, pv = trend_test(sub)
    byinc = {}
    for gname, g in sub.groupby('Income'):
        k, n = int(g['MDR'].sum()), int(len(g)); lo, hi = wilson(k, n)
        byinc[gname] = dict(n=n, k=k, p=k / n, lo=lo, hi=hi)
    ind = sub[sub['Country'] == 'India']
    k, n = int(ind['MDR'].sum()), int(len(ind)); lo, hi = wilson(k, n)
    carb = sub['CarbapenemNS'].dropna(); carb_ind = ind['CarbapenemNS'].dropna()
    desc[sp] = dict(n=int(len(sub)), mdr=float(sub['MDR'].mean()), by_year=yr, trend_z=z, trend_p=pv,
                    by_income=byinc, india=dict(n=n, k=k, p=k / n if n else np.nan, lo=lo, hi=hi),
                    carb_ns=float(carb.mean()), carb_ns_india=float(carb_ind.mean()) if len(carb_ind) else None,
                    cats_tested_median=float(sub['cats_tested'].median()))
results['descriptive'] = desc
allinc = {}
for gname, g in d.groupby('Income'):
    k, n = int(g['MDR'].sum()), int(len(g)); lo, hi = wilson(k, n)
    allinc[gname] = dict(n=n, k=k, p=k / n, lo=lo, hi=hi, countries=int(g['Country'].nunique()))
ind = d[d['Country'] == 'India']; k, n = int(ind['MDR'].sum()), int(len(ind)); lo, hi = wilson(k, n)
allinc['India'] = dict(n=n, k=k, p=k / n, lo=lo, hi=hi, years=sorted(map(int, ind['Year'].unique())))
results['descriptive_income'] = allinc
results['test_counts'] = {g: int(len(t)) for g, t in tests.items()}
results['study_counts'] = d['Study'].value_counts().to_dict()

json.dump(results, open(os.path.join(outdir, 'results.json'), 'w'), indent=1, default=float)
print('done')
