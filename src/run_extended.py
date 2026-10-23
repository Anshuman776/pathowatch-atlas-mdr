"""Extended analyses for the journal version of the PathoWatch ATLAS paper.

Sections: (B) temporal models + ROC + calibration + subgroups, (C) learning curve,
(D) sensitivity analyses, (E) transfer: calibration, recalibration variants,
leave-one-WHO-region-out, (F) funnel-plot hotspot analysis, (G) yearly forecasting.
"""
import sys, json, os, time
import numpy as np, pandas as pd
from scipy import stats
from scipy.special import expit, logit
from scipy.optimize import brentq
from sklearn.preprocessing import OneHotEncoder, OrdinalEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, HistGradientBoostingClassifier
from sklearn.metrics import roc_auc_score, roc_curve, brier_score_loss, average_precision_score

src, outdir = sys.argv[1], sys.argv[2]
SEED = 42
rng = np.random.default_rng(SEED)
exec(open(os.path.join(os.path.dirname(__file__), 'common_prep.py')).read())

WHO = {}
for c in ['South Africa', 'Nigeria', 'Kenya', 'Namibia', 'Mauritius']: WHO[c] = 'Africa'
for c in ['United States', 'Canada', 'Mexico', 'Guatemala', 'Honduras', 'El Salvador', 'Nicaragua', 'Panama',
          'Colombia', 'Venezuela', 'Brazil', 'Argentina', 'Chile', 'Jamaica', 'Puerto Rico']: WHO[c] = 'Americas'
for c in ['India', 'Thailand', 'Indonesia']: WHO[c] = 'South-East Asia'
for c in ['Pakistan', 'Egypt', 'Jordan', 'Kuwait', 'Lebanon', 'Morocco', 'Oman', 'Saudi Arabia', 'Tunisia']:
    WHO[c] = 'Eastern Mediterranean'
for c in ['Spain', 'France', 'Germany', 'Italy', 'Belgium', 'Portugal', 'Israel', 'Greece', 'Czech Republic',
          'Hungary', 'United Kingdom', 'Russia', 'Turkey', 'Denmark', 'Poland', 'Netherlands', 'Sweden', 'Ireland',
          'Austria', 'Switzerland', 'Romania', 'Croatia', 'Finland', 'Bulgaria', 'Slovenia', 'Latvia', 'Lithuania',
          'Slovak Republic', 'Serbia', 'Norway', 'Estonia']: WHO[c] = 'Europe'
for c in ['China', 'Australia', 'Korea, South', 'Taiwan', 'Philippines', 'Japan', 'Malaysia', 'Hong Kong',
          'Singapore', 'Vietnam']: WHO[c] = 'Western Pacific'
unm = sorted(set(d['Country']) - set(WHO)); assert not unm, unm
d['Region'] = d['Country'].map(WHO)

CAT = ['Species', 'SourceGroup', 'Speciality', 'In / Out Patient', 'Age Group', 'Gender', 'Country']
CATg = [c for c in CAT if c != 'Country']
NUM = ['Year']
R = {}

def X_(df, cats):
    X = df[cats + NUM].copy(); X['Year'] = X['Year'] - 2010; return X

def gbm(cats):
    pre = ColumnTransformer([('cat', OrdinalEncoder(handle_unknown='use_encoded_value', unknown_value=-1,
                                                    encoded_missing_value=-1), cats), ('num', 'passthrough', NUM)])
    clf = HistGradientBoostingClassifier(learning_rate=0.05, max_iter=600, max_leaf_nodes=31, l2_regularization=1.0,
                                         early_stopping=True, validation_fraction=0.1, random_state=SEED,
                                         categorical_features=list(range(len(cats))))
    return Pipeline([('pre', pre), ('clf', clf)])

def onehot_model(kind, cats):
    pre = ColumnTransformer([('cat', OneHotEncoder(handle_unknown='ignore', min_frequency=20), cats), ('num', 'passthrough', NUM)])
    clf = LogisticRegression(C=1.0, max_iter=3000) if kind == 'LR' else \
        RandomForestClassifier(n_estimators=300, min_samples_leaf=20, max_features='sqrt', n_jobs=2, random_state=SEED)
    return Pipeline([('pre', pre), ('clf', clf)])

def cal_metrics(y, p, bins=10):
    p = np.clip(p, 1e-6, 1 - 1e-6); lp = logit(p)
    a = brentq(lambda t: expit(lp + t).mean() - y.mean(), -15, 15)  # calibration-in-the-large (offset MLE)
    lr = LogisticRegression(penalty=None, max_iter=1000).fit(lp.reshape(-1, 1), y)
    slope = float(lr.coef_[0][0])
    edges = np.linspace(0, 1, bins + 1); idx = np.clip(np.digitize(p, edges) - 1, 0, bins - 1)
    ece = sum(abs(p[idx == b].mean() - y[idx == b].mean()) * (idx == b).mean() for b in range(bins) if (idx == b).any())
    return dict(cal_intercept=float(a), cal_slope=slope, ece=float(ece), mean_pred=float(p.mean()), observed=float(y.mean()),
                brier=float(brier_score_loss(y, p)), auroc=float(roc_auc_score(y, p)))

def reliability(y, p, q=10):
    qs = pd.qcut(p, q, labels=False, duplicates='drop')
    return [dict(pred=float(p[qs == k].mean()), obs=float(y[qs == k].mean()), n=int((qs == k).sum())) for k in sorted(np.unique(qs))]

def boot(y, p, fn, n=300):
    vals = []
    for _ in range(n):
        i = rng.integers(0, len(y), len(y))
        if y[i].min() != y[i].max(): vals.append(fn(y[i], p[i]))
    return [float(np.percentile(vals, 2.5)), float(np.percentile(vals, 97.5))]

train = d[d.Year <= 2014].copy(); test = d[d.Year >= 2015].copy()
ytr, yte = train['MDR'].values, test['MDR'].values
t0 = time.time()

# ---------------- B. temporal models: ROC, calibration, subgroups ----------------
preds = {}
m_base = Pipeline([('pre', ColumnTransformer([('cat', OneHotEncoder(handle_unknown='ignore'), ['Species']), ('num', 'passthrough', NUM)])),
                   ('clf', LogisticRegression(max_iter=2000))]).fit(X_(train, ['Species']), ytr)
preds['Baseline'] = m_base.predict_proba(X_(test, ['Species']))[:, 1]
preds['LR'] = onehot_model('LR', CAT).fit(X_(train, CAT), ytr).predict_proba(X_(test, CAT))[:, 1]
preds['RF'] = onehot_model('RF', CAT).fit(X_(train, CAT), ytr).predict_proba(X_(test, CAT))[:, 1]
g_main = gbm(CAT).fit(X_(train, CAT), ytr)
preds['GBM'] = g_main.predict_proba(X_(test, CAT))[:, 1]
grid = np.linspace(0, 1, 41)
R['roc'] = {}
for k, p in preds.items():
    fpr, tpr, _ = roc_curve(yte, p)
    R['roc'][k] = dict(auroc=float(roc_auc_score(yte, p)), tpr=[float(np.interp(g, fpr, tpr)) for g in grid])
R['roc_grid'] = [float(g) for g in grid]
R['calibration_temporal'] = {k: cal_metrics(yte, p) for k, p in preds.items() if k != 'Baseline'}
R['reliability_temporal_gbm'] = reliability(yte, preds['GBM'])
print('B done', round(time.time() - t0), flush=True)

pg = preds['GBM']; sub = {}
def grp_icu(s): return 'ICU' if 'ICU' in s else ('Unknown' if s in ('None Given', 'Other') else 'Non-ICU')
def grp_age(a):
    return {'0 to 2 Years': 'Children (0-18)', '3 to 12 Years': 'Children (0-18)', '13 to 18 Years': 'Children (0-18)',
            '19 to 64 Years': 'Adults (19-64)', '65 to 84 Years': 'Older adults (65+)', '85 and Over': 'Older adults (65+)'}.get(a, 'Unknown')
groupings = {
    'Care setting': test['Speciality'].map(grp_icu),
    'Patient type': test['In / Out Patient'],
    'Specimen': test['SourceGroup'].where(test['SourceGroup'].isin(['CVS', 'Respiratory', 'GU', 'INT', 'Bodily Fluids']), 'Other'),
    'Age': test['Age Group'].map(grp_age),
    'Gender': test['Gender'],
    'Income group': test['Income'],
    'Study': test['Study'],
    'WHO region': test['Region'],
}
for gname, series in groupings.items():
    sub[gname] = {}
    for lv in sorted(series.unique()):
        msk = (series == lv).values
        if msk.sum() < 200 or yte[msk].min() == yte[msk].max(): continue
        sub[gname][lv] = dict(n=int(msk.sum()), prevalence=float(yte[msk].mean()), auroc=float(roc_auc_score(yte[msk], pg[msk])),
                              ci=boot(yte[msk], pg[msk], roc_auc_score, n=200))
R['subgroups'] = sub
print('subgroups done', round(time.time() - t0), flush=True)

# ---------------- C. learning curve ----------------
lc = {}
Xtr_full = X_(train, CAT); Xte_full = X_(test, CAT)
for frac in [0.005, 0.01, 0.02, 0.05, 0.1, 0.25, 0.5, 1.0]:
    reps = 3 if frac < 0.25 else 1; vals = []
    for r in range(reps):
        idx = rng.choice(len(train), max(int(frac * len(train)), 200), replace=False) if frac < 1 else np.arange(len(train))
        m = gbm(CAT).fit(Xtr_full.iloc[idx], ytr[idx])
        vals.append(roc_auc_score(yte, m.predict_proba(Xte_full)[:, 1]))
    lc[str(frac)] = dict(n=int(max(int(frac * len(train)), 200)), auroc_mean=float(np.mean(vals)), auroc_sd=float(np.std(vals)))
R['learning_curve'] = lc
print('C done', round(time.time() - t0), flush=True)

# ---------------- D. sensitivity analyses ----------------
sens = {}
def fit_eval(trd, ted, cats, label):
    m = gbm(cats).fit(X_(trd, cats), trd['MDR'].values)
    p = m.predict_proba(X_(ted, cats))[:, 1]; y = ted['MDR'].values
    return dict(label=label, n_train=int(len(trd)), n_test=int(len(ted)), prevalence=float(y.mean()),
                auroc=float(roc_auc_score(y, p)), auprc=float(average_precision_score(y, p)), brier=float(brier_score_loss(y, p)))
sens['main'] = dict(label='Main analysis', n_train=int(len(train)), n_test=int(len(test)), prevalence=float(yte.mean()),
                    auroc=float(roc_auc_score(yte, pg)), auprc=float(average_precision_score(yte, pg)), brier=float(brier_score_loss(yte, pg)))
sens['no_country'] = fit_eval(train, test, CATg, 'Country variable removed')
sens['test_only'] = fit_eval(train[train.Study == 'TEST'], test[test.Study == 'TEST'], CAT, 'TEST study only')
strict = d.copy(); strict['MDR'] = (strict['cats_ns'] >= 4).astype(int)
sens['strict'] = fit_eval(strict[strict.Year <= 2014], strict[strict.Year >= 2015], CAT, 'Stricter label ($\\geq$4 categories)')
perm = rng.permutation(len(d)); cut = int(0.7 * len(d))
sens['random_split'] = fit_eval(d.iloc[perm[:cut]], d.iloc[perm[cut:]], CAT, 'Random 70/30 split (not temporal)')
R['sensitivity'] = sens
print('D done', round(time.time() - t0), flush=True)

# ---------------- E. transfer: calibration + recalibration variants + LORO ----------------
hic_tr = train[train.Income == 'HIC']
g_hic = gbm(CATg).fit(X_(hic_tr, CATg), hic_tr['MDR'].values)
def shift(lp, a): return expit(lp + a)
trans = {}
for gname in ['HIC', 'UMIC', 'LMIC', 'India']:
    ted = test[test.Country == 'India'] if gname == 'India' else test[test.Income == gname]
    rcd = train[train.Country == 'India'] if gname == 'India' else train[train.Income == gname]
    y = ted['MDR'].values; p = g_hic.predict_proba(X_(ted, CATg))[:, 1]
    res = {'none': cal_metrics(y, p)}
    if gname != 'HIC':
        last = rcd['Year'].max(); rc_recent = rcd[rcd.Year == last]
        lp_te = logit(np.clip(p, 1e-6, 1 - 1e-6))
        for tag, rc in [('all_years', rcd), ('recent_year', rc_recent)]:
            prc = g_hic.predict_proba(X_(rc, CATg))[:, 1]; lprc = logit(np.clip(prc, 1e-6, 1 - 1e-6)); yrc = rc['MDR'].values
            a = brentq(lambda t: expit(lprc + t).mean() - yrc.mean(), -15, 15)
            res[f'intercept_{tag}'] = cal_metrics(y, shift(lp_te, a))
            lr = LogisticRegression(penalty=None, max_iter=1000).fit(lprc.reshape(-1, 1), yrc)
            res[f'logistic_{tag}'] = cal_metrics(y, lr.predict_proba(lp_te.reshape(-1, 1))[:, 1])
            res[f'n_{tag}'] = int(len(rc)); res[f'years_{tag}'] = [int(rc.Year.min()), int(rc.Year.max())]
        res['reliability_before'] = reliability(y, p, 5)
        res['reliability_recent_logistic'] = reliability(y, lr.predict_proba(lp_te.reshape(-1, 1))[:, 1], 5)
    trans[gname] = res
R['transfer_calibration'] = trans
print('E1 done', round(time.time() - t0), flush=True)

loro = {}
g_all = gbm(CATg).fit(X_(train, CATg), ytr)
p_all = g_all.predict_proba(X_(test, CATg))[:, 1]
for reg in sorted(d['Region'].unique()):
    trd = train[train.Region != reg]; msk = (test.Region == reg).values
    if msk.sum() < 100: continue
    m = gbm(CATg).fit(X_(trd, CATg), trd['MDR'].values)
    ted = test[msk]; y = ted['MDR'].values; p = m.predict_proba(X_(ted, CATg))[:, 1]
    loro[reg] = dict(n_test=int(msk.sum()), n_train_heldout=int(len(trd)), held_out=cal_metrics(y, p),
                     included=cal_metrics(y, p_all[msk]))
    print(reg, round(loro[reg]['held_out']['auroc'], 3), round(loro[reg]['included']['auroc'], 3), flush=True)
R['loro'] = loro
print('E2 done', round(time.time() - t0), flush=True)

# ---------------- F. funnel-plot hotspot analysis (2015-2017) ----------------
def funnel(df, unit, min_n=30):
    p_s = df.groupby('Species')['MDR'].mean()
    df = df.assign(pexp=df['Species'].map(p_s))
    pbar = df['MDR'].mean(); rows = []
    for u, g in df.groupby(unit):
        if len(g) < min_n: continue
        O = g['MDR'].sum(); E = g['pexp'].sum(); V = (g['pexp'] * (1 - g['pexp'])).sum()
        z = (O - E) / np.sqrt(V)
        rows.append(dict(unit=u, n=int(len(g)), O=int(O), E=float(E), ratio=float(O / E), z=float(z),
                         crude=float(g['MDR'].mean()), income=g['Income'].iloc[0], region=g['Region'].iloc[0]))
    return dict(pbar=float(pbar), units=sorted(rows, key=lambda r: -r['z']))
R['funnel_country'] = funnel(test, 'Country')
st = pd.read_csv(sys.argv[3], skiprows=5, usecols=['Isolate Id', 'State'], dtype=str).dropna(subset=['Isolate Id'])
us = test[test.Country == 'United States'].copy(); us['Isolate Id'] = us['Isolate Id'].astype(str)
us = us.merge(st, on='Isolate Id', how='left').dropna(subset=['State'])
R['funnel_us_state'] = funnel(us, 'State')
R['funnel_note'] = 'species-standardised O/E with funnel limits; 2015-2017; units with >=30 isolates'
print('F done', round(time.time() - t0), flush=True)

# ---------------- G. forecasting yearly MDR prevalence per country x species ----------------
agg = d.groupby(['Country', 'Species', 'Year'])['MDR'].agg(['sum', 'size']).reset_index()
agg = agg[agg['size'] >= 20]; agg['p'] = agg['sum'] / agg['size']
fc_rows = []
def holt(y, alpha, beta):
    l, b = y[0], (y[1] - y[0]) if len(y) > 1 else 0.0
    for t in range(1, len(y)):
        l_new = alpha * y[t] + (1 - alpha) * (l + b); b = beta * (l_new - l) + (1 - beta) * b; l = l_new
    return l, b
def holt_fit(y):
    best = None
    for a in np.linspace(0.1, 0.9, 9):
        for bt in np.linspace(0.05, 0.5, 10):
            err = 0
            for t in range(2, len(y)):
                l, b = holt(y[:t], a, bt); err += (y[t] - (l + b)) ** 2
            if best is None or err < best[0]: best = (err, a, bt)
    return best[1], best[2]
series_used = 0
pooled_X, pooled_y = [], []
for (c, s), g in agg.groupby(['Country', 'Species']):
    tr_g = g[g.Year <= 2014].sort_values('Year'); te_g = g[g.Year >= 2015].sort_values('Year')
    if len(tr_g) < 6 or len(te_g) < 1 or tr_g['Year'].max() < 2012: continue
    series_used += 1
    yrs = tr_g['Year'].values.astype(float); ys = tr_g['p'].values
    a, bt = holt_fit(ys); l, b = holt(ys, a, bt)
    slope, icpt = np.polyfit(yrs, ys, 1)
    for _, row in te_g.iterrows():
        h = int(row['Year'] - tr_g['Year'].max())
        fc = {'Naive': ys[-1], 'Mean (last 3)': ys[-3:].mean(), 'Linear trend': icpt + slope * row['Year'],
              'Holt': l + h * b * 1.0}
        for k in fc: fc[k] = float(np.clip(fc[k], 0, 1))
        fc_rows.append(dict(country=c, species=s, year=int(row['Year']), h=h, obs=float(row['p']), n=int(row['size']), **fc))
fc = pd.DataFrame(fc_rows)
methods = ['Naive', 'Mean (last 3)', 'Linear trend', 'Holt']
fres = {'n_series': int(series_used), 'n_forecasts': int(len(fc))}
for mth in methods:
    err = (fc[mth] - fc['obs']).abs()
    fres[mth] = dict(mae=float(100 * err.mean()), mae_ci=[float(100 * x) for x in np.percentile(
        [err.sample(frac=1, replace=True, random_state=i).mean() for i in range(300)], [2.5, 97.5])],
        by_h={int(h): float(100 * e.mean()) for h, e in err.groupby(fc['h'])},
        by_species={s: float(100 * e.mean()) for s, e in err.groupby(fc['species'])})
fres['obs_change_sd'] = float(100 * (fc['obs'] - fc['Naive']).std())
R['forecasting'] = fres
print('G done', round(time.time() - t0), json.dumps({m: round(fres[m]['mae'], 2) for m in methods}), flush=True)

json.dump(R, open(os.path.join(outdir, 'results_ext.json'), 'w'), indent=1)
print('done', round(time.time() - t0))
