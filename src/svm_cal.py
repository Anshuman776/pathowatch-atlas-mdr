import sys, os, json, numpy as np, pandas as pd
from scipy.special import expit, logit
from scipy.optimize import brentq
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.svm import LinearSVC
from sklearn.calibration import CalibratedClassifierCV
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score, brier_score_loss
src, outdir = sys.argv[1], sys.argv[2]
exec(open(os.path.join(os.path.dirname(__file__), 'common_prep.py')).read())
CAT = ['Species', 'SourceGroup', 'Speciality', 'In / Out Patient', 'Age Group', 'Gender', 'Country']
def X_(df, cats): X = df[cats + ['Year']].copy(); X['Year'] -= 2010; return X
tr, te = d[d.Year <= 2014], d[d.Year >= 2015]
def cal(y, p):
    p = np.clip(p, 1e-6, 1 - 1e-6); lp = logit(p)
    a = brentq(lambda t: expit(lp + t).mean() - y.mean(), -15, 15)
    s = float(LogisticRegression(C=np.inf, max_iter=1000).fit(lp.reshape(-1, 1), y).coef_[0][0])
    e = np.linspace(0, 1, 11); i = np.clip(np.digitize(p, e) - 1, 0, 9)
    ece = sum(abs(p[i == b].mean() - y[i == b].mean()) * (i == b).mean() for b in range(10) if (i == b).any())
    return dict(cal_intercept=float(a), cal_slope=s, ece=float(ece), auroc=float(roc_auc_score(y, p)), brier=float(brier_score_loss(y, p)))
out = {}
svm = Pipeline([('pre', ColumnTransformer([('cat', OneHotEncoder(handle_unknown='ignore', min_frequency=20), CAT), ('num', 'passthrough', ['Year'])])),
                ('clf', CalibratedClassifierCV(LinearSVC(C=0.1, max_iter=5000), method='sigmoid', cv=3))]).fit(X_(tr, CAT), tr.MDR)
out['LinearSVM'] = cal(te.MDR.values, svm.predict_proba(X_(te, CAT))[:, 1])
base = Pipeline([('pre', ColumnTransformer([('cat', OneHotEncoder(handle_unknown='ignore'), ['Species']), ('num', 'passthrough', ['Year'])])),
                 ('clf', LogisticRegression(max_iter=2000))]).fit(X_(tr, ['Species']), tr.MDR)
out['Baseline'] = cal(te.MDR.values, base.predict_proba(X_(te, ['Species']))[:, 1])
json.dump(out, open(os.path.join(outdir, 'results_svmcal.json'), 'w'), indent=1); print(out)
