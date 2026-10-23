"""Fill the journal manuscript template with numbers from the analysis outputs."""
import sys, json, re
import numpy as np
tpl, rdir, out = sys.argv[1], sys.argv[2], sys.argv[3]
L = lambda f: json.load(open(f'{rdir}/{f}'))
R, X, D, E, C, SV = L('results.json'), L('results_extra.json'), L('results_desc.json'), L('results_ext.json'), L('results_cohort.json'), L('results_svmcal.json')

def pct(x, nd=1): return f'{100 * x:.{nd}f}\\%'
def pn(x, nd=1): return f'{100 * x:.{nd}f}'
def f3(x): return f'{x:.3f}'
def f2(x): return f'{x:.2f}'
def n(x): return f'{int(x):,}'.replace(',', '{,}')
def ci3(v, lo, hi): return f'{v:.3f} ({lo:.3f}--{hi:.3f})'
def orr(v): return f"{v['or_']:.2f} ({v['lo']:.2f}--{v['hi']:.2f})"
def orr_ci(v): return f"{v['or_']:.2f} (95\\% CI {v['lo']:.2f}--{v['hi']:.2f})"

SP = ['Escherichia coli', 'Klebsiella pneumoniae', 'Pseudomonas aeruginosa', 'Acinetobacter baumannii']
AB = {s: '\\sci{' + s.split()[0][0] + '. ' + s.split()[1] + '}' for s in SP}
e1, tr, desc, dinc = R['exp1_temporal'], R['exp2_transfer'], R['descriptive'], R['descriptive_income']
tc, lo, fc, sens = E['transfer_calibration'], E['loro'], E['forecasting'], E['sensitivity']
V = {}
V['SKLEARN'] = '1.9.1'
V['N_TOTAL'] = n(R['meta']['n_total']); V['N_COUNTRIES'] = str(D['n_countries'])
V['N_TRAIN'] = n(R['meta']['n_train']); V['N_TEST'] = n(R['meta']['n_test']); V['MDR_TEST'] = pct(R['meta']['mdr_test'])
V['MDR_OVERALL'] = pct(D['overall_mdr'])
g = e1['GBM']; V['GBM_AUROC'] = f3(g['auroc']); V['GBM_AUROC_CI'] = ci3(g['auroc'], *g['auroc_ci'])
ct = E['calibration_temporal']
V['GBM_SLOPE'] = f2(ct['GBM']['cal_slope']); V['GBM_CI_INT'] = f2(ct['GBM']['cal_intercept']); V['GBM_ECE'] = f3(ct['GBM']['ece'])
V['RF_AUROC'] = f3(e1['RF']['auroc']); V['BASE_AUROC'] = f3(e1['Baseline']['auroc'])
V['GBM_SENS'] = pct(g['sens']); V['GBM_SPEC'] = pct(g['spec'])
V['RS_AUROC'] = f3(sens['random_split']['auroc']); V['RS_GAP'] = f'{sens["random_split"]["auroc"] - g["auroc"]:.3f}'
V['SENS_NOC'] = f3(sens['no_country']['auroc'])
V['INC_P_HIC'], V['INC_P_UMIC'], V['INC_P_LMIC'] = [pct(dinc[k]['p']) for k in ('HIC', 'UMIC', 'LMIC')]
V['IND_P'] = pct(dinc['India']['p']); V['IND_N'] = n(dinc['India']['n'])
V['IND_AB'] = pct(D['india_by_species']['Acinetobacter baumannii'][1]); V['IND_EC'] = pct(D['india_by_species']['Escherichia coli'][1])
V['INC_HIC'], V['INC_UMIC'], V['INC_LMIC'] = [str(D['inc_countries'][k]) for k in ('HIC', 'UMIC', 'LMIC')]
V['N_TRAIN_HIC'] = n(tr['HIC-trained']['n_train'])
V['CT_EC'], V['CT_KP'], V['CT_PA'], V['CT_AB'] = [str(int(D['cats_tested_median'][s])) for s in SP]
V['N_SRC'] = str(D['n_source_groups']); V['N_SPEC'] = str(D['n_spec'])
V['SHARE_HIC'] = pct(C['HIC']['n'] / C['All']['n']); V['N_LMIC'] = n(C['LMIC']['n'])
V['IND_ICU'] = pct(C['India']['icu']); V['ALL_ICU'] = pct(C['All']['icu'])
V['IND_BLOOD'] = pct(C['India']['blood']); V['ALL_BLOOD'] = pct(C['All']['blood'])
V['IND_AB_SHARE'] = pct(C['India']['sp_Acinetobacter baumannii']); V['ALL_AB_SHARE'] = pct(C['All']['sp_Acinetobacter baumannii'])
kp = desc['Klebsiella pneumoniae']
V['IND_CARB_KP'] = pct(kp['carb_ns_india']); V['ALL_CARB_KP'] = pct(kp['carb_ns'])
V['KP_2004'] = pct(kp['by_year']['2004']['p']); V['KP_2017'] = pct(kp['by_year']['2017']['p'])
at = X['adjusted_trend']
for key, s in [('TR_KP', 'Klebsiella pneumoniae'), ('TR_AB', 'Acinetobacter baumannii'), ('TR_EC', 'Escherichia coli'), ('TR_PA', 'Pseudomonas aeruginosa')]:
    v = at[s]; V[key] = f"OR {v['or_per_year']:.3f}, 95\\% CI {v['lo']:.3f}--{v['hi']:.3f}"

# --- funnel (countries) with overdispersion ---
fcn = E['funnel_country']; units = fcn['units']; z = np.array([u['z'] for u in units])
lo_q, hi_q = np.percentile(z, [10, 90]); zw = np.clip(z, lo_q, hi_q); phi = float(np.mean(zw ** 2))
lim = 3.09
V['FUN_UNITS'] = str(len(units)); V['FUN_ABOVE'] = str(int((z > lim).sum())); V['FUN_BELOW'] = str(int((z < -lim).sum()))
V['FUN_PHI'] = f'{phi:.1f}'; V['FUN_PHI_RAW'] = f'{phi:.4f}'; V['FUN_PBAR'] = f"{fcn['pbar']:.4f}"
adj = [u for u in units if u['z'] / np.sqrt(phi) > lim]
V['FUN_ABOVE_ADJ'] = str(len(adj)); V['FUN_ADJ_LIST'] = ', '.join(u['unit'].replace('Korea, South', 'South Korea') for u in adj) if adj else 'none'
ind = [u for u in units if u['unit'] == 'India'][0]
V['FUN_IND_RATIO'] = f"{ind['ratio']:.2f}"; V['FUN_IND_PCT'] = f"{100 * (ind['ratio'] - 1):.0f}\\%"
V['FUN_POINTS'] = ' '.join(f"({u['E']:.1f},{u['ratio']:.3f})" for u in units)
lab = {'India': 'west', 'Pakistan': 'east', 'Vietnam': 'west', 'United States': 'east', 'Guatemala': 'west', 'Mexico': 'west', 'Russia': 'north'}
V['FUN_LABELS'] = '\n'.join(f"\\node[font=\\scriptsize, anchor={a}] at (axis cs:{u['E']:.1f},{u['ratio']:.3f}) {{{u['unit']}}};"
                             for u in units for k, a in lab.items() if u['unit'] == k)
us = E['funnel_us_state']['units']; zus = [u for u in us if u['z'] > lim]
def re_adj(uu):
    y = np.array([u['ratio'] for u in uu]); O = np.array([u['O'] for u in uu]); Ex = np.array([u['E'] for u in uu]); zz = np.array([u['z'] for u in uu])
    Vv = ((O - Ex) / zz) ** 2; s2 = Vv / Ex ** 2; w = 1 / s2; yb = (w * y).sum() / w.sum(); Q = (w * (y - yb) ** 2).sum(); k = len(y)
    t2 = max(0.0, (Q - (k - 1)) / (w.sum() - (w ** 2).sum() / w.sum())); za = (y - 1) / np.sqrt(s2 + t2)
    return t2, [uu[i]['unit'] for i in np.argsort(-za) if za[i] > 3.09], [uu[i]['unit'] for i in np.argsort(-za) if 1.96 < za[i] <= 3.09]
t2c, a99, a95 = re_adj(units); t2u, u99, u95 = re_adj(us)
V['FUN_TAU2'] = f'{t2c:.5f}'
def lst(x): return ', '.join(x[:-1]) + ' and ' + x[-1] if len(x) > 1 else x[0]
if a99: V['FUN_RE_SENTENCE'] = f"After adding a between-country random-effects term (estimated standard deviation of the ratio {np.sqrt(t2c):.2f}) to the limits, {len(a99)} countries remained above the upper 99.8\\% limit ({lst(a99)})."
else:
    tail = f" and only {lst(a95)} exceeded the 95\\% limit" if a95 else ''
    V['FUN_RE_SENTENCE'] = (f"After adding a between-country random-effects term (estimated standard deviation of the ratio {np.sqrt(t2c):.2f}) to the limits, no country lay outside the 99.8\\% limits{tail}. "
                            "High-burden countries therefore form part of a broad, systematic gradient rather than a few extreme outliers.")
V['US_RE_SENTENCE'] = (f"after the random-effects adjustment, {lst(u95)} remained above the 95\\% limit." if u95 and not u99 else
                       (f"after the random-effects adjustment, {lst(u99)} remained above the 99.8\\% limit." if u99 else "none remained outside the limits after the random-effects adjustment."))
V['US_ABOVE'] = str(len(zus)); V['US_ABOVE_LIST'] = ', '.join(u['unit'] for u in zus)

# --- importance / ORs ---
imp = R['importance']; V['IMP_COUNTRY'] = f3(imp['Country']['mean']); V['IMP_SPECIES'] = f3(imp['Species']['mean'])
order = [('Country', 'Country'), ('Species', 'Species'), ('SourceGroup', 'Specimen source'), ('Speciality', 'Specialty'),
         ('Age Group', 'Age group'), ('Gender', 'Gender'), ('In / Out Patient', 'Patient type'), ('Year', 'Year')]
V['IMP_COORDS'] = ','.join(l for _, l in order)
V['IMP_DATA'] = ' '.join(f"({max(imp[k]['mean'], 0):.4f},{l})" for k, l in order)
o = X['adjusted_or']; oi = X['adjusted_or_india']['Country=India']
V['OR_MICU'] = orr_ci(o['Speciality=Medicine ICU']); V['OR_SICU'] = orr(o['Speciality=Surgery ICU'])
V['OR_NH'] = orr(o['Speciality=Nursing Home / Rehab']); V['OR_DEV'] = orr(o['SourceGroup=Instruments'])
V['OR_INDIA'] = f"{oi['or_']:.1f} (95\\% CI {oi['lo']:.1f}--{oi['hi']:.1f})"
sel = [('Species (ref. \\sci{E. coli})', None), ('\\quad \\sci{A. baumannii}', 'Species=Acinetobacter baumannii'),
       ('\\quad \\sci{K. pneumoniae}', 'Species=Klebsiella pneumoniae'), ('\\quad \\sci{P. aeruginosa}', 'Species=Pseudomonas aeruginosa'),
       ('Specialty (ref. general medicine)', None), ('\\quad Medical ICU', 'Speciality=Medicine ICU'), ('\\quad Surgical ICU', 'Speciality=Surgery ICU'),
       ('\\quad General/unspecified ICU', 'Speciality=General Unspecified ICU'), ('\\quad Nursing home / rehabilitation', 'Speciality=Nursing Home / Rehab'),
       ('\\quad Surgery general', 'Speciality=Surgery General'), ('\\quad Paediatric general', 'Speciality=Pediatric General'),
       ('\\quad Emergency room', 'Speciality=Emergency Room'), ('\\quad Clinic / office', 'Speciality=Clinic / Office'),
       ('Specimen (ref. urinary)', None), ('\\quad Medical devices', 'SourceGroup=Instruments'), ('\\quad Respiratory', 'SourceGroup=Respiratory'),
       ('\\quad Skin / wound', 'SourceGroup=INT'), ('\\quad Blood / cardiovascular', 'SourceGroup=CVS'), ('\\quad Body fluids', 'SourceGroup=Bodily Fluids'),
       ('Age (ref. 19--64 years)', None), ('\\quad 0--2 years', 'Age Group=0 to 2 Years'), ('\\quad 3--12 years', 'Age Group=3 to 12 Years'),
       ('\\quad 13--18 years', 'Age Group=13 to 18 Years'), ('\\quad 65--84 years', 'Age Group=65 to 84 Years'), ('\\quad 85 years and over', 'Age Group=85 and Over'),
       ('Female (ref. male)', 'Gender=Female'), ('Outpatient (ref. inpatient)', 'In / Out Patient=Outpatient'), ('Calendar year (per year)', 'Year')]
rows = []
for l, k in sel:
    rows.append(f"\\multicolumn{{2}}{{l}}{{\\textit{{{l}}}}} \\\\" if k is None else f"{l} & {orr(o[k])} \\\\")
rows.append(f"India (ref. United States) & {orr(oi)} \\\\")
V['TAB_OR'] = '\n'.join(rows)

# --- transfer ---
h, a = tr['HIC-trained'], tr['All-trained']
V['T_HIC_LMIC_AUROC'] = f3(h['LMIC']['auroc']); V['T_HIC_LMIC_PRED'] = pct(h['LMIC']['mean_pred']); V['T_HIC_LMIC_OBS'] = pct(h['LMIC']['prevalence'])
V['T_HIC_IND_PRED'] = pct(h['India']['mean_pred']); V['T_HIC_IND_OBS'] = pct(h['India']['prevalence'])
V['T_HIC_HIC_PRED'] = pct(h['HIC']['mean_pred']); V['T_HIC_HIC_OBS'] = pct(h['HIC']['prevalence'])
V['N_LMIC_TEST'] = n(h['LMIC']['n'])
gaps = [h[k]['prevalence'] - h[k]['mean_pred'] for k in ('UMIC', 'LMIC', 'India')]
V['GAP_MIN'], V['GAP_MAX'] = f'{100 * min(gaps):.0f}', f'{100 * max(gaps):.0f}'
V['ECE_HIC'] = f3(tc['HIC']['none']['ece'])
V['CI_UMIC'], V['CI_LMIC'], V['CI_IND'] = [f2(tc[k]['none']['cal_intercept']) for k in ('UMIC', 'LMIC', 'India')]
rows = []
for k in ('HIC', 'UMIC', 'LMIC', 'India'):
    t = tc[k]['none']
    rows.append(f"{k} & {n(h[k]['n'])} & {pn(h[k]['prevalence'])} & {f3(h[k]['auroc'])} & {pn(h[k]['mean_pred'])} & {f2(t['cal_intercept'])} & "
                f"{f3(t['ece'])} & {f3(a[k]['auroc'])} & {pn(a[k]['mean_pred'])} \\\\")
V['TAB_TRANSFER'] = '\n'.join(rows)
V['ECE_LMIC_NONE'] = f3(tc['LMIC']['none']['ece']); V['ECE_LMIC_RECENT'] = f3(tc['LMIC']['intercept_recent_year']['ece'])
V['ECE_UMIC_NONE'] = f3(tc['UMIC']['none']['ece']); V['ECE_UMIC_RECENT'] = f3(tc['UMIC']['intercept_recent_year']['ece'])
V['ECE_IND_NONE'] = f3(tc['India']['none']['ece']); V['ECE_IND_RECENT'] = f3(tc['India']['intercept_recent_year']['ece'])
V['IND_RECENT_N'] = n(tc['India']['n_recent_year'])
rows = []
for k in ('UMIC', 'LMIC', 'India'):
    t = tc[k]; ya, yr = t['years_all_years'], t['years_recent_year']
    spec = [('None (imported model)', 'none', None), (f'Intercept, {ya[0]}--{ya[1]}', 'intercept_all_years', 'n_all_years'),
            (f'Logistic, {ya[0]}--{ya[1]}', 'logistic_all_years', 'n_all_years'), (f'Intercept, {yr[0]} only', 'intercept_recent_year', 'n_recent_year'),
            (f'Logistic, {yr[0]} only', 'logistic_recent_year', 'n_recent_year')]
    for i, (lab_, key, nk) in enumerate(spec):
        m = t[key]; first = k if i == 0 else ''
        rows.append(f"{first} & {lab_} & {n(t[nk]) if nk else '--'} & {pn(m['mean_pred'])} & {f2(m['cal_intercept'])} & {f3(m['ece'])} & {f3(m['brier'])} \\\\")
    rows.append(f"& \\textit{{Observed prevalence}} & & {pn(t['none']['observed'])} & & & \\\\")
    if k != 'India': rows.append('\\midrule')
V['TAB_RECAL'] = '\n'.join(rows)

# --- LORO ---
rows = []; drops = []
for reg, v in lo.items():
    ho, inc_ = v['held_out'], v['included']; drops.append(inc_['auroc'] - ho['auroc'])
    rows.append(f"{reg} & {n(v['n_test'])} & {pn(ho['observed'])} & {f3(ho['auroc'])} & {pn(ho['mean_pred'])} / {f3(ho['ece'])} & "
                f"{f3(inc_['auroc'])} & {pn(inc_['mean_pred'])} / {f3(inc_['ece'])} \\\\")
V['TAB_LORO'] = '\n'.join(rows); V['LORO_MAXDROP'] = f'{max(drops):.3f}'
V['LORO_EMR_ECE'] = f3(lo['Eastern Mediterranean']['held_out']['ece']); V['LORO_SEA_ECE'] = f3(lo['South-East Asia']['held_out']['ece'])

# --- learning curve ---
lc = E['learning_curve']; full = lc['1.0']['auroc_mean']
V['LC_N'] = n(lc['0.05']['n']); V['LC_AUROC'] = f3(lc['0.05']['auroc_mean']); V['LC_SHARE'] = f"{100 * lc['0.05']['auroc_mean'] / full:.0f}\\%"
V['LC_SMALL_N'] = '2{,}000'; V['LC_SMALL_AUROC'] = f"AUROC {lc['0.01']['auroc_mean']:.3f} with {n(lc['0.01']['n'])} isolates"
V['FIG_LC'] = ' '.join(f"({v['n']},{v['auroc_mean']:.4f}) +- (0,{v['auroc_sd']:.4f})" for k, v in sorted(lc.items(), key=lambda t: float(t[0])))

# --- subgroups ---
sub = E['subgroups']; ren = {'CVS': 'Blood / cardiovascular', 'GU': 'Genito-urinary', 'INT': 'Skin / wound', 'Bodily Fluids': 'Body fluids',
                             'None Given': 'Not recorded', 'Inform': 'INFORM'}
rows = []; vals = []
for grp, levels in sub.items():
    for i, (lv, v) in enumerate(levels.items()):
        vals.append(v['auroc'])
        rows.append(f"{grp if i == 0 else ''} & {ren.get(lv, lv)} & {n(v['n'])} & {pn(v['prevalence'])} & {ci3(v['auroc'], *v['ci'])} \\\\")
    rows.append('\\addlinespace[2pt]')
V['TAB_SUB'] = '\n'.join(rows[:-1]); V['SUB_MIN'] = f3(min(vals)); V['SUB_MAX'] = f3(max(vals))
V['SUB_LMIC'] = f3(sub['Income group']['LMIC']['auroc']); V['SUB_SEA'] = f3(sub['WHO region']['South-East Asia']['auroc'])
V['SUB_AFR'] = f3(sub['WHO region']['Africa']['auroc'])
ps = R['exp1_per_species']
V['SP_AB'] = ci3(ps['Acinetobacter baumannii']['GBM']['auroc'], *ps['Acinetobacter baumannii']['GBM']['auroc_ci'])
V['SP_PA'] = ci3(ps['Pseudomonas aeruginosa']['GBM']['auroc'], *ps['Pseudomonas aeruginosa']['GBM']['auroc_ci'])
rows = []
for sname in SP:
    v = ps[sname]
    rows.append(f"{AB[sname]} & {n(v['GBM']['n'])} & {pn(v['GBM']['prevalence'])} & {f3(v['LR']['auroc'])} & {ci3(v['GBM']['auroc'], *v['GBM']['auroc_ci'])} \\\\")
iv = R['exp1_india']
rows.append('\\midrule')
rows.append(f"India, test subset$^a$ & {n(iv['GBM']['n'])} & {pn(iv['GBM']['prevalence'])} & {f3(iv['LR']['auroc'])} & {ci3(iv['GBM']['auroc'], *iv['GBM']['auroc_ci'])} \\\\")
V['TAB_SPECIES'] = '\n'.join(rows)
V['IND_POOLED_AUROC'] = ci3(iv['GBM']['auroc'], *iv['GBM']['auroc_ci']); V['IND_TEST_N'] = n(iv['GBM']['n']); V['IND_TEST_P'] = pct(iv['GBM']['prevalence'])
V['T_HIC_IND_AUROC'] = f3(tr['HIC-trained']['India']['auroc']); V['T_ALL_IND_AUROC'] = f3(tr['All-trained']['India']['auroc'])
V['RF_CI_INT'] = f2(ct['RF']['cal_intercept']); V['SVM_SLOPE'] = f2(SV['LinearSVM']['cal_slope'])

# --- sensitivity ---
rows = []
for k in ('main', 'random_split', 'no_country', 'test_only', 'strict'):
    v = sens[k]; rows.append(f"{v['label']} & {n(v['n_train'])} & {n(v['n_test'])} & {pn(v['prevalence'])} & {f3(v['auroc'])} & {f3(v['auprc'])} \\\\")
V['TAB_SENS'] = '\n'.join(rows)

# --- model table ---
rows = []
calt = {**E['calibration_temporal'], **SV}
for k, lab_ in [('Baseline', 'Baseline'), ('LR', 'Logistic regression'), ('LinearSVM', 'Linear SVM'), ('RF', 'Random forest'), ('GBM', 'Gradient boosting')]:
    v = e1[k]; c = calt[k]
    au = ci3(v['auroc'], *v['auroc_ci']); au = '\\textbf{' + au + '}' if k == 'GBM' else au
    ss = '--' if k == 'Baseline' else pn(v['sens']); sp_ = '--' if k == 'Baseline' else pn(v['spec'])
    rows.append(f"{lab_} & {au} & {f3(v['auprc'])} & {f3(v['brier'])} & {f2(c['cal_intercept'])} & {f2(c['cal_slope'])} & {f3(c['ece'])} & {ss} & {sp_} \\\\")
V['TAB_MODELS'] = '\n'.join(rows)

# --- cohort table ---
cols = ['All', 'HIC', 'UMIC', 'LMIC', 'India']
def row(label, key, fmt=pn): return f"{label} & " + ' & '.join(fmt(C[c][key]) for c in cols) + ' \\\\'
rows = [f"Isolates ($n$) & " + ' & '.join(n(C[c]['n']) for c in cols) + ' \\\\',
        f"Countries & " + ' & '.join(str(C[c]['countries']) for c in cols) + ' \\\\',
        f"Years & " + ' & '.join(C[c]['years'] for c in cols) + ' \\\\', '\\midrule']
for s in SP: rows.append(row(AB[s], 'sp_' + s))
rows += ['\\midrule', row('Inpatient', 'inpatient'), row('Intensive care unit', 'icu'), row('Age $\\geq$65 years', 'age65'),
         row('Age $\\leq$18 years', 'child'), row('Female', 'female'), row('Blood / cardiovascular specimen', 'blood'), row('Respiratory specimen', 'resp'),
         row('Genito-urinary specimen', 'urine'), row('INFORM study', 'inform')]
V['TAB_COHORT'] = '\n'.join(rows)

# --- prevalence table ---
rows = []
for s in SP:
    v = desc[s]; bi = v['by_income']
    rows.append(f"{AB[s]} & {n(v['n'])} & {pn(v['mdr'])} & {pn(bi['HIC']['p'])} & {pn(bi['UMIC']['p'])} & {pn(bi['LMIC']['p'])} & {pn(v['india']['p'])} & {pn(v['carb_ns'])} \\\\")
rows.append('\\midrule')
rows.append(f"All four & {n(R['meta']['n_total'])} & {pn(D['overall_mdr'])} & {pn(dinc['HIC']['p'])} & {pn(dinc['UMIC']['p'])} & {pn(dinc['LMIC']['p'])} & {pn(dinc['India']['p'])} & -- \\\\")
V['TAB_PREV'] = '\n'.join(rows)

# --- forecasting ---
V['FC_SERIES'] = str(fc['n_series']); V['FC_N'] = str(fc['n_forecasts'])
V['FC_MEAN3'] = f"{fc['Mean (last 3)']['mae']:.2f}"; V['FC_NAIVE'] = f"{fc['Naive']['mae']:.2f}"
V['FC_LIN'] = f"{fc['Linear trend']['mae']:.2f}"; V['FC_HOLT'] = f"{fc['Holt']['mae']:.2f}"; V['FC_SD'] = f"{fc['obs_change_sd']:.1f}"
rows = []
for mth, lab_ in [('Naive', 'Naive (last value)'), ('Mean (last 3)', 'Mean of last three values'), ('Linear trend', 'Linear trend'), ('Holt', "Holt's linear method")]:
    v = fc[mth]; m = f"{v['mae']:.2f} ({v['mae_ci'][0]:.2f}--{v['mae_ci'][1]:.2f})"
    if mth == 'Mean (last 3)': m = '\\textbf{' + m + '}'
    rows.append(f"{lab_} & {m} & {v['by_h']['1']:.2f} & {v['by_h']['2']:.2f} & {v['by_h']['3']:.2f} \\\\")
V['TAB_FC'] = '\n'.join(rows)

# --- figures ---
styles = {'Escherichia coli': 'solid, mark=*', 'Klebsiella pneumoniae': 'dashed, mark=square*',
          'Pseudomonas aeruginosa': 'dotted, mark=triangle*', 'Acinetobacter baumannii': 'dashdotted, mark=diamond*'}
V['FIG_TREND'] = '\n'.join(f"\\addplot[black, thick, {styles[s]}] coordinates {{" + ' '.join(f"({y},{100 * v['p']:.1f})" for y, v in sorted(desc[s]['by_year'].items())) +
                           f"}};\n\\addlegendentry{{{AB[s]}}}" for s in SP)
roc, grid = E['roc'], E['roc_grid']
rs = {'Baseline': 'gray, dashed', 'LR': 'black, dotted, thick', 'RF': 'black!60, dashdotted, thick', 'GBM': 'black, thick'}
names = {'Baseline': 'Baseline', 'LR': 'Logistic regression', 'RF': 'Random forest', 'GBM': 'Gradient boosting'}
V['FIG_ROC'] = '\n'.join(f"\\addplot[{rs[k]}] coordinates {{" + ' '.join(f"({x:.3f},{y:.3f})" for x, y in zip(grid, roc[k]['tpr'])) +
                         f"}};\n\\addlegendentry{{{names[k]} ({roc[k]['auroc']:.3f})}}" for k in ('Baseline', 'LR', 'RF', 'GBM'))
relH = X['reliability']['HIC']; relL = tc['LMIC']['reliability_before']; relR = tc['LMIC']['reliability_recent_logistic']
V['FIG_REL'] = '\n'.join([
    "\\addplot[black, thick, mark=*] coordinates {" + ' '.join(f"({r['mean_pred']:.3f},{r['observed']:.3f})" for r in relH) + "};\n\\addlegendentry{HIC, imported model}",
    "\\addplot[black, thick, dashed, mark=square*] coordinates {" + ' '.join(f"({r['pred']:.3f},{r['obs']:.3f})" for r in relL) + "};\n\\addlegendentry{LMIC, imported model}",
    "\\addplot[black!55, thick, mark=triangle*] coordinates {" + ' '.join(f"({r['pred']:.3f},{r['obs']:.3f})" for r in relR) + "};\n\\addlegendentry{LMIC, recalibrated (2014)}"])
V['WORDS'] = '@@WORDS@@'

s = open(tpl).read()
missing = sorted(set(re.findall(r'@@(\w+)@@', s)) - set(V)); assert not missing, missing
for k, v in V.items():
    if k != 'WORDS': s = s.replace(f'@@{k}@@', v)
# word count of main text (Introduction .. Conclusions), LaTeX stripped
if '\\section{Introduction}' in s and '\\section*{Declarations}' in s:
    body = s[s.index('\\section{Introduction}'):s.index('\\section*{Declarations}')]
else:
    body = ''
body = re.sub(r'\\begin\{(table|figure)\}.*?\\end\{\1\}', ' ', body, flags=re.S)
body = re.sub(r'\\(cite|ref|label)\{[^}]*\}', ' ', body); body = re.sub(r'\\[a-zA-Z]+\*?(\[[^\]]*\])?', ' ', body)
body = re.sub(r'[{}$\\%&~^_]', ' ', body); words = len([w for w in body.split() if re.search(r'[A-Za-z0-9]', w)])
s = s.replace('@@WORDS@@', f'{round(words, -2):,}'.replace(',', '{,}'))
open(out, 'w').write(s)
json.dump({k: v for k, v in V.items() if not k.startswith(('TAB_', 'FIG_', 'FUN_POINTS', 'FUN_LABELS'))}, open(out.replace('.tex', '_values.json'), 'w'), indent=1)
print('filled', len(V), 'placeholders; main-text words ~', words)
