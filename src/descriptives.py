"""Descriptive statistics for the cohort tables (paper Tables II and III, Section III/IV-A text).

Usage: python3 descriptives.py <mdr_base.csv> <outdir>
Writes results_desc.json and results_cohort.json.
"""
import sys, os, json
import pandas as pd

src, outdir = sys.argv[1], sys.argv[2]
exec(open(os.path.join(os.path.dirname(os.path.abspath(__file__)), 'common_prep.py')).read())  # defines d, inc

SPECIES = ['Escherichia coli', 'Klebsiella pneumoniae', 'Pseudomonas aeruginosa', 'Acinetobacter baumannii']
ICU = ['Medicine ICU', 'Surgery ICU', 'Pediatric ICU', 'General Unspecified ICU']
AGE65 = ['65 to 84 Years', '85 and Over']
CHILD = ['0 to 2 Years', '3 to 12 Years', '13 to 18 Years']

# ---- results_desc.json -------------------------------------------------------
desc = {
    'overall_mdr': float(d['MDR'].mean()),
    'overall_k': int(d['MDR'].sum()),
    'n_countries': int(d['Country'].nunique()),
    'inc_countries': {g: int(sum(1 for v in inc.values() if v == g)) for g in ['HIC', 'UMIC', 'LMIC']},
    'cats_tested_median': {s: float(g.median()) for s, g in sorted(d.groupby('Species')['cats_tested'])},
    'cats_tested_range': {s: [int(g.min()), int(g.max())] for s, g in sorted(d.groupby('Species')['cats_tested'])},
    'n_source_groups': int(d['SourceGroup'].nunique()),
    'source_groups': sorted(d['SourceGroup'].unique().tolist()),
    'n_spec': int(d['Speciality'].nunique()),
    'india_by_species': {s: [int(len(g)), float(g['MDR'].mean())]
                         for s, g in sorted(d[d['Country'] == 'India'].groupby('Species'))},
    'train_hic': int(((d['Income'] == 'HIC') & (d['Year'] <= 2014)).sum()),
    'lmic_countries_test': sorted(d[(d['Income'] == 'LMIC') & (d['Year'] >= 2015)]['Country'].unique().tolist()),
    'pct_inpatient': float((d['In / Out Patient'] == 'Inpatient').mean()),
}
json.dump(desc, open(os.path.join(outdir, 'results_desc.json'), 'w'), indent=1)

# ---- results_cohort.json (Table II) ------------------------------------------
def profile(g):
    r = {'n': int(len(g)), 'countries': int(g['Country'].nunique())}
    for s in SPECIES:
        r['sp_' + s] = float((g['Species'] == s).mean())
    r.update({
        'inpatient': float((g['In / Out Patient'] == 'Inpatient').mean()),
        'icu': float(g['Speciality'].isin(ICU).mean()),
        'age65': float(g['Age Group'].isin(AGE65).mean()),
        'child': float(g['Age Group'].isin(CHILD).mean()),
        'female': float((g['Gender'] == 'Female').mean()),
        'blood': float((g['SourceGroup'] == 'CVS').mean()),          # blood / cardiovascular specimens
        'resp': float((g['SourceGroup'] == 'Respiratory').mean()),
        'urine': float((g['SourceGroup'] == 'GU').mean()),           # genito-urinary specimens
        'inform': float((g['Study'] == 'Inform').mean()),
        'years': f"{int(g['Year'].min())}--{int(g['Year'].max())}",
    })
    return r

cohort = {'All': profile(d)}
for grp in ['HIC', 'UMIC', 'LMIC']:
    cohort[grp] = profile(d[d['Income'] == grp])
cohort['India'] = profile(d[d['Country'] == 'India'])
json.dump(cohort, open(os.path.join(outdir, 'results_cohort.json'), 'w'), indent=1)
print('descriptives written to', outdir)
