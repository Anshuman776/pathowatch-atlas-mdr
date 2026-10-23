"""Build an MDR-labelled isolate table from the 2018 Open ATLAS reuse file.

MDR definition: Magiorakos et al. 2012 (non-susceptible [I or R] to >=1 agent in
>=3 antimicrobial categories), species-specific category lists, restricted to the
agents present in ATLAS. Intrinsic resistances are not counted.
"""
import sys, json
import pandas as pd, numpy as np

src, out = sys.argv[1], sys.argv[2]
df = pd.read_csv(src, skiprows=5, dtype=str, low_memory=False).dropna(how='all')

SPECIES = ['Escherichia coli', 'Klebsiella pneumoniae', 'Pseudomonas aeruginosa', 'Acinetobacter baumannii']
df = df[df['Species'].isin(SPECIES)].copy()

ENTERO = {
    'Aminoglycosides': ['Amikacin', 'Gentamicin'],
    'Anti-MRSA cephalosporins': ['Ceftaroline'],
    'Antipseudomonal penicillins + BLI': ['Piperacillin tazobactam'],
    'Carbapenems': ['Ertapenem', 'Imipenem', 'Meropenem', 'Doripenem'],
    'Extended-spectrum cephalosporins': ['Ceftriaxone', 'Ceftazidime', 'Cefepime'],
    'Cephamycins': ['Cefoxitin'],
    'Fluoroquinolones': ['Ciprofloxacin', 'Levofloxacin'],
    'Folate pathway inhibitors': ['Trimethoprim sulfa'],
    'Glycylcyclines': ['Tigecycline'],
    'Monobactams': ['Aztreonam'],
    'Penicillins': ['Ampicillin'],
    'Penicillins + BLI': ['Amoxycillin clavulanate', 'Ampicillin sulbactam'],
    'Polymyxins': ['Colistin'],
    'Tetracyclines': ['Tetracycline', 'Minocycline'],
}
KPN = {k: v for k, v in ENTERO.items() if k != 'Penicillins'}  # intrinsic ampicillin resistance
PSA = {
    'Aminoglycosides': ['Amikacin', 'Gentamicin'],
    'Antipseudomonal carbapenems': ['Imipenem', 'Meropenem', 'Doripenem'],
    'Antipseudomonal cephalosporins': ['Ceftazidime', 'Cefepime'],
    'Antipseudomonal fluoroquinolones': ['Ciprofloxacin', 'Levofloxacin'],
    'Antipseudomonal penicillins + BLI': ['Piperacillin tazobactam'],
    'Monobactams': ['Aztreonam'],
    'Polymyxins': ['Colistin'],
}
ACB = {
    'Aminoglycosides': ['Amikacin', 'Gentamicin'],
    'Antipseudomonal carbapenems': ['Imipenem', 'Meropenem', 'Doripenem'],
    'Antipseudomonal fluoroquinolones': ['Ciprofloxacin', 'Levofloxacin'],
    'Antipseudomonal penicillins + BLI': ['Piperacillin tazobactam'],
    'Extended-spectrum cephalosporins': ['Ceftriaxone', 'Ceftazidime', 'Cefepime'],
    'Folate pathway inhibitors': ['Trimethoprim sulfa'],
    'Penicillins + BLI': ['Ampicillin sulbactam'],
    'Polymyxins': ['Colistin'],
    'Tetracyclines': ['Tetracycline', 'Minocycline'],
}
CATS = {'Escherichia coli': ENTERO, 'Klebsiella pneumoniae': KPN,
        'Pseudomonas aeruginosa': PSA, 'Acinetobacter baumannii': ACB}

rows_tested, rows_ns = [], []
for idx, r in df.iterrows():
    cats = CATS[r['Species']]
    t = ns = 0
    for agents in cats.values():
        vals = [r.get(a + '_I') for a in agents]
        vals = [v for v in vals if isinstance(v, str)]
        if vals:
            t += 1
            if any(v in ('Resistant', 'Intermediate') for v in vals):
                ns += 1
    rows_tested.append(t); rows_ns.append(ns)
df['cats_tested'] = rows_tested
df['cats_ns'] = rows_ns
df['Year'] = df['Year'].astype(int)
df['SourceGroup'] = df['Source'].fillna('Unknown').str.split(':').str[0].str.strip()
df['CarbapenemNS'] = np.nan  # descriptive only
carb = ['Imipenem_I', 'Meropenem_I', 'Doripenem_I', 'Ertapenem_I']
has = df[carb].notna().any(axis=1)
df.loc[has, 'CarbapenemNS'] = df.loc[has, carb].isin(['Resistant', 'Intermediate']).any(axis=1).astype(float)

keep = ['Isolate Id', 'Study', 'Species', 'Country', 'Gender', 'Age Group', 'Speciality',
        'Source', 'SourceGroup', 'In / Out Patient', 'Year', 'cats_tested', 'cats_ns', 'CarbapenemNS']
df[keep].to_csv(out, index=False)

print(len(df))
print(df.groupby(['Species'])['cats_tested'].describe().to_string())
print(pd.crosstab(df['Species'], df['cats_tested']).to_string())
print(pd.crosstab([df['Study'], df['Species']], df['cats_tested'] >= 5).to_string())
print(df['SourceGroup'].value_counts().to_string())
