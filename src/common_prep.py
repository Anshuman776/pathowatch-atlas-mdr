d = pd.read_csv(src)
d['MDR'] = (d['cats_ns'] >= 3).astype(int)
_fix = {'Genital/Urinary (GU)': 'GU', 'GastroIntestinal (GI)': 'GI', 'CardioVascular (CVS)': 'CVS',
        'Integumentary (INT)': 'INT', 'Central Nervous (CNS)': 'CNS', 'None Given': 'Unknown'}
d['SourceGroup'] = d['SourceGroup'].replace(_fix)
for _c in ['Gender', 'Age Group', 'Speciality', 'In / Out Patient', 'Country', 'SourceGroup']:
    d[_c] = d[_c].fillna('Unknown').astype(str)
_LMIC = ['India', 'Pakistan', 'Philippines', 'Morocco', 'Tunisia', 'Kenya', 'Nigeria', 'Honduras',
         'El Salvador', 'Egypt', 'Vietnam', 'Indonesia', 'Nicaragua']
_UMIC = ['Mexico', 'China', 'South Africa', 'Colombia', 'Russia', 'Venezuela', 'Turkey', 'Brazil',
         'Thailand', 'Romania', 'Malaysia', 'Guatemala', 'Bulgaria', 'Jordan', 'Serbia', 'Jamaica',
         'Lebanon', 'Namibia', 'Mauritius']
inc = {c: ('LMIC' if c in _LMIC else 'UMIC' if c in _UMIC else 'HIC') for c in d['Country'].unique()}
d['Income'] = d['Country'].map(inc)
