# Data (not included in this repository)

This analysis uses the **2018 open reuse release of Pfizer's ATLAS** (Antimicrobial Testing Leadership and Surveillance) database. The release covers 2004–2017 and contains 633,820 isolates.

- ATLAS data are available to researchers on request through the **Vivli AMR Register**: https://amr.vivli.org
- Save the file you obtain as `data/Open_Atlas_Reuse_Data.xlsx`.
- Do not commit the data file, or any file derived from it, to the repository. `.gitignore` already excludes `data/*.xlsx`, `data/*.csv` and `work/`.
- Read the data provider's terms of use before publishing or sharing results, and cite or acknowledge the data as the provider requires.

**Expected input format:** the first sheet holds 5 header lines above the column names. The first three read "Micron Research", "15th October 2018" and "Combined TEST and Inform Data for the Data Re-Use Project". The columns include `Isolate Id`, `Study`, `Species`, `Country`, `State`, `Gender`, `Age Group`, `Speciality`, `Source`, `In / Out Patient`, `Year`, and one interpretation column per antibiotic (`<Drug>_I`).
