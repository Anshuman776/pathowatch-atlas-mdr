# PathoWatch: MDR prediction from pre-test metadata on ATLAS (analysis code)

This repository holds the code that produced every number, table and figure in the paper:

> *Predicting Multidrug Resistance from Routine Isolate Metadata Before Susceptibility Testing: Temporal, Cross-Income and Cross-Regional Validation on Global ATLAS Surveillance Data*
> A. Maurya, P. Kushawaha, T. Alam, R. Kesarwani, A. Mishra. Dept. of IT, AKGEC Ghaziabad.

> **Keep this repository PRIVATE** until the institute's IPR Cell confirms that public release is safe, because the PathoWatch patent is not yet filed. The repository covers only the ATLAS analysis. It contains no PathoWatch platform, fusion-engine or alert logic.

## What the pipeline does

```
ATLAS .xlsx ──► 1 xlsx2csv ──► atlas.csv ──► 2 build_mdr ──► mdr_base.csv (232,527 labelled isolates)
                                                              │
     ┌────────────────┬──────────────────┬───────────────────┼──────────────────┐
     ▼                ▼                  ▼                   ▼                  ▼
3 descriptives   4 run_experiments   5 run_extra       6 run_extended      7 svm_cal
     │                │                  │                   │                  │
     └────────────────┴───────► results/*.json ◄─────────────┴──────────────────┘
                                         │
                               8 fill_paper_numbers ──► paper/ieee/build/main.tex (+ figures)
```

Every number in the paper is inserted by step 8 from `results/*.json`, so the text cannot drift from the analysis.

| Step | Script | What it does | Output | Used in paper |
|---|---|---|---|---|
| 1 | `src/xlsx2csv.py` | Streams the large Excel file to CSV without loading it into memory | `work/atlas.csv` | — |
| 2 | `src/build_mdr.py` | MDR label per isolate. Uses species-specific Magiorakos et al. (2012) categories restricted to ATLAS agents: *E. coli* 14 categories, *K. pneumoniae* 13 (ampicillin intrinsic), *P. aeruginosa* 7, *A. baumannii* 9. Levofloxacin is counted with ciprofloxacin. MDR = non-susceptible (I or R) in ≥3 categories | `work/mdr_base.csv` | Sec. III-B, Table XIII |
| 3 | `src/descriptives.py` | Cohort counts and percentages by income group and for India | `results_desc.json`, `results_cohort.json` | Tables II–III, Sec. III-A, IV-A |
| 4 | `src/run_experiments.py` | Train on 2004–2014, test on 2015–2017. Four models: LR, linear SVM, RF, HistGradientBoosting, plus a species+year baseline. Bootstrap 95% CIs. F1 threshold chosen on 2014. Per-species AUROC, permutation importance, HIC → UMIC/LMIC/India transfer | `results.json` | Tables IV, V, IX; Figs. 4–5 |
| 5 | `src/run_extra.py` | Unpenalised LR odds ratios (IRLS, Wald CIs), country-adjusted yearly trends, recalibration, reliability by quintile | `results_extra.json` | Table VIII, Sec. IV-B, Fig. 6 |
| 6 | `src/run_extended.py` | ROC curves, calibration (intercept, slope, ECE), subgroups, learning curve, sensitivity analyses, four recalibration strategies, leave-one-WHO-region-out, funnel plots (standard and DerSimonian–Laird limits; US states), yearly forecasting (naive, 3-year mean, linear, Holt) | `results_ext.json` | Tables VI, VII, X, XI, XII; Figs. 2, 3, 6, 7 |
| 7 | `src/svm_cal.py` | Calibration metrics for the linear SVM and the baseline | `results_svmcal.json` | Table IV |
| 8 | `paper/fill_paper_numbers.py` | Replaces each `@@KEY@@` placeholder in the LaTeX templates with values from the JSON files. Stops with an error if any value is missing | `paper/ieee/build/` | whole paper |
| — | `src/compare_results.py` | Checks new results against `results/expected/` | console | reproducibility check |

`src/common_prep.py` holds the shared cleaning steps used by steps 3–7: the MDR flag, specimen-group names, unknown categories kept as explicit levels, and the World Bank FY2019 income groups.

## How to reproduce

### 1. Get the data

The ATLAS data cannot be included in this repository. See `data/README.md`, then place the file at `data/Open_Atlas_Reuse_Data.xlsx`.

### 2. Set up Python

The original run used Python 3.13, CPU only, on Linux.

```
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
```

### 3. Run everything

```
bash run_all.sh data/Open_Atlas_Reuse_Data.xlsx
```

The full run took 14 minutes (830 s) on the original machine and may take longer on a laptop; random forest, bootstrapping and the learning curve are the slow parts. At the end, `compare_results.py` should print **"All results reproduced exactly."** All models and resampling use a fixed seed (42).

### Running steps one at a time

```
python3 src/xlsx2csv.py data/Open_Atlas_Reuse_Data.xlsx work/atlas.csv
python3 src/build_mdr.py work/atlas.csv work/mdr_base.csv
python3 src/descriptives.py work/mdr_base.csv results
python3 src/run_experiments.py work/mdr_base.csv results
python3 src/run_extra.py work/mdr_base.csv results
python3 src/run_extended.py work/mdr_base.csv results work/atlas.csv
python3 src/svm_cal.py work/mdr_base.csv results
python3 paper/fill_paper_numbers.py paper/ieee/main_template.tex results paper/ieee/build/main.tex
```

## Repository layout

```
src/                 analysis scripts (steps 1-7) + compare_results.py
paper/
  fill_paper_numbers.py
  ieee/              IEEE conference template (@@KEY@@ placeholders), figure templates, references.bib
  journal/           journal manuscript template
results/expected/    the result files and run logs behind the submitted paper
data/                put the ATLAS file here (not committed)
run_all.sh           one-command reproduction
```

## Key design choices (the reasons behind the code)

- **Only pre-test predictors:** species, specimen group, specialty, patient type, age group, gender, country and year. Susceptibility results and the ESBL phenotype are excluded because they would leak the outcome.
- **Forward-in-time validation**, because surveillance models are always applied to future data. A random split is run only to show how much it inflates the AUROC.
- **Calibration reported separately from discrimination**, because the main failure on transfer is mis-calibration.
- **India.** Three different samples are used; keep them apart:
  - all Indian isolates (n = 1,133, 2004–2016);
  - the Indian test subset (n = 287, 2015–2016 only);
  - the recalibration data (2004–2009, n = 846; 2009 only, n = 58).

## Limitations of this code

- It is retrospective only, on one sponsored surveillance database.
- It does not validate on Indian hospital records and is not prospective.
- It is for population-level surveillance research, not individual treatment decisions.

## Before making this repository public

- [ ] Get IPR Cell clearance.
- [ ] Add a licence, e.g. MIT, as the `LICENSE` file.
- [ ] Archive a release on Zenodo to get a DOI.
- [ ] Put the GitHub link and DOI in the paper's Code availability statement.
- [ ] Confirm that no data file was committed by checking `git log --stat`.
