#!/usr/bin/env bash
# Reproduce every number in the PathoWatch ATLAS MDR paper from the raw ATLAS file.
# Usage:  bash run_all.sh data/Open_Atlas_Reuse_Data.xlsx
# Output: work/ (intermediate tables), results/ (JSON results), paper/ieee/build/ (filled LaTeX)
set -euo pipefail

XLSX="${1:-data/Open_Atlas_Reuse_Data.xlsx}"
[ -f "$XLSX" ] || { echo "ATLAS file not found: $XLSX (see data/README.md)"; exit 1; }
mkdir -p work results paper/ieee/build/figures
PY="python3 -I"
t0=$(date +%s)
step() { echo; echo "[$(( $(date +%s) - t0 ))s] $*"; }

step "1/8 Convert the Excel release to CSV"
$PY src/xlsx2csv.py "$XLSX" work/atlas.csv

step "2/8 Label MDR status (species-specific Magiorakos categories)"
$PY src/build_mdr.py work/atlas.csv work/mdr_base.csv

step "3/8 Cohort description (Tables II-III)"
$PY src/descriptives.py work/mdr_base.csv results

step "4/8 Temporal validation of four classifiers, importance, transfer (Tables IV-V, VIII-IX; Figs. 4-5)"
$PY src/run_experiments.py work/mdr_base.csv results | tee results/run.log

step "5/8 Adjusted odds ratios, adjusted trends, intercept recalibration (Table VIII, Sec. IV-B)"
$PY src/run_extra.py work/mdr_base.csv results > results/run_extra.log

step "6/8 Extended analyses: ROC, calibration, subgroups, learning curve, sensitivity, recalibration, LORO, funnel, forecasting"
$PY src/run_extended.py work/mdr_base.csv results work/atlas.csv 2>&1 | tee results/run_ext.log

step "7/8 SVM and baseline calibration metrics (Table IV)"
$PY src/svm_cal.py work/mdr_base.csv results

step "8/8 Insert the numbers into the IEEE paper and figures"
$PY paper/fill_paper_numbers.py paper/ieee/main_template.tex results paper/ieee/build/main.tex
for f in paper/ieee/figures/*.tex; do
  $PY paper/fill_paper_numbers.py "$f" results "paper/ieee/build/figures/$(basename "$f")" > /dev/null
done
rm -f paper/ieee/build/figures/*_values.json
cp paper/ieee/references.bib paper/ieee/build/

step "Compare new results with the expected results used in the paper"
$PY src/compare_results.py results results/expected
echo "Done in $(( $(date +%s) - t0 )) s. Upload paper/ieee/build/ to Overleaf, or compile with pdflatex/bibtex."
