# CASF2016 / PhysGater Scripts

This folder contains training, inference, feature extraction, and benchmarking scripts.

## What was updated

- All Chinese `#` comments were converted to English placeholder comments.
- Path/config hardcoding is now overridable from CLI using repeated `--set key=value`.
- This applies to both top-level scripts and scripts under `benchmark/` and `feat_extract/`.

## Unified CLI override pattern

All scripts now support:

```bash
python <script>.py --set KEY=VALUE --set OTHER_KEY=VALUE
```

Rules:

- `KEY` must match an existing config/global variable name in that script.
- If the variable exists in a `Config` class (or `AblationConfig`), it is overridden there.
- If the variable exists as a module-level global, it is overridden as well.

## Common examples

### 1) Path A inference

```bash
python physgater_pathA_inference.py \
  --set test_csv_path=/data/new_dekois_ids.csv \
  --set weights_root=/data/PathA_FP_Suppression_V3 \
  --set esm2_root=/data/DEKOIS_ESM2_FEAT \
  --set masif_root=/data/dekois_256_pro_feat \
  --set plif_root=/data/DEKOIS2_PLIF_Flat \
  --set plif_cache=/data/dekois_plif_cache.pkl \
  --set morgan_cache=/data/dekois_morgan_cache.pkl \
  --set output_csv_path=/data/pathA_predictions.csv
```

### 2) PhysGater benchmark inference

```bash
python physgater_inference_benchmark.py \
  --set BENCHMARK_CSV=/data/final_lit_pcba.csv \
  --set BENCHMARK_MORGAN_CACHE=/data/lit_morgan_cache.pkl \
  --set BENCHMARK_PLIF_CACHE=/data/lit_plif_cache.pkl \
  --set BENCHMARK_DATA_ROOT=/data/LIT_PCBA_PLIF_Flat \
  --set ESM2_ROOT=/data/esm2_feat \
  --set MASIF_ROOT=/data/masif_feat \
  --set PATH_A_ROOT=/data/PathA_FP_Suppression_V3 \
  --set PATH_B_ROOT=/data/PathB_Hunter_Corrected_v3 \
  --set OUTPUT_FILE=/data/physgater_litpcba_results.csv
```

### 3) Plot scripts

```bash
python benchmark/plot_family_performance.py \
  --set DATA_PATHS="{'DUD-E':'/data/DUD-E_unified_comparison.csv','DEKOIS':'/data/DEKOIS2_unified_comparison.csv'}" \
  --set OUTPUT_DIR=/data/bench_plots
```

Note: for complex dict/list values, prefer editing the script directly if shell quoting is inconvenient.

## Script groups

### Top-level scripts

- `pathA_final_masif_model.py`: Path A training pipeline.
- `pathB_final_masif_model_V2.py`: Path B training pipeline.
- `physgater_ablation.py`: ablation training/evaluation workflow.
- `physgater_dual_inference.py`: two-stage inference cascade.
- `physgater_inference_benchmark.py`: benchmark inference pipeline.
- `physgater_pathA_inference.py`: Path A only inference.

### Feature extraction (`feat_extract/`)

- `esm2_feat.py`
- `final_label.py`
- `final_morgan.py`
- `final_plif.py`

### Benchmark/plot utilities (`benchmark/`)

- `ablation_study_plot.py`
- `merge_benchmarks.py`
- `merge_modality_plots.py`
- `plot_10fold_panel_with_legend.py`
- `plot_10fold_weights_labeled.py`
- `plot_ablation_rerank.py`
- `plot_family_performance.py`
- `plot_lambda_sensitivity.py`
- `plot_modality_weights.py`
- `plot_screening_efficiency.py`

## Validation

A syntax compile check has been run across all Python files in this folder tree.
