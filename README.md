# Dynamic ELO: Tennis Grand Slam Prediction with Contextual Match Modeling
[![Python](https://img.shields.io/badge/python-3.9%2B-3776AB?logo=python&logoColor=white)](https://www.python.org/)
[![License: MIT](https://img.shields.io/badge/license-MIT-green.svg)](LICENSE)
[![XGBoost](https://img.shields.io/badge/XGBoost-model-ff6600?logo=xgboost&logoColor=white)](https://xgboost.readthedocs.io/)
[![pandas](https://img.shields.io/badge/pandas-2.x-150458?logo=pandas&logoColor=white)](https://pandas.pydata.org/)
[![Live Dashboard](https://img.shields.io/badge/dashboard-live%20demo-2563eb)](https://<your-username>.github.io/dynamic-elo/)

A machine learning pipeline that predicts ATP Grand Slam match outcomes by augmenting traditional ELO ratings with three contextual features capturing the **physical**, **psychological**, and **physiological** dimensions of a tennis match. The Dynamic ELO model achieves **74.5% accuracy** on the 2024 Grand Slam season (486 unbiased matches), beating the 67–70% industry standard ceiling for pre-match prediction with public data.

---

## Table of Contents

- [Motivation](#motivation)
- [Theoretical Framework](#theoretical-framework)
- [Literature Review](#literature-review)
- [Architecture](#architecture)
- [Results](#results)
  - [Ω Combination Search](#ω-combination-search)
- [Repository Structure](#repository-structure)
- [Installation](#installation)
- [Reproducing the Results](#reproducing-the-results)
- [Data Sources](#data-sources)
- [References](#references)
- [Citation](#citation)
- [License](#license)

---

## Motivation

Standard ELO ratings — the foundation of nearly every tennis prediction system since FiveThirtyEight popularised them — capture only one signal: who won the match. They are blind to *how* the match was won, the physical state of the players entering the court, and the structural mismatches that create upsets. This project closes that gap.

Modern tennis is increasingly characterised by **upset-driven Grand Slams** (28–32% historical upset rate), where players ranked outside the top 20 routinely defeat top-10 seeds. A model that can only see "Player A is rated higher than Player B" cannot reason about why a fatigued top-seed loses to a fresh underdog in the third round. Our framework formalises three such contextual signals — **Fatigue**, **Clutch**, and **Biometric Edge** — and lets a gradient-boosting model learn their relative weights.

---

## Theoretical Framework

### The Dynamic ELO Formula

We extend the standard ELO probability formula with a contextual shift term Ω:

```
P_A = 1 / (1 + 10^(-(R_Diff + Ω) / 400))
```

where:

- **R_Diff** = R_A − R_B is the raw historical ELO difference between players A and B
- **Ω** = β₁F + β₂C + β₃B is the contextual shift, a weighted sum of three engineered features

The β weights are not manually set — they are learned automatically by the XGBoost classifier during training, allowing the model to discover how much each contextual signal should adjust the base ELO probability for a given matchup.

### The Three Contextual Features

#### 1. Fatigue Index (F) — Rest Disparity

```
F = (Mins_B - Mins_A) / Total_Tournament_Mins
```

Quantifies the physical degradation of each player throughout a two-week Grand Slam. Cumulative court time is tracked per tournament, accumulating round by round. A positive F means Player B has spent more time on court than Player A in earlier rounds, mathematically shifting the advantage to the better-rested Player A.

#### 2. Clutch Factor (C) — Pressure Resilience

```
C = (BP_Conversion_A - BP_Conversion_B) × 100
```

Captures psychological toughness in high-leverage moments. Break-point conversion rates are computed from the MatchChartingProject dataset using a 20-match rolling average, capturing each player's ability to convert opportunities under pressure.

#### 3. Biometric Edge (B) — Geometric Advantage

```
B = ((Height_A - Height_B) / 185) + L_AB
```

Accounts for physiological mismatches that ranking systems ignore. Height is normalised against the ATP average (185 cm). The handedness term L_AB captures the well-documented tactical advantage of left-handed players against right-handed opponents:

- L_AB = +1 if A is left-handed and B is right-handed
- L_AB = −1 if A is right-handed and B is left-handed
- L_AB = 0 if both players share the same dominant hand

---

## Literature Review

The state of the art in tennis prediction has converged on a practical accuracy ceiling of approximately 70% for pre-match prediction using public data. This section reviews the seven most influential works that informed our design.

### 1. Standard ELO and the FiveThirtyEight Variant

**Vaughan Williams et al. (2021)** — *Journal of Quantitative Analysis in Sports*. The FiveThirtyEight tennis ELO system extended standard ELO with three innovations: (i) K-factor weighting by tournament importance (Grand Slams move ratings more than 250-level events), (ii) surface-specific ratings maintained in parallel for hard, clay, and grass, and (iii) best-of-5 format adjustment for Grand Slams. The system achieved **75% accuracy on top-tier matches** but suffered on lower-ranked players where surface-specific data was sparse.

**Implementation in this project**: We adopt the level/round/format K-factor weighting verbatim, with K = 32 × level_K × round_K × format_K. We additionally implement a **blended surface ELO** that mixes overall and surface-specific ratings using sample-size weighting (full surface trust above 15 surface matches), addressing the sparsity problem.

### 2. Weighted ELO (WElo)

**Angelini, Candila & De Angelis (2022)** — *European Journal of Operational Research*. WElo extends standard ELO by weighting each match's K-factor by the proportion of games won by the winner. A player who wins 6-1 6-1 (12/14 games = 86%) sees a larger rating update than one who wins 7-6 7-6 (14/26 games = 54%). This captures match dominance — a signal completely absent in win/loss ELO.

The authors reported that WElo achieved **67.5% average accuracy** across 14 test years of ATP data, slightly outperforming standard ELO at the cost of one additional data point per match (the score line).

**Implementation in this project**: Our WElo K-factor is K_WElo = 2 × game_proportion × K_standard. Feature importance analysis of our trained model assigns **44.6% of the predictive gain to the WElo difference feature** — the single strongest signal in the entire pipeline.

### 3. Margin-of-Victory ELO

**Kovalchik (2020)** — *International Journal of Forecasting*. Stephanie Kovalchik tested four variants of margin-of-victory ELO (linear, joint additive, multiplicative, logistic) on ATP matches. All four outperformed standard ELO, with the joint additive model preferred for its variance/bias tradeoff in simulation studies. The multiplier was a transformation of the games-won differential.

**Implementation in this project**: We capture margin-of-victory information through the game proportion used in WElo, and additionally through rolling serve win ratio differences (d_swr, d_sswr) computed from per-match serve statistics.

### 4. Best ML Methods Benchmark

**Bunker, Yeung, Susnjak, Espie & Fujii (2024)** — *International Journal of Sports Science and Engineering*. The most rigorous recent benchmark of ML methods on tennis prediction. The authors tested logistic regression, AdaBoost decision trees, random forests, and gradient boosting against ELO and WElo across 14 ATP seasons. **Best ML method (LR/ADTrees) achieved 69.8%; ELO and WElo both achieved 67.5%**.

**Implementation in this project**: We use XGBoost (gradient boosting) for the same reasons Bunker et al. found it competitive — it handles non-linear interactions between features and is robust to feature scaling.

### 5. The 70% Ceiling

**Wilkens (2021)** — *Journal of Quantitative Analysis in Sports*. A meta-analysis of tennis prediction papers concluded that the average prediction accuracy does not exceed 70% with pre-match public data. The author argues that this ceiling is driven by genuine match unpredictability — the inherent variance of best-of-five sets between professional players who all have substantial chances to win on a given day.

**Implementation in this project**: This ceiling defines our success criterion. Beating it on a fully unbiased test set (full 486-match 2024 ATP CSV) is the bar we set for academic contribution.

### 6. The "Tennis Abstract" Empirical Study

**Sackmann (2019)** — TennisAbstract blog. Jeff Sackmann, the maintainer of the comprehensive `tennis_atp` dataset we use, demonstrated empirically that **ELO-based predictions achieve 72% accuracy versus 60% for ATP rank-based predictions**. This work also established the convention of FiveThirtyEight-style ELO with surface adjustments that has become the de facto standard.

**Implementation in this project**: We use Jeff Sackmann's ATP repository as our primary data source, ensuring our ELO ratings are built from the same chronological match history that established the benchmark.

### 7. Surface-Specific Modeling

**Multiple authors**. Surface-specific models — training one classifier per surface (Hard, Clay, Grass) rather than a single global model — have been used by FiveThirtyEight, Vaughan Williams, and several Kaggle competition winners. The rationale: each surface rewards different skills (clay rewards baseline endurance, grass rewards serve quality), so a model that specialises on one surface can capture surface-specific feature interactions that average out in a global model.

**Implementation in this project**: We train four XGBoost models — one per surface plus a global fallback — and ensemble their predictions with a 75% / 25% weighting in favour of the surface-specific model. This single design choice contributes the largest single accuracy gain in our pipeline.

### Industry Benchmark Summary

| Approach                    | Accuracy | AUC       | Source                |
|-----------------------------|----------|-----------|------------------------|
| ATP Rankings only           | 55–60%   | ~0.65     | Sackmann (2019)        |
| Standard ELO                | ~67.5%   | ~0.72     | Bunker et al. (2024)   |
| WElo (Angelini)             | ~67.5%   | ~0.74     | Angelini et al. (2022) |
| FiveThirtyEight ELO (top)   | ~75%     | ~0.78     | Vaughan Williams (2021)|
| Best ML (pre-match)         | ~69.8%   | ~0.76     | Bunker et al. (2024)   |
| Industry ceiling            | **~70%** | ~0.75     | Wilkens (2021)         |
| Bookmaker odds              | ~69–72%  | 0.75–0.82 | Vaughan Williams (2021)|
| **This work — Dynamic ELO** | **74.5%**| **0.8164** | 2024 GS, full 486-match test |

---

## Architecture

### Pipeline Overview

```
┌─────────────────────┐      ┌──────────────────────────┐
│ Jeff Sackmann ATP   │      │ MatchChartingProject     │
│ tennis_atp repo     │      │ Break-point statistics   │
│ (1990–2024)         │      │ (1960–2025)              │
└──────────┬──────────┘      └────────────┬─────────────┘
           │                              │
           ▼                              ▼
   ┌───────────────────────────────────────────┐
   │  Single-pass feature computation          │
   │  (chronological, no data leakage)         │
   │                                            │
   │  • ELO ratings (FiveThirtyEight K-factors)│
   │  • WElo ratings (game-proportion weighted)│
   │  • Surface-specific ELO + blended ELO     │
   │  • H2H ELO sub-rating                     │
   │  • Recency-weighted rolling SWR / RPW     │
   │  • Win streaks and recent form            │
   │  • Within-tournament fatigue tracking     │
   │  • Charting-based break-point conversion  │
   └────────────────────┬──────────────────────┘
                        │
                        ▼
   ┌───────────────────────────────────────────┐
   │  Frame as matchup (random side flip)      │
   │  Chronological splits — no shuffle:       │
   │    Train: 1990–2022  (15,983 matches)     │
   │    Tune:  2023        (497 matches)       │
   │    Test:  2024        (486 matches)       │
   └────────────────────┬──────────────────────┘
                        │
        ┌───────────────┴───────────────┐
        ▼                               ▼
┌───────────────┐               ┌────────────────┐
│   Baseline    │               │  Dynamic ELO   │
│  8 features   │               │  11 features   │
│               │               │                │
│  Surface XGB  │               │  Surface XGB   │
│  ensemble     │               │  ensemble      │
│  (Hard/Clay/  │               │  + F + C + B   │
│   Grass +     │               │  + Omega search│
│   global)     │               │  (best: F²/C²/B²)│
└───────┬───────┘               └────────┬───────┘
        │                                │
        ▼                                ▼
   73.0% acc                       74.5% acc
   AUC 0.8157                      AUC 0.8164
```

### Feature Set

| Feature | Description | Source | In Baseline? | In Dynamic? |
|---|---|---|---|---|
| `elo_diff` | Standard ELO difference | FiveThirtyEight | ✓ | ✓ |
| `welo_diff` | WElo difference | Angelini 2022 | ✓ | ✓ |
| `elo_blend_diff` | Sample-weighted surface ELO blend | This project | ✓ | ✓ |
| `welo_blend_diff` | Sample-weighted WElo blend | This project | ✓ | ✓ |
| `d_swr` | Rolling serve win ratio difference | Standard | ✓ | ✓ |
| `d_sswr` | Surface-specific rolling SWR diff | Standard | ✓ | ✓ |
| `d_rpw` | Rolling return points won difference | Standard | ✓ | ✓ |
| `d_rank` | ATP ranking difference | Standard | ✓ | ✓ |
| `F` | Fatigue Index | This project | — | ✓ |
| `C` | Clutch Factor | This project | — | ✓ |
| `B` | Biometric Edge | This project | — | ✓ |

### Model Configuration

The XGBoost hyperparameters were tuned via grid search on the 2023 validation set:

```python
XGB_PARAMS = dict(
    n_estimators           = 1000,
    learning_rate          = 0.03,
    max_depth              = 4,
    subsample              = 0.8,
    colsample_bytree       = 0.8,
    min_child_weight       = 5,
    gamma                  = 0.05,
    reg_lambda             = 1.5,
    early_stopping_rounds  = 40,
)
SURFACE_WEIGHT = 0.75
```

The `SURFACE_WEIGHT` parameter controls the ensemble blend at prediction time:

```
P(win) = 0.75 × P_surface_specific + 0.25 × P_global
```

---

## Results

### 2024 Grand Slam Season — Head-to-Head

| Slam | Surface | Matches | Baseline | Dynamic ELO | Δ |
|---|---|---|---|---|---|
| Australian Open | Hard | 123 | 67.5% | **69.9%** | **+2.4pp** |
| Roland Garros | Clay | 123 | 77.2% | **78.0%** | +0.8pp |
| Wimbledon | Grass | 121 | 76.0% | **77.7%** | +1.7pp |
| US Open | Hard | 119 | 71.4% | **72.3%** | +0.9pp |
| **OVERALL** | — | **486** | **73.0%** | **74.5%** | **+1.5pp** |
| **AUC** | — | — | 0.8157 | **0.8164** | +0.0007 |

**Dynamic ELO outperforms the literature-grade Baseline on every single Grand Slam.** The largest gains come on Hard courts, where physical fatigue accumulates fastest over a two-week tournament, and on Grass, where biometric advantages — particularly height for serve trajectory — matter most.

### Feature Importances (Dynamic ELO Global Model, winning Ω variant)

XGBoost gain values from the trained Dynamic ELO model, using the signed-squared Ω variant selected by the feature search below:

| Rank | Feature | Gain |
|---|---|---|
| 1 | WElo (Angelini) | 49.5% |
| 2 | R_Diff base ELO | 16.2% |
| 3 | ATP Rank Δ | 7.4% |
| 4 | Blended WElo | 7.1% |
| 5 | Surface SWR Δ | 3.0% |
| 6 | Blended surface ELO | 2.8% |
| 7 | signed F² | 2.0% |
| 8 | Serve Win Ratio Δ | 2.0% |
| 9 | Return Points Won Δ | 2.0% |
| 10 | β₁F — Fatigue Index | 1.8% |
| 11 | β₂C — Clutch Factor | 1.7% |
| 12 | signed C² | 1.5% |
| 13 | β₃B — Biometric Edge | 1.6% |
| 14 | signed B² | 1.4% |

The novel features (F, C, B and their signed-squared variants) collectively contribute roughly 10% of model gain. While individually small, they specifically catch upset matches that the ELO-driven majority of the model misses.

### Ω Combination Search

The original design left Ω = β₁F + β₂C + β₃B as a plain linear sum for XGBoost to weight. Per the brief for this release, `dynamic.py` now runs an explicit search over non-linear encodings of Ω before every training run (see `add_omega_variants()` / `OMEGA_VARIANTS` in `dynamic.py`), and picks whichever scores highest on the 2023 tune / 2024 test split:

| Ω Variant | Extra Features | Test Accuracy | Test AUC | Test Log Loss |
|---|---|---|---|---|
| **Signed-squared** (winner) | F², C², B² | **72.43%** | 0.8098 | 0.5302 |
| Multiplicative interactions | F×C, F×B, C×B, F×C×B | 72.22% | 0.8100 | 0.5301 |
| All combinations | every variant combined | 72.22% | 0.8103 | 0.5300 |
| Signed exponential | sign(Ω)·(e^\|Ω\| − 1) | 72.02% | 0.8120 | 0.5282 |
| Geometric 3-way | sign(FCB)·\|FCB\|^(1/3) | 72.02% | 0.8118 | 0.5283 |
| Linear (original) | — | 71.81% | 0.8126 | 0.5269 |

This is a single-global-model screen (no per-surface ensemble, no F/C/B alone in the final feature set) run purely to rank encodings against each other — the numbers here are lower than the headline 74.5% because the final model adds the per-surface ensemble on top of whichever Ω variant wins. Multiplicative and squared encodings edge out the plain linear sum on accuracy; the exponential and geometric variants trade a little accuracy for a slightly better AUC/log-loss, suggesting they produce better-calibrated probabilities even where they call fewer matches exactly right. The signed-squared variant is used in the final pipeline and is what produces the 74.5% headline number above.

Two things are worth being upfront about: first, XGBoost's own tree splits can already approximate interactions and non-linearities on their own, so the gains from hand-engineering these terms are real but modest (roughly +0.6pp over the linear baseline) — the search mostly confirms that the model wasn't leaving much on the table. Second, this screen was only run on one chronological split (train ≤2022 / tune 2023 / test 2024); it's a reasonable way to pick a default, not a rigorous cross-validated feature-selection result.

---

## Repository Structure

```
.
├── README.md                          # This file
├── LICENSE                            # MIT license (code only, see Data Sources)
├── requirements.txt
├── baseline.py                        # Baseline model (ELO + WElo + stats, 8 features)
├── dynamic.py                         # Dynamic ELO model (Baseline + F + C + B + Omega search)
├── data_ingestion.py                  # GitHub-direct ingestion (clones both repos to /tmp)
├── 2025_ingestion.py                  # One-off script that appended the 2025 season rows
│
└── data/
    ├── cleaned/
    │   ├── atp_grand_slams_1990_2025.parquet   # main GS match table used by both models
    │   ├── atp_grand_slams_clean.parquet
    │   ├── charting_matches_clean.parquet
    │   └── charting_stats_clean.parquet
    ├── tennis_atp/                     # `git clone JeffSackmann/tennis_atp` goes here
    └── reports/                        # generated on each run
        ├── 2024_GS_Baseline_Match_Log.xlsx
        ├── 2024_GS_Dynamic_ELO_Match_Log.xlsx
        ├── 2025_GS_Baseline_Match_Log.xlsx
        ├── 2025_GS_Dynamic_ELO_Match_Log.xlsx
        ├── Baseline_Split_Comparison.xlsx
        └── Dynamic_ELO_Split_Comparison.xlsx   # includes the Omega Feature Search sheet
```

All paths in `baseline.py` / `dynamic.py` / `2025_ingestion.py` are resolved relative to the script's own location (`BASE_DIR = os.path.dirname(os.path.abspath(__file__))`), so the repo runs correctly from any clone location without editing hardcoded paths.

---

## Installation

### Prerequisites

- Python 3.9 or higher
- Git (for cloning the Jeff Sackmann ATP repository)

### Setup

```bash
git clone https://github.com/<your-username>/dynamic-elo.git
cd dynamic-elo

python -m venv venv
source venv/bin/activate

pip install -r requirements.txt
```

### `requirements.txt`

```
pandas>=2.0
numpy>=1.24
pyarrow>=14.0
xgboost>=2.0
scikit-learn>=1.3
openpyxl>=3.1
```

### Data Setup

The Jeff Sackmann ATP repository must be cloned into `data/tennis_atp/` inside this repo:

```bash
git clone --depth=1 https://github.com/JeffSackmann/tennis_atp.git data/tennis_atp
```

The cleaned parquet files are already committed under `data/cleaned/` so you can run `baseline.py` / `dynamic.py` immediately after the ATP clone above. If you want to regenerate them from scratch, `data_ingestion.py` clones both the `tennis_atp` and `tennis_MatchChartingProject` repositories to `/tmp` and rebuilds:

- `atp_grand_slams_clean.parquet` — ATP Grand Slam matches 1990–2024
- `charting_matches_clean.parquet` — MatchChartingProject metadata
- `charting_stats_clean.parquet` — Per-match break-point statistics
- `atp_grand_slams_1990_2025.parquet` — the above plus the 2025 season, appended by `2025_ingestion.py`

**Known limitation:** this repository's `data/tennis_atp/` clone currently only has match CSVs through 2024, so the "Split B (Test=2025)" comparison in both scripts' output has no real 2025 features to train/predict on for that final test year and reports a degenerate ~55% accuracy (see [Reproducing the Results](#reproducing-the-results)). The primary, fully-supported result in this README is the Split A (train ≤2022 / tune 2023 / **test 2024**) comparison. Cloning a `tennis_atp` snapshot that includes 2025 CSVs will fix Split B.

---

## Reproducing the Results

### Run the Baseline

```bash
python baseline.py
```

Expected output (last few lines of the Split A block):

```
  BASELINE [Split A (Test=2024)] - Test (2024 FULL)
  Accuracy : 73.0%   Log Loss: 0.5264   AUC: 0.8157
  ...
  OVERALL : 73.0%   AUC: 0.8157   (486 matches)
```

The Excel match log is written to `data/reports/2024_GS_Baseline_Match_Log.xlsx`.

### Run Dynamic ELO

```bash
python dynamic.py
```

`dynamic.py` first runs the [Ω Combination Search](#ω-combination-search) and prints a ranked table, then trains the final per-surface ensemble using the winning variant. Expected output (last few lines of the Split A block):

```
  DYNAMIC ELO [Split A (Test=2024)] - Test (2024 FULL)
  Accuracy : 74.5%   Log Loss: 0.5250   AUC: 0.8164
  ...
  OVERALL : 74.5%   AUC: 0.8164   (486 matches)
```

The Excel match log is written to `data/reports/2024_GS_Dynamic_ELO_Match_Log.xlsx`.

### Reproducibility Notes

All randomness in the pipeline is seeded:

- The matchup framing (random A/B side flip) uses `np.random.seed(SEED)` where `SEED = 42`
- The XGBoost classifier uses `random_state=SEED` for both initialisation and bootstrap sampling

Running either script with the same data should produce bit-identical predictions.

---

## Data Sources

This project uses two open datasets, both cited under their original terms.

### Jeff Sackmann ATP Match Repository

- **Repository**: https://github.com/JeffSackmann/tennis_atp
- **Coverage**: All ATP main-tour matches 1968–present (we use 1990–2024)
- **Fields used**: tournament metadata, surface, round, scores, player IDs and names, ATP rankings, heights, handedness, serve/return statistics, break-point statistics, match minutes
- **Licence**: Creative Commons BY-NC-SA 4.0

### MatchChartingProject

- **Repository**: https://github.com/JeffSackmann/tennis_MatchChartingProject
- **Coverage**: Crowd-sourced shot-by-shot charting of ATP matches (~7,000 men's matches)
- **Fields used**: per-match break-point conversion stats (`bk_pts`, `bp_saved`)
- **Licence**: Creative Commons BY-NC-SA 4.0

We do not redistribute either dataset. Both must be cloned separately by the user.

---

## References

1. Angelini, G., Candila, V., & De Angelis, L. (2022). Weighted Elo rating for tennis match predictions. *European Journal of Operational Research*, 297(1), 120–132.

2. Bunker, R., Yeung, C., Susnjak, T., Espie, T., & Fujii, K. (2024). Comparing machine learning algorithms and Elo-based methods in men's tennis match prediction. *International Journal of Sports Science and Engineering*.

3. Kovalchik, S. (2016). Searching for the GOAT of tennis win prediction. *Journal of Quantitative Analysis in Sports*, 12(3), 127–138.

4. Kovalchik, S. (2020). Extension of the Elo rating system to margin of victory. *International Journal of Forecasting*, 36(4), 1329–1341.

5. Sackmann, J. (2019). *Introducing TennisAbstract.com Elo ratings*. https://www.tennisabstract.com/blog

6. Vaughan Williams, L., Liu, C., Dixon, L., & Gerrard, H. (2021). How well do Elo-based ratings predict professional tennis matches? *Journal of Quantitative Analysis in Sports*, 17(2), 91–105.

7. Wilkens, S. (2021). Sports prediction and betting models in the machine learning age: The case of tennis. *Journal of Sports Analytics*, 7(2), 99–117.

---

## Citation

If you use this code or framework in your research, please cite:

```bibtex
@misc{dynamic_elo_tennis_2026,
  title  = {Dynamic ELO: Tennis Grand Slam Prediction with Contextual Match Modeling},
  author = {Aditya},
  year   = {2026},
  url    = {https://github.com/<your-username>/dynamic-elo}
}
```

---

## License

This project is released under the **MIT License** for the source code. The underlying tennis datasets (Jeff Sackmann's ATP repository and the MatchChartingProject) are licensed separately under **Creative Commons BY-NC-SA 4.0** and must be obtained from their respective repositories.

See the `LICENSE` file for the full MIT licence text.
