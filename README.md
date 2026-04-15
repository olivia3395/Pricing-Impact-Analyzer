<div align="center">

# 🚦 Congestion Pricing Impact Analyzer
### *Causal Inference on NYC Urban Mobility*

**The first U.S. congestion pricing policy — analyzed with a full causal inference stack.**

[![Python](https://img.shields.io/badge/Python-3.11+-3776AB?style=flat-square&logo=python&logoColor=white)](https://python.org)
[![DuckDB](https://img.shields.io/badge/DuckDB-0.10-FFC832?style=flat-square&logo=duckdb&logoColor=black)](https://duckdb.org)
[![EconML](https://img.shields.io/badge/EconML-Causal%20Forest-00A36C?style=flat-square)](https://econml.azurewebsites.net/)
[![Streamlit](https://img.shields.io/badge/Streamlit-Dashboard-FF4B4B?style=flat-square&logo=streamlit&logoColor=white)](https://streamlit.io)
[![Data: NYC TLC](https://img.shields.io/badge/Data-NYC%20TLC%2012M%2B%20trips-0055A4?style=flat-square)](https://www.nyc.gov/site/tlc/about/tlc-trip-record-data.page)

<br/>

> On **January 5, 2025**, NYC enacted the MTA Congestion Relief Zone toll — the **first congestion pricing program in U.S. history**.
> This project uses NYC TLC trip records and a full **L1→L5b causal inference stack** to answer:
> *how did this policy reshape mobility demand, fare structure, and trip geography?*

</div>

---

## 🎯 Motivation

This project directly maps to the analytical challenges at **Waymo, WeRide, and Uber**: understanding how regulatory shocks reshape demand geography, fare structure, and trip composition at scale.

- 📍 **Policy shock:** ~60 Manhattan zones below 60th St began paying the MTA toll
- 📊 **Data scope:** 12M+ Yellow / Green / HVFHV trip records, Jul 2024 → Jun 2025
- 🧠 **Methods:** Full causal inference stack from TWFE DiD through Causal Forest (CATE)

---

## 🔬 Causal Inference Stack

| Layer | Estimator | Purpose |
|:---:|---|---|
| **L1** | Two-Way Fixed Effects DiD | ATE baseline; exposes parallel-trends assumption |
| **L2** | Event Study + Callaway-Sant'Anna | Dynamic effects; honest pre-trend test; no negative-weighting |
| **L3** | Continuous Treatment DiD | Dose-response curve using `cbd_congestion_fee` as dose |
| **L4** | Synthetic DiD *(Arkhangelsky 2021)* | Reweights control units; no parallel-trends required |
| **L5a** | Double/Debiased ML *(Chernozhukov 2018)* | ML-debiased ATE via cross-fitting; controls high-dim confounders |
| **L5b** | Causal Forest *(Wager & Athey 2018)* | Zone-level CATE; feature importance; policy tree; RATE curve |

---

## 🚀 Quick Start

```bash
# 1 ── Environment setup
chmod +x scripts/bootstrap.sh && ./scripts/bootstrap.sh
conda activate mobility_policy

# 2 ── Run full pipeline end-to-end
./scripts/run_all.sh

# 3 ── Skip download (data already cached)
./scripts/run_all.sh --skip-download

# 4 ── Run specific pipeline steps only
python src/run_pipeline.py --steps 5,6,7,10,11

# 5 ── Launch interactive dashboard
streamlit run src/dashboard/app.py
```

---

## 📁 Repository Structure

```
mobility_policy_shock_analytics/
│
├── 📄 configs/default.yaml            # Dates, zones, ML hyperparameters
│
├── 📜 scripts/
│   ├── bootstrap.sh                   # One-command environment setup
│   └── run_all.sh                     # End-to-end pipeline with logging
│
├── 🧩 src/
│   ├── data/
│   │   ├── download_tlc.py            # Download TLC parquets from AWS S3
│   │   ├── preprocess.py              # Unify Yellow/Green/HVFHV schemas + QC
│   │   └── build_zone_panel.py        # DuckDB aggregation → zone × day panel
│   │
│   ├── geospatial/
│   │   ├── zone_lookup.py             # Zone → borough / geometry / distance to CBD
│   │   └── cbd_mapping.py             # Treatment: binary, ring, continuous dose
│   │
│   ├── features/
│   │   └── build_features.py          # Lags, rolling stats, calendar, zone statics
│   │
│   ├── causal/
│   │   ├── run_did.py                 # L1 TWFE + L2 Event Study + CS-DiD
│   │   ├── run_continuous_did.py      # L3 Continuous treatment + dose-response
│   │   ├── run_sdid.py                # L4 Synthetic DiD (pure Python)
│   │   ├── run_dml.py                 # L5a Double ML with cross-fitting
│   │   ├── run_causal_forest.py       # L5b Causal Forest: CATE, BLP, RATE
│   │   └── robustness.py             # R1–R7: placebo, anticipation, LOCO…
│   │
│   ├── evaluation/
│   │   ├── plots.py                   # F1–F10: event study, CATE map, etc.
│   │   └── tables.py                  # T1–T4: LaTeX + CSV result tables
│   │
│   ├── dashboard/app.py               # Streamlit interactive dashboard
│   └── run_pipeline.py                # Master orchestrator (steps 0–14)
│
└── 📓 notebooks/
    ├── 01_eda.ipynb                   # Data exploration, trends, fare composition
    └── 02_causal_analysis.ipynb       # Full causal walkthrough L1→L5b
```

---

## 📦 Data Sources

| Source | Description | Period |
|---|---|:---:|
| NYC TLC Yellow / Green / HVFHV | Trip records; includes `cbd_congestion_fee` field (2025+) | Jul 2024 → Jun 2025 |
| TLC Zone Lookup CSV | `LocationID` → Borough / Zone name | Static |
| TLC Zone Shapefile | GeoJSON for choropleth mapping | Static |

**Treatment definitions:**

| Type | Definition |
|---|---|
| **Binary** | Zone inside MTA Congestion Relief Zone (~60 zones) |
| **Continuous dose** | `avg_cbd_fee` per zone-day — used in L3 dose-response |
| **Ring** | CBD core / adjacent buffer / far control — used for heterogeneity analysis |

---

## 📊 Outputs

<details>
<summary><strong>📈 Figures (F1–F10)</strong></summary>

| File | Description |
|---|---|
| `F1_event_study.png` | Dynamic effects + pre-trend falsification test |
| `F2_cs_did.png` | Callaway-Sant'Anna ATT(g,t) estimates |
| `F3_dose_response.png` | Continuous dose-response curve |
| `F5_dml_results.png` | DML multi-outcome coefficients with CIs |
| `F6_cate_map.png` | Zone-level CATE choropleth map |
| `F9_rate_curve.png` | RATE targeting curve |
| `F10_robustness.png` | Forest plot of all 7 robustness check families |

</details>

<details>
<summary><strong>📋 Tables (T1–T4)</strong></summary>

| File | Description |
|---|---|
| `T1_main_did.{csv,tex}` | L1 / L2 / L4 estimator comparison |
| `T2_dml.{csv,tex}` | DML results across all outcome metrics |
| `T4_robustness.{csv,tex}` | 7-family robustness check summary |

</details>

---

## 💡 Why Not A/B Test?

Congestion pricing is a **city-wide simultaneous policy** — no zones were randomly assigned to treatment vs. control. This is a classic observational setting.

| Setting | Gold Standard |
|---|---|
| Randomized (e.g., product A/B test) | A/B Testing / RCT |
| Observational (e.g., city-wide policy shock) | **Double ML + Causal Forest** ✅ |

These tools are **complementary, not interchangeable**. The right response to "no randomization" is not to force fake A/B framing — it is to use the correct quasi-experimental estimators, which this project implements in full.

---

## 🏙️ Business Implications

| # | Insight | Application |
|:---:|---|---|
| 1 | **Fleet staging** | If CBD trip counts decline, robotaxi staging should shift to CBD periphery and high-substitution outer boroughs |
| 2 | **Demand elasticity** | L3 dose-response curve directly quantifies elasticity w.r.t. toll cost — useful for dynamic pricing and fee pass-through modeling |
| 3 | **Service expansion** | Borough-level substitution analysis identifies demand growth zones, signaling where to expand capacity |
| 4 | **Time-of-day targeting** | CATE heterogeneity by `peak_share` reveals asymmetric peak vs. off-peak elasticity |
| 5 | **Policy scenario modeling** | Pipeline generalizes to any future city-level shock: parking reform, bridge tolls, EV mandates |

---

## 📝 Resume Bullets

```
• Built end-to-end causal inference pipeline (L1–L5b) on 12M+ NYC TLC trips to
  quantify mobility demand shifts from the 2025 MTA congestion pricing shock

• Implemented Double/Debiased ML (Chernozhukov 2018) with 5-fold cross-fitting
  and Random Forest nuisance learners to control high-dimensional zone-temporal
  confounders; estimated ATE robust to confounding across 5 outcome metrics

• Estimated conditional average treatment effects (CATE) via Causal Forest
  (econml/GRF); produced zone-level CATE maps, feature importance rankings,
  interpretable depth-2 policy trees, and RATE targeting curves

• Implemented Synthetic DiD estimator (Arkhangelsky 2021) in pure Python with
  placebo bootstrap SE; confirmed L1 parallel trends assumption empirically

• Validated all results via 7 robustness check families (placebo dates,
  anticipation, bandwidth, leave-one-borough-out, DML learner sensitivity)
```

---

<div align="center">

Made with 🗽 using NYC TLC open data · Methods from Chernozhukov, Arkhangelsky, Wager & Athey

</div>
