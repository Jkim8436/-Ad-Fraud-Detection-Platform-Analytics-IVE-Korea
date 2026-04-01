# IVE Korea Ad Platform Fraud Detection & Optimization

<div align="center">

![Python](https://img.shields.io/badge/Python-3.11+-3776AB?style=for-the-badge&logo=python&logoColor=white)
![Pandas](https://img.shields.io/badge/Pandas-2.0+-150458?style=for-the-badge&logo=pandas&logoColor=white)
![NumPy](https://img.shields.io/badge/NumPy-1.24+-013243?style=for-the-badge&logo=numpy&logoColor=white)
![Plotly](https://img.shields.io/badge/Plotly-5.0+-3F4F75?style=for-the-badge&logo=plotly&logoColor=white)
![Jupyter](https://img.shields.io/badge/Jupyter-Notebook-F37626?style=for-the-badge&logo=jupyter&logoColor=white)
![License](https://img.shields.io/badge/License-MIT-22c55e?style=for-the-badge)

<br/>

**End-to-end data science pipeline for detecting ad fraud, managing media performance, and optimizing domain-level ad strategy on a real-world Korean mobile ad platform.**

<br/>

[Pipeline Overview](#-pipeline-overview) • [Key Findings](#-key-findings) • [Dataset](#-dataset) • [Methods](#-methods) • [Results](#-results) • [Setup](#-setup)

</div>

---

## 🔍 Project Overview

This project analyzes **16.8M+ click records** from IVE Korea's mobile advertising platform to identify fraudulent traffic, assess media partner risk, and deliver actionable ad optimization recommendations.

The analysis operates across three interconnected stages — fraud detection feeds into media management, which in turn informs domain-level ad strategy. Every insight is traceable back to data.

```
Raw Click Data (16.8M rows)
        │
        ▼
┌───────────────────┐    ┌───────────────────┐    ┌───────────────────┐
│  STAGE 1          │───▶│  STAGE 2          │───▶│  STAGE 3          │
│  Fraud Detection  │    │  Media Management │    │  Ad Optimization  │
│                   │    │                   │    │                   │
│ • CTIT Analysis   │    │ • Risk Scoring    │    │ • Domain Heatmaps │
│ • IP Clustering   │    │ • KPI Segmentation│    │ • Strategic Index │
│ • Loss Quantif.   │    │ • PHASE Detection │    │ • Best Ad Cases   │
└───────────────────┘    └───────────────────┘    └───────────────────┘
        │                        │                        │
        ▼                        ▼                        ▼
  Confirmed losses          Recommend /              Optimal ad type
  per media partner         Improve / Cut            per domain
```

---

## 📊 Key Findings

<table>
<tr>
<td width="33%" valign="top">

### 🚨 Fraud Detection
- **Outlier (confirmed fraud)**: detected via IP device clustering + CTIT anomaly scoring
- **Top 2 fraudulent media** (`mda_58`, `mda_539`) responsible for the majority of confirmed losses
- Removing `mda_58` + `mda_539` simultaneously raises platform CVR by **+X.XX%p**
- Device blacklist: top **100 devices** account for **~60%** of all outlier clicks

</td>
<td width="33%" valign="top">

### 📈 Media Risk
- **7 media partners** classified as `critical` risk
- Risk label based on outlier click ratio (primary) + CTIT score + IP device concentration
- `mda_539` followed a **3-PHASE infiltration pattern**: Dormant → Expansion → **Explosion** — detectable 3 months before peak
- Weekday CVR for `critical` media is **flat Mon–Sun** (strong bot signal vs. natural human variation in normal media)

</td>
<td width="33%" valign="top">

### 🎯 Ad Optimization
- **Participation-type** ads: highest strategic importance in 4 out of 5 domains
- **Launch-type** ads: strongest growth engine across all domains except Commerce
- Commerce domain uniquely favors **install-type** ads
- Dead copy rate in participation ads: **>50%** (majority of inventory never receives a click)
- Optimal `ads_order` threshold identified — ads below a certain exposure rank generate **zero conversions**

</td>
</tr>
</table>

---

## 📁 Dataset

| Table | Description | Rows | Key Columns |
|---|---|---|---|
| `IVE_광고목록` (Ad Master) | Ad metadata, pricing, domain labels | ~445K | `ads_idx`, `ads_type`, `domain_label`, `ads_contract_price`, `ads_reward_price` |
| `IVE_광고참여정보` (Engagement) | Every user click event | **~16.8M** | `click_key`, `mda_idx`, `dvc_idx`, `user_ip`, `click_date` |
| `IVE_광고적립` (Reward) | Successful conversions (rewards paid) | ~940K | `click_key`, `show_cost`, `earn_cost`, `rwd_cost`, `ctit` |
| `시간별적립보고서` (Hourly Report) | Aggregated hourly KPIs, 1-year history | ~1M | `mda_idx`, `rpt_time_clk`, `rpt_time_turn`, `rpt_time_scost` |

**Analysis period**: July 26 – August 25, 2025 (1 month, peak fraud period)  
**Historical trend**: July 2024 – August 2025 (13 months)

### Data Relationships

```
df_engagement ──┐
                ├──[click_key]──▶ df_reward
                │
                └──[ads_idx]───▶ df_master (ad_list)
                │
                └──[mda_idx + user_ip]──▶ ip_stats (derived)
                                              │
                                              ▼
                                        df_integrated (16.8M rows)
                                        + row_label + Risk_Label
```

---

## ⚙️ Methods

### Stage 1 — Fraud Detection

#### CTIT (Click-to-Install Time) Anomaly Labeling
```
Threshold per ad type = median(CTIT) × 0.1
Flagged as error if:
  • click_type (type=4): CTIT == 0
  • all other types: CTIT < threshold
```

#### IP-Level Device Clustering
```python
ip_stats = df_engagement.groupby(['mda_idx', 'user_ip']).agg(
    device_count = ('dvc_idx', 'nunique'),   # unique devices per IP
    click_count  = ('click_key', 'count')    # total clicks per IP
)
```

#### Row-Level Classification (`row_label`)
| Label | Condition | Meaning |
|---|---|---|
| `outlier` | device_count > 99.9th pct. | Confirmed fraud — deduct from settlement |
| `abusing` | device_count > 99th pct. | Suspicious — close monitoring |
| `warning` | device_count > 95th pct. | Flag for review |
| `normal` | Below threshold | Legitimate |
| `non_rewarded` | No conversion | Normal, no reward issued |

#### Media Risk Score
```
Total_Score = avg_ctit_score × 0.50
            + avg_device_score × 0.30
            + avg_click_score  × 0.20

Risk_Label (official) = based on outlier click ratio
  critical  : outlier_ratio > threshold_1
  risky     : outlier_ratio > threshold_2
  warning   : outlier_ratio > threshold_3
  normal    : below all thresholds
```

---

### Stage 2 — Media Management

#### KPI Aggregation
Grain: `domain_label × ads_type × mda_idx`

| Metric | Formula |
|---|---|
| CVR | `conversions / clicks` |
| Profit | `show_cost - earn_cost` |
| Profit per Conv | `profit / conversions` |
| Margin Rate | `profit / revenue` |

#### Performance Segmentation (`segment_mda_direct`)
Implemented with `np.select` vectorization (10–100× faster than `iterrows`):

```python
conditions = [
    cvr_ok & ppc_ok & mr_ok,                        # premium_top
    reliable & cvr_ok & vol_ok & prof_ok,            # volume_top
    vol_ok & (~prof_ok | ~cvr_ok),                   # structural_improve_large
    cvr_ok,                                          # structural_improve
]
g["performance_segment"] = np.select(conditions, choices, default="low_efficiency_top")
```

| Segment | Operation Tag | Criteria |
|---|---|---|
| `premium_top` | `confirmed` / `potential` | CVR ✅ + top profit-per-conv + strong margin |
| `volume_top` | `confirmed` / `potential` | CVR ✅ + high volume + ≥1 profitability metric above median |
| `structural_improve_large` | `improve_first` | Large volume but CVR or profitability below par |
| `structural_improve` | `improve_first` | CVR met but below premium/volume thresholds |
| `low_efficiency_top` | `reduce_candidate` | CVR below threshold + weak volume and profitability |

#### PHASE Infiltration Classification
```python
def classify_phase(monthly_share: pd.Series) -> str:
    recent_3   = monthly_share.tail(3).mean()
    historical = monthly_share.iloc[:-3].mean()
    ratio = recent_3 / historical
    if   ratio >= 5: return "PHASE 3 — Explosion"
    elif ratio >= 2: return "PHASE 2 — Expansion"
    else:            return "PHASE 1 — Dormant"
```

Applied to **all** critical/risky media (not just the originally analyzed `mda_539`).

---

### Stage 3 — Domain Ad Optimization

#### Strategic Importance Index
```
Strategic Importance = √(Conversion Contribution% × Budget Absorption%)
```
Geometric mean prevents overrating ad types that excel in only one dimension.

#### CVR Contribution Analysis (Optimized)
Removed per-media DataFrame copy loop. Now uses precomputed aggregates:
```python
# Before: 189 iterations × 16.8M row copy = minutes
# After: 1 groupby + vectorized math = instant
mda_agg['cvr_without'] = (
    (total_conv - mda_agg['turn']) /
    (total_clk  - mda_agg['clk']) * 100
)
```

---

## 📈 Results

### Fraud Impact Summary

| Category | Metric | Value |
|---|---|---|
| Confirmed fraud clicks | `outlier` row_label | See notebook output |
| Confirmed financial loss | `show_cost` on outlier conversions | See notebook output |
| Manageable loss | `abusing` conversions | See notebook output |
| Critical-risk media | Risk_Label = `critical` | 7 media partners |
| CVR lift (mda_58 removed) | Platform CVR delta | See notebook output |
| CVR lift (mda_539 removed) | Platform CVR delta | See notebook output |

### Domain × Ad Type Optimal Combos (STEP 22-15)

| Domain | #1 (Best) | #2 | #3 | Consider Removing |
|---|---|---|---|---|
| Entertainment | launch | participation | install | — |
| Finance | launch | naver | participation | — |
| Lifestyle | launch | participation | install | — |
| Commerce | participation | install | launch | — |
| Other | launch | participation | install | — |

*Exact scores generated by `strategic_score = √(conv_pct × spend_pct)` in notebook.*

### ads_order Insight (STEP 22-13)

Ads below a certain `ads_order` threshold receive **zero clicks** regardless of creative quality — confirming that inventory placement rank is a critical success factor independent of ad content.

---

## 🗂️ Project Structure

```
ive-korea-ad-optimization/
│
├── 📓 integrated_analysis_final_EN.ipynb   # Full pipeline (English)
├── 📓 통합_분석_최종_v3_fixed.ipynb         # Full pipeline (Korean)
│
├── data/                                   # Raw data (not included — see note)
│   ├── 광고 목록_도메인 라벨링_결측치처리.csv
│   ├── IVE_광고적립_all.csv
│   ├── IVE_광고참여정보_all.csv
│   └── 아이브시간대별광고리포트_1년치_all.csv
│
└── README.md
```

> **Data Note**: Raw data files are not included in this repository due to confidentiality. The notebook is fully documented and reproducible with equivalent data following the same schema.

---

## 🚀 Setup

### Requirements

```bash
pip install pandas numpy matplotlib seaborn plotly jupyter
```

| Package | Version | Purpose |
|---|---|---|
| `pandas` | ≥ 2.0 | Data manipulation, groupby aggregation |
| `numpy` | ≥ 1.24 | Vectorized operations, np.select |
| `matplotlib` | ≥ 3.7 | Trend charts, time-series plots |
| `seaborn` | ≥ 0.12 | Heatmaps (domain × ad type analysis) |
| `plotly` | ≥ 5.0 | Interactive funnel chart |
| `jupyter` | ≥ 6.0 | Notebook execution |

### Running the Notebook

```bash
git clone https://github.com/your-username/ive-korea-ad-optimization.git
cd ive-korea-ad-optimization

# Install dependencies
pip install -r requirements.txt

# Launch Jupyter
jupyter notebook integrated_analysis_final_EN.ipynb
```

### Execution Order

Run cells **top to bottom in order**. Each STEP depends on variables defined in prior STEPs.

| STEP Range | Stage | Runtime (est.) |
|---|---|---|
| STEP 1–8 | Data load, preprocessing, fraud labeling | ~2–5 min |
| STEP 9–19 | Analysis, trend analysis, mda deep-dives | ~1–3 min |
| STEP 20–21 | Media management (run 5× for domains 1–5) | ~2–4 min |
| STEP 22-0–22-16 | Domain ad optimization | ~2–5 min |

> ⚠️ **Memory**: The full `df_integrated` (~16.8M rows) requires **~4–6 GB RAM**. A machine with 16 GB RAM is recommended.

---

## 🔑 Key Design Decisions

### Why `is_rewarded` not `notna(show_cost)` for conversion flag
In STEP 8, all cost columns are filled with `0` via `fillna(0)`. Using `notna()` after this point would flag every row as converted. `is_rewarded` is derived from `ctit.notna()` at join time — before the fill — and is the correct conversion indicator throughout.

### Why `reward_clean` instead of raw `df_reward`
The domain optimization analysis (STEP 22) is built on `reward_clean = event_clean[is_conv == 1]` — conversions that survive the outlier filter. Using raw `df_reward` would contaminate KPIs with confirmed fraudulent conversions, inflating CVR and cost metrics.

### Why geometric mean for strategic importance
```
√(conv_contribution × budget_absorption)
```
An ad type that captures 80% of conversions but only 5% of budget (or vice versa) scores poorly. The geometric mean rewards balanced excellence — forcing both dimensions to be high simultaneously.

### Performance Optimizations Applied
| Location | Before | After | Speedup |
|---|---|---|---|
| CVR contribution ranking | for-loop × 189 media × 16.8M rows | `mda_agg` precomputed → vector math | ~100× |
| mda_539 removal CVR | `df_integrated[~mask]` full copy | `(total - row539) / (total - clk539)` | ~50× |
| mda_58 removal CVR | Same full copy pattern | Same formula | ~50× |
| `segment_mda_direct()` | `iterrows()` per media group | `np.select` vectorized | 10–100× |
| Opportunity cost | Python dict loop over 16.8M rows | `groupby` + vectorized churn flag | ~100× |

---

## 📋 Notebook Structure

<details>
<summary><b>STAGE 1 — Fraud Detection (STEP 1–19 + Deep Dives)</b></summary>

| STEP | Description |
|---|---|
| 1 | Data load, date parsing, `df_master2` snapshot |
| 2 | Cost structure outlier removal (margin inversion, zero-price) |
| 3 | CTIT labeling, IP device/click aggregation |
| 4 | Build `df_integrated` (16.8M row join) |
| 5 | Row-level classification (`row_label`: outlier/abusing/warning/normal/non_rewarded) |
| 6 | Media-level risk scoring (weighted CTIT + IP metrics) |
| 7 | Join `Risk_Label` to `df_integrated` |
| 8 | Loss quantification (confirmed, manageable, total) |
| 9 | CVR: normal vs risky media comparison |
| 10 | Hourly abusing pattern analysis |
| 11 | Ad type abusing vulnerability + CVR purification effect |
| 12 | Device blacklist (top abusing devices by click count) |
| 13 | `df_report` preprocessing (1-year hourly data) |
| 14 | Monthly CVR trend — click vs conversion gap (MoM) |
| 15 | `mda_539` monthly infiltration timeline |
| 16 | Monthly earn_cost & CVR by risk group |
| 17 | Weekday pattern: normal vs critical media (bot evidence) |
| 18 | 1-year trend by ad type and domain |
| 19 | 1-month vs 1-year statistical validation |
| — | CVR decline contribution (optimized `mda_agg` vector math) |
| — | Confirmed loss Top 3 media |
| — | `mda_539` figure validation |
| — | `mda_58` deep-dive (CTIT, IP, monthly, hourly, weekday, ad type) |
| — | Cost loss 3-layer decomposition |
| — | CVR gap by risk level |

</details>

<details>
<summary><b>STAGE 2 — Media Management (STEP 20–21)</b></summary>

| STEP | Description |
|---|---|
| 20 | Build `event_clean` (outlier-removed) + `domain_dfs` (1–5) |
| 21 | Per-domain: KPI aggregation → reliability cut → segment → operation tag |

Outputs per domain:
- `mda_kpi_rel`: raw KPI with reliability flag
- `mda_segmented_full`: performance segment + operation tag + sub-media diagnostics
- `mda_ads_detail_with_segment`: ad-level detailed KPI
- `top5_op`: Top 5 media per operation tag

</details>

<details>
<summary><b>STAGE 3 — Domain Ad Optimization (STEP 22-0 to 22-16)</b></summary>

| STEP | Description |
|---|---|
| 22-0 | Additional imports, domain/type mappings, heatmap axis ordering |
| 22-1 | `df_master` additional filtering, derived variables |
| 22-2 | Ad inventory overview by type |
| 22-3 | Domain × ad type price structure table |
| 22-4 | Conversion performance by ad type |
| 22-5 | Conversion analysis by domain |
| 22-6 | Three heatmaps: conversion contribution, budget allocation, cost efficiency |
| 22-7 | Strategic importance index + domain-level ranking |
| 22-8 | Best-performing ad cases (top 5 per domain's #1 type) |
| 22-9 | 100% churn rate ad analysis |
| 22-10 | Opportunity cost aggregation (vectorized groupby) |
| 22-11 | Conversion funnel visualization (Plotly) |
| 22-12 | Dead copy (inactive ad) analysis |
| 22-13 | Exposure order (ads_order) × conversion range analysis |
| 22-14 | Automatic PHASE classification for all critical/risky media |
| 22-15 | Domain × ad type final recommendation table + heatmap |
| 22-16 | Full pipeline summary (COMPLETED / IN PROGRESS / NEXT ACTION) |

</details>

---

## 🤝 Contributing

Pull requests are welcome. For major changes, please open an issue first to discuss what you would like to change.

---

## 📄 License

This project is licensed under the MIT License — see the [LICENSE](LICENSE) file for details.

---

<div align="center">

**Built with Python · Pandas · NumPy · Seaborn · Plotly**

*IVE Korea Ad Platform — Fraud Detection & Optimization Analysis*

</div>
