# IVE Korea — Reward Ad Platform: Anomaly Detection & Risk Analytics

End-to-end analysis of **16.8M click events** and **14 months of hourly performance data** from a
Korean reward-based mobile ad platform: a hypothesis-driven fraud detection pipeline, media-level
risk aggregation, financial exposure quantification, and an out-of-time backtest of whether the
platform's largest traffic anomaly could have been caught before it became obvious in hindsight.

**This repository is the author's individual contribution** (data integration through fraud
detection and validation). A teammate-built media operations and domain ad-optimization layer is
included for pipeline continuity and documented more lightly, since it isn't the author's work.

---

## Why this project is framed the way it is

A rule-based anomaly detector on unlabeled traffic can produce a number like "6.39M fraudulent
clicks, ₩260M in losses" that *sounds* authoritative but overstates what the data actually
supports. There is no independently verified invalid-traffic label here — no settlement-rejection
record, no manual fraud confirmation. Every claim in this project is calibrated to that:

| Avoided | Used instead |
|---|---|
| Confirmed fraud | High-confidence anomaly |
| Total loss | Financial exposure |
| CVR recovery | Remaining low-risk traffic CVR |
| Bot evidence | Behavior consistent with automation |

The payoff isn't just more careful wording — it's that the analysis had to survive being checked.
Two structural bugs (a row-classification priority bug, and a domain-relabeling fix that wasn't
applied at the source) were only caught because real re-runs against the full dataset produced
numbers that didn't match what the code should have produced. A late-stage retrospective check
also invalidated one of the project's own earlier claims (`ads_order` correlating with ad
inactivity) once the underlying filter was verified stable across independent runs. Being wrong
and catching it is part of what this repo demonstrates.

---

## Data

Raw files are commercial and not included. Expected under `./data/`:

| File | Table | Grain |
|---|---|---|
| `ad_catalog.csv` | Ad master | one ad campaign (`ads_idx`) |
| `ad_engagement.csv` | Click/participation log | one click (`click_key`) |
| `ad_rewards.csv` | Conversion log | one rewarded event (`click_key`, subset of engagement) |
| `hourly_report_1yr.csv` | 14-month performance report | one (date, hour, ad, media) bucket |

Click/reward/catalog data covers **2025-07-26 to 2025-08-25** (31 days); the hourly report covers
**2024-07-27 to 2025-08-29** (14 months), used for historical validation.

**Cost structure** (who pays whom): `show_cost` (advertiser billing) ≥ `adv_cost` (platform take)
≥ `earn_cost` (media payout) ≥ `rwd_cost` (user reward) ≥ 0. Rows violating this ordering are
data-integrity errors, removed before any anomaly analysis.

---

## Methodology

### 1. Data quality, before anything else

- Cost-structure ordering enforced on every table; explicit non-negativity checks
- `click_key` / `ads_idx` uniqueness asserted, `merge(validate=...)` used on every join — a silent
  row explosion is the easiest way for a 16M-row merge pipeline to quietly produce wrong numbers
- The ad catalog's `9999-12-31` "no defined end date" sentinel (96.7% of the catalog) resolved to
  a real date at the string level before parsing, rather than relying on a Timestamp-range
  workaround that can silently produce `NaT` on some pandas versions
- Conversion status defined by reward-table membership, not by whether `ctit` happens to be
  non-null — a structurally safer definition even though this dataset never actually triggers the
  difference between the two

### 2. Five hypotheses, tested explicitly

| | Signal | Logic |
|---|---|---|
| H1 | Speed | A conversion faster than ~10% of its ad type's median CTIT is behaviorally implausible for a human |
| H2 | Device farm | An unusually large number of distinct devices behind one IP indicates emulator/device farms |
| H3 | Click farm | An unusually large click volume from one IP indicates scripted repeat participation |
| H4 | Concentration | If H1–H3 hold, anomalous traffic should cluster in specific media and ad types, not spread evenly |
| H5 | Temporal signature | Automated traffic shouldn't follow human activity rhythms (time of day, weekday/weekend) |

### 3. Row-level classification

Priority order matters: `outlier` (any of the 3 signals hits its extreme percentile) is checked
first, then `abusing` (any signal in the risky/warning band), then `non_rewarded` (never
converted), with `normal` as the default. `abusing` must be checked *before* `non_rewarded` —
device/click-count signals are IP-level properties that apply to a click whether or not it
converted, so checking non-reward status first would silently hide anomalous non-converting
clicks inside the "normal, just didn't convert" bucket.

### 4. Media-level risk aggregation

Click-level signals roll up to `critical / risky / warning / normal` tiers per media, using the
proportion of high-confidence and monitoring-band traffic. **Media risk thresholds are heuristic**
and use raw ratios with no sample-size adjustment — a media with 4 clicks and 40% extreme-ratio
currently gets the same label as one with 1M clicks and 35% (documented as a known limitation,
not silently ignored).

### 5. Historical (14-month) validation, with an `unscored` category

August risk labels are checked against 14 months of prior behavior for the same media. Media the
August click-level detector never evaluated are kept as `unscored` — not silently folded into
`normal`, which would understate how much of the historical data was actually checked.

### 6. Acute vs. chronic classification

Not every high-risk media is the same kind of problem. Splitting by average share and
recent-vs-historical ratio across all 19 risky/critical media reveals three distinct patterns:
one **acute infiltration** (sudden, large jump — needs an incident response), two **chronic
high-share** media (structurally large the entire period — needs a contract/rate conversation,
not a block), and sixteen **chronic low-share** media (routine monitoring). The single
highest-exposure media turns out to be chronic, not the acute one — conflating the two would have
produced the wrong operational recommendation.

### 7. Threshold sensitivity

Every threshold in the main pipeline (CTIT factor, extreme percentile) is a choice, not a law of
nature. Three coherent scenarios (not a full grid search) test whether the *ranking* of
highest-exposure media survives a 3x range of strictness:

| Scenario | CTIT factor | Extreme percentile |
|---|---:|---:|
| Conservative | ×0.05 | 99.95% |
| Base | ×0.10 | 99.90% |
| Aggressive | ×0.15 | 99.50% |

### 8. Out-of-time alert backtest — the look-ahead-bias fix

Every threshold above is computed from the same 31-day window that contains the anomaly it's
used to flag — legitimate for retrospective discovery, not for claiming an operational detector.
This section is the exception: a 30-day rolling median/MAD baseline built **only from prior days**
(`.shift(1)` before `.rolling()`, so a day's own value never leaks into its own baseline) is used
to ask a sharper question than "was this detectable" — *when* would it have been detectable, using
only information available at the time.

---

## Results (last verified run)

| Metric | Value |
|---|---:|
| Clicks analyzed | 16,831,054 |
| High-confidence anomaly | 6,387,902 (38.0%) |
| Monitoring-band | 7,782,515 (46.2%) |
| Normal / non-converted low-risk | 811,565 + 1,849,072 (15.8%) |
| Raw platform CVR → remaining low-risk traffic CVR | 8.74% → 30.50% |
| High-confidence financial exposure | ₩67,083,271 |
| Monitoring-band financial exposure | ₩193,575,035 |
| Media evaluated | 189 (7 critical / 12 risky / 35 warning / 135 normal) |
| Exposure concentration | top 2 media = 75.5% of high-confidence exposure |
| Media typology | 1 acute infiltration, 2 chronic high-share, 16 chronic low-share |
| Media unevaluated by the August detector (`unscored`) | 12.6% of 14-month report rows |

**Threshold sensitivity:** flagged-click share ranges 34.6%–51.6% across scenarios, and the
critical-media count nearly doubles at the loosest setting (7→12) — but the top-4 highest-exposure
media are identical in all three scenarios. The exact volume is threshold-sensitive; the identity
of the highest-risk media is not.

**Out-of-time backtest:** the pre-specified alert rule (click-share z-score > 5 **and**
same-day CVR z-score < −3) never fires, including during the actual event — a genuine null result,
reported as such. The diagnosis: click-share alone crosses the anomaly threshold one full day
*before* the event's visible plateau, but the media's own CVR is historically volatile enough that
even a near-total collapse doesn't register as an extreme z-score. **The AND-combination is what
suppressed the alert, not an absence of signal** — a share-only rule at the same strict threshold
would have caught it with roughly one day of latency.

---

## Repository structure

```
├── ive_korea_fraud_detection_portfolio.ipynb   # Part 1 (individual) + Part 2-3 (team, lighter)
├── requirements.txt
├── data/                                       # not included — see Data section above
└── README.md
```

## Setup

```bash
pip install -r requirements.txt
jupyter notebook ive_korea_fraud_detection_portfolio.ipynb
```

Run top to bottom (Kernel → Restart & Run All). Part 1 alone processes ~16.8M engagement rows and
14 months of hourly report data; 16GB RAM is recommended.

---

## Known limitations

- No independently verified invalid-traffic ground truth — every label here is a statistical
  extreme relative to the platform's own distribution, not a confirmed determination
- Look-ahead bias in the main 31-day pipeline (the out-of-time backtest is the one exception)
- Domain labeling (rule + LLM hybrid) spot-checked on 100 ads (94% agreement), not exhaustively
  audited
- Media-risk rates have no confidence interval / sample-size adjustment
- The out-of-time alert design was validated on one media, not platform-wide

## Future work

Media-risk confidence intervals (Wilson / Beta-Binomial shrinkage), inter-click interval features,
IP-device graph structure (device-farm detection via connected components), signal
ablation/overlap analysis, peer-group (media-size-relative) thresholds, and a properly-regrained
rule-vs-unsupervised comparison (Isolation Forest at the media×IP grain rather than per-click).

## Individual contribution

Data integration, data-quality validation, feature engineering, the fraud detection pipeline,
financial exposure quantification, 14-month historical validation, acute/chronic classification,
threshold sensitivity, and the out-of-time backtest are the author's individual work. Media
operations management and domain ad-type optimization were built by teammates.
