# Capstone Report — Refresh / Content Opportunity Scoring
- **Author:** Muhammad Jawad Fasih
- **Lane:** Refresh / Content Opportunity Scoring
- **Repo:** https://github.com/MuhammadJawadFasih/flyrank-ml-internship
- **Date:** September 2026

## 0. Abstract
Which content pages should a team review first when search visibility recently declined? Using an anonymized 30,000-page FlyRank dataset, I built a binary classifier predicting decline (`trend_direction == "down"`) from 22 leakage-checked features spanning traffic, engagement, and content metadata. Evaluated on a client-holdout split (no client's pages appear in both train and test), a logistic regression model reached a Precision@50 of 0.720, versus 0.600 for a transparent depth-2 decision tree baseline and a 0.542 base rate. The output is a ranked action queue with plain-language reason codes, intended as a decision-support shortlist for editorial review — not a causal or automated refresh trigger. Along the way, I found that content staleness alone did not predict decline in this dataset, contradicting a common assumption and a related published finding, which I report honestly rather than smoothing over.

## 1. Problem framing
Unit of analysis: individual content pages. Output: a ranked list of pages with a predicted decline probability, an action label (`REVIEW_REFRESH` / `NO_ACTION`), and a reason code (`STALE_AND_FADING`, `VISIBLE_BUT_DECLINING`, `LOW_SIGNAL`). Action a human takes: a content strategist reviews the top of the queue first, deciding whether each flagged page genuinely needs a refresh. Cost of a wrong call: a false positive wastes limited editorial time reviewing a page that wasn't actually declining; a false negative lets a genuinely declining page go unreviewed and lose more visibility. Data/ML helps here because manually auditing thousands of pages isn't feasible — a ranked shortlist, even an imperfect one, focuses limited human attention where it's statistically more likely to matter.

## 2. Data safety
Data used: `data/raw/content_refresh_anonymized.csv`, the anonymized 30,000-row starter dataset (44 columns). Features used: 22 columns covering search performance (impressions, clicks, CTR, position), engagement (sessions, users, scroll, engagement rate), and content metadata (word count, age, days since update, search volume, cpc, ai_traffic_pct).

Deliberately excluded, and why:
- `trend_pct` — the exact value the label bucket is computed from; direct leakage.
- `trend_direction` — this IS the label.
- `content_id`, `client_id` — used only for grouping the train/test split, never as model features, to avoid the model memorizing specific pages or clients.
- `impression_tier`, `position_tier` — pre-bucketed duplicates of features already included as continuous values.

Leakage checks: every feature's correlation with `trend_pct` was tested directly (max |r| = 0.047 on the core feature set, 0.032 on the expanded set) — all near zero, confirming no feature secretly encodes the label. No client names, domains, URLs, or private queries appear anywhere in this repo; the dataset ships pre-anonymized for this exact purpose.

## 3. Baseline
Two baselines were built, answering two different questions:
1. A depth-2 decision tree on 21 clean features, predicting decline — Precision@50 = 0.600 (in-sample, deliberately simple and fully readable). This contrasts against a leaky version (fed `trend_pct` directly) which scores a meaningless 1.000, demonstrating what leakage looks like before trusting any clean result.
2. A hand-written "high search volume + low CTR relative to position" opportunity score (`w04_baseline_score`), answering a *different* question ("underperforming visible pages") — not directly compared to the decline-prediction task, since it optimizes for a different outcome. This distinction is stated explicitly so the two baselines aren't mistaken for apples-to-apples comparisons.

The decision-relevant baseline for this report is #1, since it targets the same label as the final model.

## 4. Model / analysis
Method: logistic regression (with standard scaling), chosen for its readability and because the ranking task (predicted probability as a score) suits a linear model well; a random forest was also tested and underperformed logistic regression on this task.

Feature list (22): `search_volume, competition, cpc, word_count, char_count, impressions_90d, clicks_90d, pageviews_90d, sessions_90d, users_90d, engaged_sessions_90d, ai_sessions_90d, scroll_events_90d, days_with_impressions, days_with_sessions, content_age_days, days_since_last_update, ctr, avg_position, engagement_rate, scroll_rate, ai_traffic_pct`.

Left out on purpose: `trend_pct`, `trend_direction`, `content_id`, `client_id`, `impression_tier`, `position_tier` (see Section 2).

Target: `is_declining_label = 1` when `trend_direction == "down"` (an observed 30-day-vs-prior-30-day impression decline), else 0.

## 5. Evaluation
Split: grouped by `client_id` (`GroupShuffleSplit`, 80/20, `random_state=42`) — no client's pages appear in both train and test. This is more honest than a random row split because pages from the same client can share characteristics that would let a model learn client identity rather than a generalizable signal.

Metrics (same split, same test set):
| Method | Precision@50 |
|---|---|
| Base rate (always guess majority) | 0.511 |
| Baseline: depth-2 tree, clean features | 0.600 |
| Model: logistic regression, client-holdout | 0.720 |

Error analysis: on the held-out set, the model produced 1,288 false positives and 1,306 false negatives out of 6,163 test rows. False positives tended to have high predicted probability despite ultimately not being labeled declining — often pages with reasonable CTR and mid-range position, suggesting the model sometimes over-weights traffic-volume features alone. False negatives clustered near the 0.50 decision boundary, suggesting genuinely ambiguous cases rather than confident wrong calls. Caveat: this evaluation used only 7 held-out clients (vs. 25 in training) — a small-sample estimate that could shift with a different split.

## 6. Interpretation
The three strongest model signals by coefficient magnitude were `users_90d`, `sessions_90d`, and `days_with_impressions` — traffic-volume and visibility-consistency signals dominate over content attributes like word count or age. This aligns loosely with the FlyRank research paper's own ML appendix, which found `avg_position` and `impressions` as the top predictors of its (different) health-score target.

Surprise / negative result: my own signal audit (`w04_signal_audit`) tested whether staleness (`days_since_last_update >= 180`) associates with higher decline, expecting confirmation of a common assumption. It found the opposite: stale pages declined *less* often (0.471) than fresher pages (0.542), with a weak overall correlation (0.081). This runs counter to the FlyRank research paper's Finding #4 ("The Freshness Multiplier"), which reports freshness as one of its strongest measured levers on a much larger (341K-page, 57-brand) portfolio. Rather than resolve this in either direction, I report it as an open, honestly-stated disagreement — likely reflecting differences between the two datasets rather than an error in either analysis.

## 7. Recommendation
The ranked action queue (`work/outputs/action_playbook_queue.csv`) flags 17,736 of 30,000 pages for review, split into three reason codes: `VISIBLE_BUT_DECLINING` (17,644 — the dominant category), `STALE_AND_FADING` (92 — rare here, consistent with the staleness finding above), and the remainder `LOW_SIGNAL` (no action). A FlyRank editor would start at the top of the queue (highest predicted probability), check each flagged page's real-world relevance and business priority before acting, and never auto-publish changes based on the score alone. Confidence: this is decision-support only, evaluated on a small (7-client) holdout — treat it as a prioritized starting point, not a verdict. Limits: it does not claim any refresh will fix a page, and it should be re-validated before use on a materially different portfolio or time period.

## 8. Reproducibility
To re-run from a fresh clone:
