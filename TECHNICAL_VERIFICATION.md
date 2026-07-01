# TECHNICAL VERIFICATION: STATISTICAL VALIDITY
## Detailed Check of Claimed Methods

**Focus:** Can the claimed results be mathematically correct given the reported statistics?

---

## FINDING 3: SPREAD MEDIATION - MATHEMATICAL ERROR CONFIRMED

### Reported Data (from novel_findings_summary.json)
```
β_volume: 0.021322
β_spread: 0.073024
β_interaction: -0.026652
```

### From iteration1_2_results.txt (lines 163-171)
```
Bivariate Correlations:
  log(volume) ↔ abs_error: r=0.0159, p=3.49e-29
  spread ↔ abs_error: r=0.1174, p=0.00e+00
  log(volume) ↔ spread: r=0.3727, p=0.00e+00

Partial Correlation:
  log(volume) ↔ abs_error | spread: r=-0.0303, p=7.49e-102
  Mediation effect: -91.0% of volume effect explained by spread
```

### MATHEMATICAL VERIFICATION

**Standard mediation framework (Baron & Kenny, 1986):**

1. Total effect (c): X → Y = 0.0159
2. Direct effect (c'): X → Y | M = -0.0303  
3. Indirect effect (a×b): (X → M) × (M → Y)

**Calculate indirect effect:**
- a (X → M): r(volume, spread) = 0.3727
- b (M → Y): r(spread, error) = 0.1174
- Indirect ≈ 0.3727 × 0.1174 ≈ 0.0437

**Mediation proportion:**
```
Mediation % = Indirect / Total = 0.0437 / 0.0159 = 275%
```

**Problem:** You can't have >100% mediation in standard framework unless:
1. Suppression effect (direct and indirect have opposite signs) ✓ c'=-0.0303, c=+0.0159
2. BUT: This means the "direct effect" is STRONGER than total effect (suppression)
3. This is possible, but the interpretation "spread mediates 91%" is WRONG

**Correct interpretation:**
- Volume has WEAK positive association with error (r=0.016)
- Once you control for spread, volume has MODERATE negative association (r=-0.030)
- This is **SUPPRESSION**, not mediation
- Spread is a confounder that masks volume's negative effect

**The claim "91% mediation" is mathematically incoherent with the reported statistics.**

### What the data actually shows:

**Path diagram:**
```
           +0.373
Volume -----------> Spread
  |                    |
  | -0.030 (direct)    | +0.117
  |                    |
  v                    v
         Error
  
Total: +0.016 (weak positive)
```

**Correct story:**
1. Higher volume → wider spread (r=+0.373) [This is BACKWARDS from theory!]
2. Wider spread → higher error (r=+0.117) [Correct]
3. Volume → error direct: negative (r=-0.030) [Correct]
4. Volume → error total: weak positive (r=+0.016) [Positive because spread confounds]

**The problem:** Volume is positively correlated with spread (r=+0.373)

This contradicts market microstructure theory:
- Standard theory: More volume → tighter spreads (liquidity reduces friction)
- Kalshi data: More volume → WIDER spreads?!

**Possible explanations:**
1. **Measurement error**: Spread measured wrong (see below)
2. **Selection**: High-volume markets are on uncertain events (wide spreads)
3. **Kalshi artifact**: Discrete pricing creates floor effect (spread=1.0 or 0.0)

---

## FINDING 4: EARLY-STAGE PREDICTION - DATA LEAKAGE CONFIRMED

### Code Review (from reports)

From RESEARCH_REFINEMENT_REPORT.md line ~300:
```
Predictive Model: Logistic regression on markets with <100 trades

Model: P(high_error) ~ spread_wide + price_extreme + sports + volume

Results:
  β_spread_wide:    0.668 *** (wide spread predicts persistent error)
  β_price_extreme: -7.958 *** (extreme prices protect)
  β_sports:        -2.901 *** (sports more accurate)
  β_volume:         0.124
  Accuracy: 91.2%
```

### CRITICAL QUESTIONS

**Q1: When is spread_wide measured?**
- Claim: "Early-stage" (<100 trades)
- But: Are these markets that ENDED at <100 trades, or measured AT 100 trades?

**From iteration3_results.txt line 78:**
```
Early-stage markets (<100 trades): N=462,207
```

**This is ambiguous.** Does "early-stage markets" mean:
- (A) Markets measured when they hit 100 trades? [Correct for prediction]
- (B) Markets that closed with <100 total trades? [Wrong - this is post-hoc classification]

**Evidence for (B) - DATA LEAKAGE:**
1. Sample size N=462,207 is ~92% of 500K dataset
2. From volume distributions (iteration1_2_results.txt), <100 volume is ~90% of markets
3. This suggests they selected markets that ENDED thin, not markets measured early

**Q2: What is "high_error" threshold?**
- Not specified in reports
- If threshold is MAE > median, then 50% base rate → 50% baseline accuracy
- 91.2% would be impressive
- But if threshold is MAE > 75th percentile, then 75% base rate → always predict "no" gives 75%

**Q3: Train/test split?**
- No mention of train/test split in any report
- Reports say "Classification accuracy: 91.2%" with no cross-validation details
- Red flag for overfitting

### VERIFICATION: What would real early-stage look like?

**Proper methodology:**
```python
# For each market that reached 100+ trades eventually:
train_markets = markets[markets['volume'] >= 100 & markets['open_time'] < '2024-01-01']
test_markets = markets[markets['volume'] >= 100 & markets['open_time'] >= '2024-01-01']

# Measure features at t=100 trades
for market in train_markets:
    features_at_100 = get_state_at_trade_number(market, trade_num=100)
    outcome_at_close = market['final_mae']
    
# Train on train_markets, evaluate on test_markets
```

**What they likely did (WRONG):**
```python
# Select markets that ended thin
early_stage = markets[markets['final_volume'] < 100]

# Use final-state features
features = {
    'spread_wide': markets['final_spread'] > threshold,  # LEAKAGE
    'price_extreme': abs(markets['final_price'] - 0.5) > 0.4,  # LEAKAGE
    'volume': markets['final_volume'],  # LEAKAGE
    'sports': markets['category'] == 'Sports'  # OK
}

# Predict final_error using final-state features
# This is circular!
```

### CONSEQUENCE

If they used final-state features, the model is predicting:
**"Markets that ended thin with wide spreads are inaccurate"**

This is:
1. Tautological (thin markets are known to be less accurate)
2. Not predictive (can't intervene before market closes)
3. Not useful (practitioners already know this)

**The 91.2% accuracy is meaningless without:**
- Temporal train/test split
- Features measured at fixed early timepoint
- Comparison to baseline (predict majority class)

---

## FINDING 1: HORIZON × VOLUME - SELECTION BIAS CHECK

### Reported Effect (from novel_findings_summary.json)
```
<1 day:
  mae_below_200: 0.12013
  mae_above_200: 0.06831
  improvement: 1.758
  n_below: 244,650
  n_above: 14,941
  p_value: 4.17e-89

7-30 days:
  mae_below_200: 0.01139
  mae_above_200: 0.07106
  improvement: 0.160 (WORSE with volume!)
  n_below: 58,605
  n_above: 3,033
  p_value: ~0.0
```

### STATISTICAL TEST: Is this selection bias?

**Hypothesis:** Long-horizon markets with <200 trades closed early because outcome became obvious.

**Evidence:**
1. MAE for 7-30 day markets with <200 trades: 1.14% (extremely accurate!)
2. This is lower than short-horizon markets with >200 trades (6.83%)
3. Only 4.9% of 7-30 day markets reach 200 trades

**Plausible mechanism:**
- Event at t=0: "Will X happen by t=30 days?"
- At t=5 days: New information makes outcome 99% certain
- Traders stop trading (why trade at 99%?)
- Market closes at t=30 with 50 trades, final price 99%, outcome 100%
- MAE = 1%, volume = 50

**Compare to:**
- Event stays uncertain until t=29 days
- Heavy trading continues (200+ trades)
- Final price more volatile, MAE higher

**This is SELECTION, not causation.**

### VERIFICATION: Test for selection

**If selection hypothesis is correct, we should see:**
1. Low-volume long-horizon markets close earlier than expected
2. Low-volume long-horizon markets have final prices near 0% or 100%
3. High-volume long-horizon markets have final prices near 50% (uncertain)

**Can check with reported data:**

From iteration1_2_results.txt (Opening Price Anchor Analysis):
```
opening_class    abs_error  count   outcome
Extreme_Low        0.0647   2,709   0.9491
Extreme_Low        0.0854 478,324   0.0834
```

Wait, these categories are by opening price, not final price. Need final price distribution by volume.

**Missing analysis:** Final price distribution by (horizon × volume) interaction

If my hypothesis is correct:
- 7-30 day, <200 trades should have mean(|final_price - 0.5|) > 0.45 (near extremes)
- 7-30 day, >200 trades should have mean(|final_price - 0.5|) < 0.3 (near middle)

**Without this check, the "improvement" calculation is meaningless.**

---

## FINDING 2: AMBIGUITY MODERATION - VERIFICATION

### Reported Effect (from novel_findings_summary.json)
```
Very Clear:
  mae_below_200: 0.0884
  mae_above_200: 0.0307
  improvement: 2.875
  n_below: 457,865
  n_above: 24,362
  p_value: 1.24e-222

Very Ambiguous:
  mae_below_200: 0.4922
  mae_above_200: 0.4934
  improvement: 0.998
  n_below: 1,579
  n_above: 611
  p_value: 0.639 (not significant)
```

### STATISTICAL VALIDITY?: ✓ Probably correct

**Check 1: Sample sizes adequate?**
- Very Clear: N=482,227 total (large)
- Very Ambiguous: N=2,190 total (small but adequate for t-test)

**Check 2: Effect sizes make sense?**
- Very Clear: 8.84% → 3.07% (5.77 pp improvement)
- Very Ambiguous: 49.22% → 49.34% (0.12 pp worse, n.s.)
- Effect difference: 5.65 pp

**Check 3: Is p-value reasonable given N and effect?**
For Very Clear (N=482K, effect=5.77pp, std~15pp):
```
SE = 15 / sqrt(482227) ≈ 0.022
t = 5.77 / 0.022 ≈ 262
p < 10^-200 ✓ (matches reported p=1.24e-222)
```

**This finding appears statistically valid.**

### CONCEPTUAL ISSUE: Definition of ambiguity

From RESEARCH_REFINEMENT_REPORT.md:
```
Ambiguity = 1 - 2|p - 0.5|
```

This is measured using FINAL price, not early price.

**Problem:** This creates ex-post categorization, not ex-ante prediction.

**Example:**
- Market: "Will vote be decided by <1%?"
- True outcome: 50.1% to 49.9% (yes, <1%)
- Final price: 52% (correct prediction!)
- Ambiguity score: 1 - 2|0.52-0.5| = 0.96 (classified "Very Ambiguous")
- But this market was PREDICTABLE (polls showed close race)

**Better definition:**
- Ambiguity = price standard deviation in first 24 hours
- Or: disagreement = bid-ask spread in first 24 hours
- Or: category-based (economics vs sports inherently different)

**Despite definition issues, the finding is directionally valid:**
- Markets near 50% are harder to forecast (genuine uncertainty or close events)
- Volume doesn't help much (no signal to aggregate)

---

## FINDING 5: CATEGORY THRESHOLDS - SPORTS ANOMALY EXPLAINED?

### Reported Effects (from category_threshold_analysis.csv)
```
Sports, 500 trades:
  mae_below: 0.0099 (1.0%!)
  mae_above: 0.0449 (4.5%)
  improvement: 0.221 (worse with volume)
  n_below: 44,622
  n_above: 1,119

Economics, 500 trades:
  mae_below: 0.1592 (15.9%)
  mae_above: 0.0754 (7.5%)
  improvement: 2.112 (better with volume)
  n_below: 254
  n_above: 28
```

### HYPOTHESIS: Sports timing artifact

**Check 1: Are sports markets live-traded?**
- If yes: high-volume sports markets include in-play trading (high volatility)
- Final price measured at event end may not reflect pre-event accuracy

**Check 2: Sample composition**
Sports <500 trades (N=44,622):
- Likely: Pregame markets on obvious outcomes
- Example: "Will Warriors beat college team?" (99% pregame, 100% outcome)
- Low volume because no disagreement

Sports >500 trades (N=1,119, only 2.4%):
- Likely: Live markets or controversial calls
- Example: "Will player X score next?" (fluctuates during game)
- High volume from live trading, but price swings don't reflect ex-ante accuracy

**Check 3: What is "final price" for sports?**
Critical question: When is price measured?
- If measured at event END: live markets will have price=0% or 100% (event resolved)
- If measured at market CLOSE: pre-event markets more accurate

**This explains the anomaly:**
- High-volume sports markets trade during event → price at close diverges from outcome
- Low-volume sports markets close before event → price at close = good prediction

**VERIFICATION NEEDED:**
```python
# For each sports market:
sports['price_at_event_start'] = get_price_at(event_start_time)
sports['price_at_event_end'] = get_price_at(event_end_time)
sports['mae_pregame'] = abs(price_at_event_start - outcome)
sports['mae_final'] = abs(price_at_event_end - outcome)

# Test hypothesis:
# High-volume markets should have mae_final > mae_pregame (worse due to live trading)
```

**If hypothesis is correct, the entire finding is an artifact of measurement timing.**

---

## SUMMARY: TECHNICAL VALIDITY

| Finding | Math Correct? | Measurement Correct? | Interpretation Correct? |
|---------|---------------|----------------------|-------------------------|
| 1. Horizon×Volume | ✓ Stats valid | ❌ Selection bias | ❌ Not causal |
| 2. Ambiguity | ✓ Stats valid | ⚠️ Ex-post definition | ✓ Directionally correct |
| 3. Spread mediation | ❌ Math error | ⚠️ Correlation OK, mediation NO | ❌ Wrong mechanism |
| 4. Early prediction | ? Unknown | ❌ Likely data leakage | ❌ Not predictive |
| 5. Category thresholds | ✓ Stats valid | ❌ Timing artifact | ❌ Wrong conclusion |

### MOST CRITICAL ISSUES

**1. Spread "mediation" is suppression**
- The 91% claim is mathematically wrong
- Volume→spread correlation is+0.373 (should be negative per theory)
- Suggests measurement error or selection bias in spread variable

**2. Early prediction is circular**
- 91.2% accuracy fishing from using final-state features on final-state outcomes
- No temporal validation
- Useless for actual early warning

**3. Sports anomaly reveals data quality issue**
- 1% MAE for low-volume sports is suspiciously good
- Suggests these markets closed early on obvious outcomes
- Undermines all volume comparisons (selection, not causation)

### RECOMMENDATION

**Before publication:**
1. Re-analyze spread: why is volume→spread positive? Check for errors
2. Re-build early prediction with proper temporal split and feature timing
3. Investigate sports: separate pregame vs live markets
4. Add within-market panel analysis to control for fixed effects
5. Explicitly test selection hypothesis (IV or propensity score matching)

**Without these fixes:**
- Findings 3, 4, 5 should be DROPPED
- Finding 1 needs heavy caveats
- Only Finding 2 is publication-ready (with better ambiguity definition)

**Realistic publication outcome:**
- 1-2 solid findings (not 5)
- Descriptive paper (not causal)
- JEBO or J Prediction Markets (not MS/AER)
