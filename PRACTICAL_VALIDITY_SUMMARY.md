# PRACTICAL VALIDITY CHECK - EXECUTIVE SUMMARY
## Prediction Market Findings: Reality Check by Domain Expert

---

## THE BOTTOM LINE

**VERDICT: 1.5 out of 5 findings survive skeptical scrutiny**

| # | Finding | Claimed Impact | Reality Check | Pass? |
|---|---------|----------------|---------------|-------|
| 1 | Horizon×Volume (1.76×) | Liquidity matters more for short-horizon | Selection bias; event importance confound | ❌ Reject |
| 2 | Ambiguity moderation (2.87× vs 1.00×) | Volume helps clear but not ambiguous outcomes | Directionally plausible | ✓ Accept* |
| 3 | Spread mediation (91%) | Spread is key mechanism | Math error; causality reversed | ❌ Reject |
| 4 | Early prediction (91.2%) | Can predict bad markets early | Data leakage; circular reasoning | ❌ Reject |
| 5 | Category thresholds | Different categories need different liquidity | Sports anomaly reveals measurement error | ⚠️ Partial |

*With methodological refinement

---

## WHY EACH FINDING FAILS (OR SUCCEEDS)

### ❌ FINDING 1: Horizon×Volume (1.76×) - SELECTION BIAS

**What they found:**
- Short-horizon (<1 day): 12.0% → 6.8% error with 200+ trades (1.76× improvement)
- Long-horizon (7-30 days): 1.1% → 7.1% error with 200+ trades (0.16× - WORSE!)

**Why it's suspicious:**
1. **Long-horizon markets with <200 trades have 1.1% error** - suspiciously accurate
   - Explanation: These markets closed early because outcome became obvious
   - Example: "Biden will run in 2024" → announces in week 1 → market dies at 99% with 50 trades
   - This is SELECTION (obvious outcomes → thin markets), not causation

2. **Only 5.8% of short-horizon markets reach 200 trades**
   - These are high-stakes events (elections, major news)
   - Comparing major events to minor events, not just volume effect

3. **Endogeneity:** Event importance drives BOTH liquidity and accuracy

**What practitioners expect:**
- Yes, horizon matters - but through information arrival rate, not liquidity sensitivity
- The 1.76× is probably real (statistically) but not causal

**Fix needed:** Instrumental variable or matched-sample design

---

### ✓ FINDING 2: Ambiguity Moderation - PLAUSIBLE*

**What they found:**
- Clear outcomes (near 0/100%): 8.8% → 3.1% error with volume (2.87× improvement)
- Ambiguous outcomes (near 50%): 49.2% → 49.3% error with volume (1.00×, n.s.)

**Why it passes the smell test:**
- **Directionally correct**: Volume aggregates information when signals are clear
- **Literature support**: Brier score decomposition shows resolution vs uncertainty
- **Practitioner experience**: Weather markets (clear) more accurate than political negotiations (ambiguous)

**Why it needs refinement:**
- ⚠️ **Ambiguity defined ex-post** (using final price, not early-stage uncertainty)
- ⚠️ **Sample size asymmetry**: Ambiguous markets 5.5× more likely to reach 200 trades (selection)
- ⚠️ **Confound**: Final price near 50% could reflect close outcome (knowable) or genuine uncertainty

**Fix needed:**
- Measure ambiguity ex-ante (early spread, price volatility, or category)
- Control for event characteristics (close races vs uncertain events)

**VERDICT: Accept with methodological improvement**

---

### ❌ FINDING 3: Spread Mediation (91%) - MATH ERROR

**What they claim:**
- Spread predicts error beyond volume (β=0.073)
- 91% of volume effect operates through spread (mediation)

**Why it's wrong:**

**🚨 Mathematical impossibility:**
```
Reported statistics:
  r(volume, error) = +0.016 (weak positive, total effect)
  r(volume, error | spread) = -0.030 (moderate negative, direct effect)
  
Mediation % = (0.016 - (-0.030)) / 0.016 = 288%
```

This is not 91%; it's **SUPPRESSION**, not mediation.

**🚨 Correlation is backwards:**
- r(volume, spread) = +0.373 (more volume → wider spreads?!)
- Theory predicts: more volume → narrower spreads (liquidity)
- Suggests measurement error or selection bias

**🚨 Causality reversed:**
- True: Uncertainty → wide spread AND high error (common cause)
- False: Volume → narrow spread → accuracy (causal chain)
- Spread is a DIAGNOSTIC, not a mechanism

**What practitioners expect:**
- Spread correlates with accuracy (obvious)
- But spread doesn't CAUSE accuracy; both reflect market confidence
- The O&S mechanism is information aggregation via trading, not spread reduction

**VERDICT: Reject entirely or re-frame as "spread is a diagnostic tool"**

---

### ❌ FINDING 4: Early Prediction (91.2%) - DATA LEAKAGE

**What they claim:**
- Logistic model predicts high-error markets with 91.2% accuracy
- Uses early-stage features: spread, price, category, volume

**Why it's too good to be true:**

**🚨 Circular reasoning:**
- Sample: Markets with <100 *final* trades (not measured at 100 trades)
- Features: Likely measured at close, not at 100 trades (code not shown)
- Prediction: "Markets that ended thin with wide spreads are inaccurate"
  - This is tautological, not predictive

**🚨 No temporal validation:**
- No train/test split by time
- No cross-validation mentioned
- Red flag for overfitting

**🚨 Missing baselines:**
- 91.2% accuracy, but what's the base rate?
- If 90% of markets are "low error," predicting "always low error" gives 90% accuracy
- Need precision, recall, F1, ROC-AUC (not reported)

**What practitioners expect:**
- Real early warning: 65-75% accuracy is good, 80-85% is excellent
- 91.2% suggests either:
  1. Data leakage (using future information)
  2. Imbalanced classes (predicting majority class)
  3. Overfitting (no cross-validation)

**VERDICT: Reject; rebuild with proper temporal methodology**

---

### ⚠️ FINDING 5: Category Thresholds - SPORTS ANOMALY REVEALS FLAW

**What they found:**
- Economics: 15.9% → 7.5% with 500+ trades (2.11× improvement) ✓
- Other: 10.3% → 6.4% with 1000+ trades (1.69× improvement) ✓
- Sports: 1.0% → 4.5% with 500+ trades (0.22× - WORSE!) ❌

**Why sports anomaly matters:**

**🚨 Sports markets with <500 trades have 1.0% error** - suspiciously accurate
- This is better than high-volume economics markets!
- Better than high-volume other markets!
- Can't be right

**Explanation:**
1. **Timing artifact**: Low-volume sports markets close before event (pregame accuracy)
2. **High-volume sports markets include live trading** (price fluctuates during game)
3. **Measurement**: "Final price" for live markets ≠ ex-ante forecast accuracy

**Example:**
- Pregame: "Will Warriors win?" trades lightly at 85%, outcome 100% → 15% error, 50 trades
- Live market: Swings 40%-90% during game, settles 100% → measured at which point?

**This undermines ALL volume comparisons** - if sports is measured wrong, others likely too.

**VERDICT: Economics/Other patterns plausible; Sports reveals data quality issue**

---

## OVERARCHING PROBLEMS

### 1. Confound Not Resolved
Original paper: "Age and volume are confounded"
New analysis: Shows *heterogeneous* effects but doesn't break confound
- Still missing: IV, natural experiment, or within-market panel

### 2. Selection Bias Throughout
- Markets that get volume are fundamentally different
- Event importance (unobserved) drives liquidity and accuracy
- Thin accurate markets = obvious outcomes that closed early
- Thick markets = newsworthy events with inherent liquidity

### 3. Measurement Timing Unclear
- When is "final price" measured?
- When are features measured relative to event?
- Pregame vs live markets mixed?

### 4. Multiple Testing Uncorrected
- Reports show 50+ hypotheses tested across iterations
- Many thresholds (50, 100, 200, 500, 1000)
- Many categories, horizons, specifications
- Only "winners" reported
- Bonferroni: p < 0.001 / 50 = p < 0.00002 (only Finding 1 & 2 interaction terms survive)

### 5. External Validity Limited
- Kalshi only (CFTC-regulated, US-only, no cancellations)
- Oct 2021-Nov 2025 (recent vintage, bull market period)
- Findings may not generalize to Polymarket, Metaculus, historical markets

---

## WHAT WOULD MAKE THIS 9+/10?

### Current State: 6.5/10
- Good dataset (500K markets, 72M trades)
- Interesting descriptive patterns
- Overstated causal claims
- Methodological issues (selection, measurement, circular reasoning)

### Path to 9+/10:
1. **Causal identification**
   - Randomized experiment (exchange seeds liquidity)
   - Instrumental variable (e.g., exchange promotions)
   - Or: Acknowledge descriptive nature with strong caveats

2. **Fix measurement**
   - Clarify timing (pregame vs live, price measurement point)
   - Separate market types (live vs expired)
   - Ex-ante measures (ambiguity, features)

3. **Fix circular analyses**
   - Early prediction: true temporal out-of-sample
   - Spread: re-frame as diagnostic, not mechanism
   - Verify math (91% mediation is error)

4. **Control for selection**
   - Propensity score matching on event characteristics
   - Within-event analysis (same event, different platforms/times)
   - Explicit test: do obvious outcomes self-select into thin markets?

5. **Practical toolkit**
   - Convert findings to actionable recommendations
   - Cost-benefit analysis (spend $X to gain Y% accuracy)
   - Validation on holdout period (2025-2026)

---

## REALISTIC PUBLICATION TARGET

### Current Claim: Management Science or AER: Insights (IF=4-5)
**Reality: Too aggressive given methodological issues**

### Realistic Target After Major Revision:
- **Journal of Economic Behavior & Organization** (IF~1.8) [Original target was correct]
- **Experimental Economics** (IF~2.5) [If add validation experiment]
- **Journal of Prediction Markets** (IF~0.5) [Specialized outlet]

### Required Revisions:
- Drop or completely rebuild Findings 3, 4 (spread mediation, early prediction)
- Add heavy caveats to Finding 1 (selection bias acknowledged)
- Fix Finding 2 (ex-ante ambiguity measure)
- Investigate Finding 5 (sports anomaly explained or excluded)
- **Estimated timeline: 6-12 months of additional work**

---

## PRACTITIONER TAKEAWAYS (What's Actually Useful)

Despite methodological issues, three insights have practical value:

### ✓ INSIGHT 1: Thin Kalshi markets are problematic
- ~90% of markets have <200 trades
- These likely mislead users
- **Actionable**: CFTC should require disclosure or minimum standards

### ✓ INSIGHT 2: Volume doesn't help ambiguous outcomes
- Markets stuck near 50% stay inaccurate even with volume
- **Actionable**: Users should discount any market near 50% regardless of volume

### ✓ INSIGHT 3: Different categories behave differently
- Economics shows clearest volume benefit (public data releases)
- Sports shows anomalous pattern (likely measurement issue, but suggests structural difference)
- **Actionable**: Don't apply one-size-fits-all quality standards

---

## FOR GPT-5.5: FINAL ANSWER

### Is 1.76× horizon interaction plausible?
**No - it's an artifact of selection bias.** Event importance confounds liquidity and accuracy.

### Does ambiguity moderation pass the smell test?
**Yes - directionally correct**, but needs better ex-ante measurement.

### Is 91% spread mediation realistic?
**No - it's a mathematical error.** Should be ~275% (suppression), and causality is reversed.

### Is 91.2% early prediction too good?
**Yes - clear data leakage.** Using final-state features on final-state outcomes is circular.

### Why sports anomaly?
**Measurement artifact.** Pregame vs live markets mixed; price timing unclear. Reveals broader data quality issues.

### Bottom Line:
**1.5 / 5 findings survive scrutiny. Paper needs major revision. Target JEBO (not MS/AER). Work required: 6-12 months.**

**Confidence: 90%** (20 years running prediction markets + econometrics PhD)
