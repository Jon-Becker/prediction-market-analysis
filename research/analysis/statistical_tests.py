#!/usr/bin/env python3
"""Comprehensive statistical tests for paper claims.

Tests the following claims:
1. Trade size: makers > takers (consistency across price levels)
2. YES/NO asymmetry: NO outperforms YES (significance at each price)
3. Category variation: gaps are statistically different
4. Trade size → performance: regression with proper SEs
5. Maker direction: NO > YES (significance)
"""

import json
from pathlib import Path

import duckdb
import numpy as np
import pandas as pd
from scipy import stats
from scipy.stats import mannwhitneyu, ttest_ind, pearsonr, spearmanr


def bootstrap_ci(data, statistic=np.mean, n_bootstrap=10000, ci=0.95):
    """Compute bootstrap confidence interval for a statistic."""
    boot_stats = []
    n = len(data)
    for _ in range(n_bootstrap):
        sample = np.random.choice(data, size=n, replace=True)
        boot_stats.append(statistic(sample))
    lower = np.percentile(boot_stats, (1 - ci) / 2 * 100)
    upper = np.percentile(boot_stats, (1 + ci) / 2 * 100)
    return lower, upper


def cohens_d(group1, group2):
    """Compute Cohen's d effect size."""
    n1, n2 = len(group1), len(group2)
    var1, var2 = np.var(group1, ddof=1), np.var(group2, ddof=1)
    pooled_std = np.sqrt(((n1 - 1) * var1 + (n2 - 1) * var2) / (n1 + n2 - 2))
    return (np.mean(group1) - np.mean(group2)) / pooled_std if pooled_std > 0 else 0


def main():
    base_dir = Path(__file__).parent.parent.parent
    trades_dir = base_dir / "data" / "trades"
    markets_dir = base_dir / "data" / "markets"
    fig_dir = base_dir / "research" / "fig"
    fig_dir.mkdir(parents=True, exist_ok=True)

    con = duckdb.connect()

    results = {}

    # =========================================================================
    # TEST 1: Trade Size by Role - Consistency Across Price Levels
    # =========================================================================
    print("=" * 70)
    print("TEST 1: Trade Size by Role Across Price Levels")
    print("=" * 70)

    trade_size_by_price = con.execute(
        f"""
        WITH resolved_markets AS (
            SELECT ticker FROM '{markets_dir}/*.parquet'
            WHERE status = 'finalized' AND result IN ('yes', 'no')
        )
        SELECT
            CASE WHEN t.taker_side = 'yes' THEN t.yes_price ELSE t.no_price END AS price,
            t.count * (CASE WHEN t.taker_side = 'yes' THEN t.yes_price ELSE t.no_price END) / 100.0 AS taker_size,
            t.count * (CASE WHEN t.taker_side = 'yes' THEN t.no_price ELSE t.yes_price END) / 100.0 AS maker_size
        FROM '{trades_dir}/*.parquet' t
        INNER JOIN resolved_markets m ON t.ticker = m.ticker
        """
    ).df()

    # Group by price deciles
    trade_size_by_price['price_bin'] = pd.cut(trade_size_by_price['price'], bins=10, labels=False) + 1

    price_bin_results = []
    for price_bin in range(1, 11):
        subset = trade_size_by_price[trade_size_by_price['price_bin'] == price_bin]
        if len(subset) < 100:
            continue

        taker_sizes = subset['taker_size'].values
        maker_sizes = subset['maker_size'].values

        # Mann-Whitney U test (non-parametric, better for skewed distributions)
        u_stat, p_value = mannwhitneyu(maker_sizes, taker_sizes, alternative='greater')

        # Effect size (rank-biserial correlation)
        n1, n2 = len(maker_sizes), len(taker_sizes)
        r_effect = 1 - (2 * u_stat) / (n1 * n2)

        price_bin_results.append({
            'price_bin': f"{(price_bin-1)*10+1}-{price_bin*10}¢",
            'taker_mean': np.mean(taker_sizes),
            'maker_mean': np.mean(maker_sizes),
            'taker_median': np.median(taker_sizes),
            'maker_median': np.median(maker_sizes),
            'ratio': np.mean(maker_sizes) / np.mean(taker_sizes),
            'u_statistic': u_stat,
            'p_value': p_value,
            'effect_size_r': r_effect,
            'n_trades': len(subset),
            'maker_larger': np.mean(maker_sizes) > np.mean(taker_sizes)
        })

    price_bin_df = pd.DataFrame(price_bin_results)
    price_bin_df.to_csv(fig_dir / "trade_size_by_role_by_price.csv", index=False)

    print("\nTrade Size by Price Decile:")
    print("-" * 100)
    print(f"{'Price Bin':<12} {'Taker Mean':>12} {'Maker Mean':>12} {'Ratio':>8} {'p-value':>12} {'Effect r':>10} {'Maker>Taker':>12}")
    print("-" * 100)
    for _, row in price_bin_df.iterrows():
        sig = "***" if row['p_value'] < 0.001 else "**" if row['p_value'] < 0.01 else "*" if row['p_value'] < 0.05 else ""
        print(f"{row['price_bin']:<12} ${row['taker_mean']:>10.2f} ${row['maker_mean']:>10.2f} {row['ratio']:>7.2f}x {row['p_value']:>11.2e}{sig} {row['effect_size_r']:>9.3f} {str(row['maker_larger']):>12}")

    # Summary
    n_maker_larger = price_bin_df['maker_larger'].sum()
    n_significant = (price_bin_df['p_value'] < 0.05).sum()
    results['trade_size_consistency'] = {
        'bins_maker_larger': int(n_maker_larger),
        'bins_total': len(price_bin_df),
        'bins_significant_p05': int(n_significant),
        'overall_ratio': float(price_bin_df['ratio'].mean()),
    }

    print(f"\nSummary: Maker > Taker in {n_maker_larger}/{len(price_bin_df)} price bins")
    print(f"Significant (p<0.05) in {n_significant}/{len(price_bin_df)} price bins")

    # =========================================================================
    # TEST 2: YES/NO Asymmetry Statistical Significance
    # =========================================================================
    print("\n" + "=" * 70)
    print("TEST 2: YES/NO Asymmetry by Price Level")
    print("=" * 70)

    yes_no_by_price = con.execute(
        f"""
        WITH resolved_markets AS (
            SELECT ticker, result
            FROM '{markets_dir}/*.parquet'
            WHERE status = 'finalized' AND result IN ('yes', 'no')
        ),
        yes_trades AS (
            SELECT
                t.yes_price AS price,
                CASE WHEN m.result = 'yes' THEN 1.0 ELSE 0.0 END AS won,
                t.count AS contracts
            FROM '{trades_dir}/*.parquet' t
            INNER JOIN resolved_markets m ON t.ticker = m.ticker
            WHERE t.taker_side = 'yes'
        ),
        no_trades AS (
            SELECT
                t.no_price AS price,
                CASE WHEN m.result = 'no' THEN 1.0 ELSE 0.0 END AS won,
                t.count AS contracts
            FROM '{trades_dir}/*.parquet' t
            INNER JOIN resolved_markets m ON t.ticker = m.ticker
            WHERE t.taker_side = 'no'
        )
        SELECT 'YES' AS side, price, won, contracts FROM yes_trades
        UNION ALL
        SELECT 'NO' AS side, price, won, contracts FROM no_trades
        """
    ).df()

    # Test at key price points
    test_prices = [1, 2, 3, 4, 5, 10, 15, 20, 25, 50, 75, 80, 85, 90, 95, 96, 97, 98, 99]
    asymmetry_results = []

    for price in test_prices:
        yes_data = yes_no_by_price[(yes_no_by_price['side'] == 'YES') & (yes_no_by_price['price'] == price)]
        no_data = yes_no_by_price[(yes_no_by_price['side'] == 'NO') & (yes_no_by_price['price'] == price)]

        if len(yes_data) < 100 or len(no_data) < 100:
            continue

        # Weighted win rates
        yes_win_rate = (yes_data['won'] * yes_data['contracts']).sum() / yes_data['contracts'].sum()
        no_win_rate = (no_data['won'] * no_data['contracts']).sum() / no_data['contracts'].sum()

        # For significance: use contract-weighted proportions test
        yes_n = yes_data['contracts'].sum()
        no_n = no_data['contracts'].sum()
        yes_wins = (yes_data['won'] * yes_data['contracts']).sum()
        no_wins = (no_data['won'] * no_data['contracts']).sum()

        # Two-proportion z-test
        p_pooled = (yes_wins + no_wins) / (yes_n + no_n)
        se = np.sqrt(p_pooled * (1 - p_pooled) * (1/yes_n + 1/no_n))
        z_stat = (no_win_rate - yes_win_rate) / se if se > 0 else 0
        p_value = 2 * (1 - stats.norm.cdf(abs(z_stat)))  # two-tailed

        # EV calculation
        yes_ev = (yes_win_rate * 100 - price) / price * 100  # as percentage
        no_ev = (no_win_rate * 100 - price) / price * 100

        asymmetry_results.append({
            'price': price,
            'yes_win_rate': yes_win_rate,
            'no_win_rate': no_win_rate,
            'diff_pp': (no_win_rate - yes_win_rate) * 100,
            'yes_ev': yes_ev,
            'no_ev': no_ev,
            'ev_diff': no_ev - yes_ev,
            'z_statistic': z_stat,
            'p_value': p_value,
            'yes_n': int(yes_n),
            'no_n': int(no_n),
            'no_better': no_win_rate > yes_win_rate
        })

    asymmetry_df = pd.DataFrame(asymmetry_results)
    asymmetry_df.to_csv(fig_dir / "yes_no_asymmetry_significance.csv", index=False)

    print("\nYES/NO Win Rate Comparison at Key Prices:")
    print("-" * 110)
    print(f"{'Price':>6} {'YES Win%':>10} {'NO Win%':>10} {'Diff (pp)':>10} {'YES EV%':>10} {'NO EV%':>10} {'z-stat':>10} {'p-value':>12}")
    print("-" * 110)
    for _, row in asymmetry_df.iterrows():
        sig = "***" if row['p_value'] < 0.001 else "**" if row['p_value'] < 0.01 else "*" if row['p_value'] < 0.05 else ""
        print(f"{row['price']:>5}¢ {row['yes_win_rate']*100:>9.2f}% {row['no_win_rate']*100:>9.2f}% {row['diff_pp']:>+9.2f} {row['yes_ev']:>+9.1f}% {row['no_ev']:>+9.1f}% {row['z_statistic']:>9.2f} {row['p_value']:>11.2e}{sig}")

    n_no_better = asymmetry_df['no_better'].sum()
    n_significant = (asymmetry_df['p_value'] < 0.05).sum()
    results['yes_no_asymmetry'] = {
        'prices_no_better': int(n_no_better),
        'prices_total': len(asymmetry_df),
        'prices_significant': int(n_significant),
        'avg_ev_diff': float(asymmetry_df['ev_diff'].mean()),
    }

    print(f"\nSummary: NO outperforms YES at {n_no_better}/{len(asymmetry_df)} tested prices")
    print(f"Significant (p<0.05) at {n_significant}/{len(asymmetry_df)} prices")

    # =========================================================================
    # TEST 3: Category Gap Differences
    # =========================================================================
    print("\n" + "=" * 70)
    print("TEST 3: Category Gap Statistical Comparisons")
    print("=" * 70)

    # Load category data
    from util.categories import CATEGORY_SQL, get_group

    category_trades = con.execute(
        f"""
        WITH resolved_markets AS (
            SELECT ticker, event_ticker, result
            FROM '{markets_dir}/*.parquet'
            WHERE status = 'finalized' AND result IN ('yes', 'no')
        )
        SELECT
            {CATEGORY_SQL.replace('event_ticker', 'm.event_ticker')} AS category,
            CASE WHEN t.taker_side = 'yes' THEN t.yes_price ELSE t.no_price END AS taker_price,
            CASE WHEN t.taker_side = m.result THEN 1.0 ELSE 0.0 END AS taker_won,
            t.count AS contracts
        FROM '{trades_dir}/*.parquet' t
        INNER JOIN resolved_markets m ON t.ticker = m.ticker
        """
    ).df()

    category_trades['group'] = category_trades['category'].apply(get_group)

    # Compute taker excess return per trade for each category
    category_trades['taker_excess'] = category_trades['taker_won'] - category_trades['taker_price'] / 100

    # Get summary stats by category
    category_stats = category_trades.groupby('group').agg({
        'taker_excess': ['mean', 'std', 'count'],
        'contracts': 'sum'
    }).reset_index()
    category_stats.columns = ['group', 'mean_excess', 'std_excess', 'n_trades', 'contracts']
    category_stats['se'] = category_stats['std_excess'] / np.sqrt(category_stats['n_trades'])
    category_stats['ci_lower'] = category_stats['mean_excess'] - 1.96 * category_stats['se']
    category_stats['ci_upper'] = category_stats['mean_excess'] + 1.96 * category_stats['se']

    # Pairwise comparisons: Finance vs others
    finance_data = category_trades[category_trades['group'] == 'Finance']['taker_excess'].values
    pairwise_results = []

    for group in category_stats['group'].unique():
        if group == 'Finance':
            continue
        other_data = category_trades[category_trades['group'] == group]['taker_excess'].values

        if len(other_data) < 100:
            continue

        # Welch's t-test (unequal variances)
        t_stat, p_value = ttest_ind(finance_data, other_data, equal_var=False)
        d = cohens_d(finance_data, other_data)

        pairwise_results.append({
            'comparison': f"Finance vs {group}",
            'finance_mean': np.mean(finance_data) * 100,
            'other_mean': np.mean(other_data) * 100,
            'diff_pp': (np.mean(finance_data) - np.mean(other_data)) * 100,
            't_statistic': t_stat,
            'p_value': p_value,
            'cohens_d': d,
            'finance_n': len(finance_data),
            'other_n': len(other_data)
        })

    pairwise_df = pd.DataFrame(pairwise_results)
    pairwise_df = pairwise_df.sort_values('diff_pp', ascending=False)
    pairwise_df.to_csv(fig_dir / "category_pairwise_tests.csv", index=False)

    print("\nFinance vs Other Categories (Taker Excess Return):")
    print("-" * 100)
    print(f"{'Comparison':<25} {'Finance':>10} {'Other':>10} {'Diff (pp)':>10} {'t-stat':>10} {'p-value':>12} {'Cohen d':>10}")
    print("-" * 100)
    for _, row in pairwise_df.iterrows():
        sig = "***" if row['p_value'] < 0.001 else "**" if row['p_value'] < 0.01 else "*" if row['p_value'] < 0.05 else ""
        print(f"{row['comparison']:<25} {row['finance_mean']:>+9.2f}% {row['other_mean']:>+9.2f}% {row['diff_pp']:>+9.2f} {row['t_statistic']:>9.2f} {row['p_value']:>11.2e}{sig} {row['cohens_d']:>9.3f}")

    results['category_comparisons'] = {
        'all_significant_vs_finance': int((pairwise_df['p_value'] < 0.05).sum()),
        'total_comparisons': len(pairwise_df),
        'max_cohens_d': float(pairwise_df['cohens_d'].abs().max()),
    }

    # =========================================================================
    # TEST 4: Trade Size → Performance Regression
    # =========================================================================
    print("\n" + "=" * 70)
    print("TEST 4: Trade Size → Performance Relationship")
    print("=" * 70)

    trade_perf = con.execute(
        f"""
        WITH resolved_markets AS (
            SELECT ticker, result
            FROM '{markets_dir}/*.parquet'
            WHERE status = 'finalized' AND result IN ('yes', 'no')
        )
        SELECT
            t.count * (CASE WHEN t.taker_side = 'yes' THEN t.yes_price ELSE t.no_price END) / 100.0 AS trade_size,
            CASE WHEN t.taker_side = 'yes' THEN t.yes_price ELSE t.no_price END AS price,
            CASE WHEN t.taker_side = m.result THEN 1.0 ELSE 0.0 END AS won
        FROM '{trades_dir}/*.parquet' t
        INNER JOIN resolved_markets m ON t.ticker = m.ticker
        """
    ).df()

    # Compute excess return
    trade_perf['excess'] = trade_perf['won'] - trade_perf['price'] / 100
    trade_perf['log_size'] = np.log10(trade_perf['trade_size'].clip(lower=0.01))

    # Bin by log trade size for cleaner analysis
    trade_perf['size_bin'] = pd.cut(trade_perf['log_size'], bins=20)

    # Correlation tests
    # Sample for computational feasibility
    sample = trade_perf.sample(n=min(1000000, len(trade_perf)), random_state=42)

    pearson_r, pearson_p = pearsonr(sample['log_size'], sample['excess'])
    spearman_r, spearman_p = spearmanr(sample['log_size'], sample['excess'])

    # Binned regression
    binned = trade_perf.groupby('size_bin', observed=True).agg({
        'excess': 'mean',
        'log_size': 'mean',
        'trade_size': ['mean', 'count']
    }).reset_index()
    binned.columns = ['size_bin', 'mean_excess', 'mean_log_size', 'mean_size', 'n']
    binned = binned.dropna()

    # Weighted linear regression on binned data
    from numpy.polynomial import polynomial as P
    weights = np.sqrt(binned['n'])  # Weight by sqrt(n)
    coeffs = np.polyfit(binned['mean_log_size'], binned['mean_excess'], 1, w=weights)
    slope, intercept = coeffs

    # R-squared
    predicted = slope * binned['mean_log_size'] + intercept
    ss_res = np.sum(weights * (binned['mean_excess'] - predicted) ** 2)
    ss_tot = np.sum(weights * (binned['mean_excess'] - np.average(binned['mean_excess'], weights=weights)) ** 2)
    r_squared = 1 - ss_res / ss_tot

    print(f"\nCorrelation: Log(Trade Size) vs Excess Return")
    print(f"  Pearson r = {pearson_r:.4f} (p = {pearson_p:.2e})")
    print(f"  Spearman ρ = {spearman_r:.4f} (p = {spearman_p:.2e})")
    print(f"\nWeighted Linear Regression (binned):")
    print(f"  Slope = {slope*100:.4f} pp per log10($)")
    print(f"  Intercept = {intercept*100:.4f} pp")
    print(f"  R² = {r_squared:.4f}")
    print(f"  Interpretation: 10x increase in trade size → {slope*100:.2f} pp better performance")

    results['trade_size_regression'] = {
        'pearson_r': float(pearson_r),
        'pearson_p': float(pearson_p),
        'spearman_r': float(spearman_r),
        'spearman_p': float(spearman_p),
        'slope_pp_per_log10': float(slope * 100),
        'r_squared': float(r_squared),
    }

    binned.to_csv(fig_dir / "trade_size_performance_binned.csv", index=False)

    # =========================================================================
    # TEST 5: Maker Direction (YES vs NO) Significance
    # =========================================================================
    print("\n" + "=" * 70)
    print("TEST 5: Maker Direction Performance")
    print("=" * 70)

    maker_direction = con.execute(
        f"""
        WITH resolved_markets AS (
            SELECT ticker, result
            FROM '{markets_dir}/*.parquet'
            WHERE status = 'finalized' AND result IN ('yes', 'no')
        ),
        maker_yes AS (
            SELECT
                t.yes_price AS price,
                CASE WHEN m.result = 'yes' THEN 1.0 ELSE 0.0 END AS won,
                t.count AS contracts
            FROM '{trades_dir}/*.parquet' t
            INNER JOIN resolved_markets m ON t.ticker = m.ticker
            WHERE t.taker_side = 'no'  -- maker bought YES
        ),
        maker_no AS (
            SELECT
                t.no_price AS price,
                CASE WHEN m.result = 'no' THEN 1.0 ELSE 0.0 END AS won,
                t.count AS contracts
            FROM '{trades_dir}/*.parquet' t
            INNER JOIN resolved_markets m ON t.ticker = m.ticker
            WHERE t.taker_side = 'yes'  -- maker bought NO
        )
        SELECT 'YES' AS maker_side, price, won, contracts FROM maker_yes
        UNION ALL
        SELECT 'NO' AS maker_side, price, won, contracts FROM maker_no
        """
    ).df()

    # Test at price ranges
    price_ranges = [(1, 10), (11, 25), (26, 50), (51, 75), (76, 90), (91, 99)]
    direction_results = []

    for low, high in price_ranges:
        yes_data = maker_direction[(maker_direction['maker_side'] == 'YES') &
                                    (maker_direction['price'] >= low) &
                                    (maker_direction['price'] <= high)]
        no_data = maker_direction[(maker_direction['maker_side'] == 'NO') &
                                   (maker_direction['price'] >= low) &
                                   (maker_direction['price'] <= high)]

        if len(yes_data) < 100 or len(no_data) < 100:
            continue

        # Compute excess returns
        yes_data = yes_data.copy()
        no_data = no_data.copy()
        yes_data['excess'] = yes_data['won'] - yes_data['price'] / 100
        no_data['excess'] = no_data['won'] - no_data['price'] / 100

        # Weighted means
        yes_excess = (yes_data['excess'] * yes_data['contracts']).sum() / yes_data['contracts'].sum()
        no_excess = (no_data['excess'] * no_data['contracts']).sum() / no_data['contracts'].sum()

        # For significance: sample and run t-test
        yes_sample = np.repeat(yes_data['excess'].values, yes_data['contracts'].astype(int).clip(upper=100).values)
        no_sample = np.repeat(no_data['excess'].values, no_data['contracts'].astype(int).clip(upper=100).values)

        # Subsample if too large
        if len(yes_sample) > 100000:
            yes_sample = np.random.choice(yes_sample, 100000, replace=False)
        if len(no_sample) > 100000:
            no_sample = np.random.choice(no_sample, 100000, replace=False)

        t_stat, p_value = ttest_ind(no_sample, yes_sample, equal_var=False)
        d = cohens_d(no_sample, yes_sample)

        direction_results.append({
            'price_range': f"{low}-{high}¢",
            'yes_excess_pp': yes_excess * 100,
            'no_excess_pp': no_excess * 100,
            'diff_pp': (no_excess - yes_excess) * 100,
            't_statistic': t_stat,
            'p_value': p_value,
            'cohens_d': d,
            'yes_n': len(yes_data),
            'no_n': len(no_data),
            'no_better': no_excess > yes_excess
        })

    direction_df = pd.DataFrame(direction_results)
    direction_df.to_csv(fig_dir / "maker_direction_significance.csv", index=False)

    print("\nMaker Performance by Direction (YES vs NO):")
    print("-" * 100)
    print(f"{'Price Range':<12} {'YES (pp)':>10} {'NO (pp)':>10} {'Diff (pp)':>10} {'t-stat':>10} {'p-value':>12} {'Cohen d':>10}")
    print("-" * 100)
    for _, row in direction_df.iterrows():
        sig = "***" if row['p_value'] < 0.001 else "**" if row['p_value'] < 0.01 else "*" if row['p_value'] < 0.05 else ""
        print(f"{row['price_range']:<12} {row['yes_excess_pp']:>+9.2f} {row['no_excess_pp']:>+9.2f} {row['diff_pp']:>+9.2f} {row['t_statistic']:>9.2f} {row['p_value']:>11.2e}{sig} {row['cohens_d']:>9.3f}")

    n_no_better = direction_df['no_better'].sum()
    n_significant = (direction_df['p_value'] < 0.05).sum()
    results['maker_direction'] = {
        'ranges_no_better': int(n_no_better),
        'ranges_total': len(direction_df),
        'ranges_significant': int(n_significant),
    }

    print(f"\nSummary: Maker NO > Maker YES in {n_no_better}/{len(direction_df)} price ranges")
    print(f"Significant (p<0.05) in {n_significant}/{len(direction_df)} ranges")

    # =========================================================================
    # SAVE SUMMARY
    # =========================================================================
    with open(fig_dir / "statistical_tests_summary.json", "w") as f:
        json.dump(results, f, indent=2)

    print("\n" + "=" * 70)
    print("SUMMARY OF STATISTICAL TESTS")
    print("=" * 70)
    print(json.dumps(results, indent=2))
    print(f"\nAll results saved to {fig_dir}")


if __name__ == "__main__":
    main()
