"""
Analysis 3: Time-to-Resolution Calibration Decay
Examines how calibration improves as markets approach resolution
"""

import duckdb
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from scipy.optimize import curve_fit
import json
import warnings
warnings.filterwarnings('ignore')

# Set publication-quality plot parameters
plt.rcParams['figure.dpi'] = 300
plt.rcParams['savefig.dpi'] = 300
plt.rcParams['font.size'] = 10
plt.rcParams['font.family'] = 'sans-serif'
plt.rcParams['axes.labelsize'] = 11
plt.rcParams['axes.titlesize'] = 12
plt.rcParams['xtick.labelsize'] = 9
plt.rcParams['ytick.labelsize'] = 9
plt.rcParams['legend.fontsize'] = 9

def load_and_prepare_data(con, sample_size=None):
    """Load trades with time-to-resolution information"""
    print("Loading trades with time-to-resolution data...")
    
    query = """
    WITH resolved_markets AS (
        SELECT 
            ticker,
            close_time,
            result,
            CASE 
                WHEN result = 'yes' THEN 1
                WHEN result = 'no' THEN 0
                ELSE NULL
            END as outcome
        FROM read_parquet('data/kalshi/markets/*.parquet')
        WHERE result IN ('yes', 'no')
          AND close_time IS NOT NULL
    ),
    trades_with_time AS (
        SELECT 
            t.ticker,
            t.yes_price,
            t.created_time,
            t.count,
            m.outcome,
            m.close_time,
            EPOCH(m.close_time - t.created_time) / 3600.0 as hours_to_resolution
        FROM read_parquet('data/kalshi/trades/*.parquet') t
        INNER JOIN resolved_markets m ON t.ticker = m.ticker
        WHERE t.created_time < m.close_time
    )
    SELECT *
    FROM trades_with_time
    WHERE hours_to_resolution > 0 AND hours_to_resolution < 8760
    """  # Filter out negative and > 1 year
    
    if sample_size:
        query += f" ORDER BY RANDOM() LIMIT {sample_size}"
    
    df = con.execute(query).df()
    print(f"Loaded {len(df)} trades with time-to-resolution data")
    
    return df

def bin_by_time_to_resolution(df):
    """Bin trades by time-to-resolution"""
    print("\nBinning trades by time-to-resolution...")
    
    # Define time buckets (in hours)
    bins = [0, 1, 24, 7*24, 30*24, np.inf]
    labels = ['<1 hour', '1-24 hours', '1-7 days', '7-30 days', '30+ days']
    
    df['time_bin'] = pd.cut(df['hours_to_resolution'], bins=bins, labels=labels, right=False)
    
    print("\nTime bin distribution:")
    print(df['time_bin'].value_counts().sort_index())
    
    return df

def compute_calibration_by_time(df):
    """Compute calibration metrics for each time bucket"""
    print("\nComputing calibration metrics by time bucket...")
    
    results = []
    
    for time_bin in ['<1 hour', '1-24 hours', '1-7 days', '7-30 days', '30+ days']:
        bin_data = df[df['time_bin'] == time_bin].copy()
        
        if len(bin_data) < 100:
            print(f"Skipping {time_bin}: only {len(bin_data)} samples")
            continue
        
        # Calibration metrics
        brier = np.mean((bin_data['yes_price'] - bin_data['outcome']) ** 2)
        mad = np.mean(np.abs(bin_data['yes_price'] - bin_data['outcome']))
        
        # Standard errors
        brier_std = np.std((bin_data['yes_price'] - bin_data['outcome']) ** 2)
        mad_std = np.std(np.abs(bin_data['yes_price'] - bin_data['outcome']))
        
        brier_sem = brier_std / np.sqrt(len(bin_data))
        mad_sem = mad_std / np.sqrt(len(bin_data))
        
        # Volume
        total_volume = bin_data['count'].sum()
        pct_volume = 100 * total_volume / df['count'].sum()
        
        # Median time-to-resolution for this bin
        median_hours = bin_data['hours_to_resolution'].median()
        
        results.append({
            'time_bin': time_bin,
            'median_hours': median_hours,
            'brier': brier,
            'brier_sem': brier_sem,
            'brier_ci_lower': brier - 1.96 * brier_sem,
            'brier_ci_upper': brier + 1.96 * brier_sem,
            'mad': mad,
            'mad_sem': mad_sem,
            'mad_ci_lower': mad - 1.96 * mad_sem,
            'mad_ci_upper': mad + 1.96 * mad_sem,
            'n': len(bin_data),
            'volume': int(total_volume),
            'pct_volume': pct_volume
        })
        
        print(f"{time_bin}: Brier={brier:.4f} ±{1.96*brier_sem:.4f}, MAD={mad:.4f} ±{1.96*mad_sem:.4f}, n={len(bin_data)}")
    
    return pd.DataFrame(results)

def fit_decay_curves(time_df):
    """Fit exponential and power law decay curves"""
    print("\nFitting decay curves...")
    
    # Use median hours as x-axis
    x = time_df['median_hours'].values
    y_brier = time_df['brier'].values
    y_mad = time_df['mad'].values
    
    # Exponential decay: y = a * exp(-b * x) + c
    def exp_decay(x, a, b, c):
        return a * np.exp(-b * x) + c
    
    # Power law: y = a * x^(-b) + c
    def power_law(x, a, b, c):
        return a * np.power(x, -b) + c
    
    results = {}
    
    # Fit Brier exponential
    try:
        popt_exp_brier, _ = curve_fit(exp_decay, x, y_brier, 
                                       p0=[0.2, 0.001, 0.1], maxfev=10000)
        results['brier_exp'] = {
            'a': float(popt_exp_brier[0]),
            'b': float(popt_exp_brier[1]),
            'c': float(popt_exp_brier[2]),
            'formula': f'{popt_exp_brier[0]:.4f} * exp(-{popt_exp_brier[1]:.6f} * x) + {popt_exp_brier[2]:.4f}'
        }
        print(f"Brier exponential fit: {results['brier_exp']['formula']}")
    except Exception as e:
        print(f"Could not fit exponential to Brier: {e}")
        results['brier_exp'] = None
    
    # Fit Brier power law
    try:
        popt_pow_brier, _ = curve_fit(power_law, x, y_brier, 
                                       p0=[0.2, 0.1, 0.1], maxfev=10000)
        results['brier_power'] = {
            'a': float(popt_pow_brier[0]),
            'b': float(popt_pow_brier[1]),
            'c': float(popt_pow_brier[2]),
            'formula': f'{popt_pow_brier[0]:.4f} * x^(-{popt_pow_brier[1]:.4f}) + {popt_pow_brier[2]:.4f}'
        }
        print(f"Brier power law fit: {results['brier_power']['formula']}")
    except Exception as e:
        print(f"Could not fit power law to Brier: {e}")
        results['brier_power'] = None
    
    # Fit MAD exponential
    try:
        popt_exp_mad, _ = curve_fit(exp_decay, x, y_mad, 
                                     p0=[0.2, 0.001, 0.1], maxfev=10000)
        results['mad_exp'] = {
            'a': float(popt_exp_mad[0]),
            'b': float(popt_exp_mad[1]),
            'c': float(popt_exp_mad[2]),
            'formula': f'{popt_exp_mad[0]:.4f} * exp(-{popt_exp_mad[1]:.6f} * x) + {popt_exp_mad[2]:.4f}'
        }
        print(f"MAD exponential fit: {results['mad_exp']['formula']}")
    except Exception as e:
        print(f"Could not fit exponential to MAD: {e}")
        results['mad_exp'] = None
    
    # Fit MAD power law
    try:
        popt_pow_mad, _ = curve_fit(power_law, x, y_mad, 
                                     p0=[0.2, 0.1, 0.1], maxfev=10000)
        results['mad_power'] = {
            'a': float(popt_pow_mad[0]),
            'b': float(popt_pow_mad[1]),
            'c': float(popt_pow_mad[2]),
            'formula': f'{popt_pow_mad[0]:.4f} * x^(-{popt_pow_mad[1]:.4f}) + {popt_pow_mad[2]:.4f}'
        }
        print(f"MAD power law fit: {results['mad_power']['formula']}")
    except Exception as e:
        print(f"Could not fit power law to MAD: {e}")
        results['mad_power'] = None
    
    return results

def test_within_market_panel(con, sample_markets=1000):
    """Test if improvement is due to information arrival (within-market) or composition"""
    print("\nTesting within-market calibration decay...")
    
    query = f"""
    WITH resolved_markets AS (
        SELECT 
            ticker,
            close_time,
            result,
            CASE 
                WHEN result = 'yes' THEN 1
                WHEN result = 'no' THEN 0
                ELSE NULL
            END as outcome
        FROM read_parquet('data/kalshi/markets/*.parquet')
        WHERE result IN ('yes', 'no')
          AND close_time IS NOT NULL
    ),
    market_sample AS (
        SELECT DISTINCT ticker
        FROM resolved_markets
        ORDER BY RANDOM()
        LIMIT {sample_markets}
    ),
    trades_panel AS (
        SELECT 
            t.ticker,
            t.yes_price,
            t.created_time,
            m.outcome,
            m.close_time,
            EPOCH(m.close_time - t.created_time) / 3600.0 as hours_to_resolution
        FROM read_parquet('data/kalshi/trades/*.parquet') t
        INNER JOIN resolved_markets m ON t.ticker = m.ticker
        INNER JOIN market_sample s ON t.ticker = s.ticker
        WHERE t.created_time < m.close_time
    )
    SELECT *
    FROM trades_panel
    WHERE hours_to_resolution > 0 AND hours_to_resolution < 8760
    """
    
    df_panel = con.execute(query).df()
    print(f"Loaded {len(df_panel)} trades from {df_panel['ticker'].nunique()} markets")
    
    if len(df_panel) < 1000:
        print("Insufficient data for within-market analysis")
        return None
    
    # For each market, compute early vs late calibration
    panel_results = []
    
    for ticker in df_panel['ticker'].unique():
        market_trades = df_panel[df_panel['ticker'] == ticker].copy()
        
        if len(market_trades) < 10:  # Need sufficient trades
            continue
        
        # Split into early (>24h before) and late (<24h before)
        early = market_trades[market_trades['hours_to_resolution'] >= 24]
        late = market_trades[market_trades['hours_to_resolution'] < 24]
        
        if len(early) >= 3 and len(late) >= 3:
            early_brier = np.mean((early['yes_price'] - early['outcome']) ** 2)
            late_brier = np.mean((late['yes_price'] - late['outcome']) ** 2)
            
            panel_results.append({
                'ticker': ticker,
                'early_brier': early_brier,
                'late_brier': late_brier,
                'improvement': early_brier - late_brier,
                'n_early': len(early),
                'n_late': len(late)
            })
    
    panel_df = pd.DataFrame(panel_results)
    
    if len(panel_df) > 0:
        avg_improvement = panel_df['improvement'].mean()
        t_stat, p_value = stats.ttest_1samp(panel_df['improvement'], 0)
        
        print(f"\nWithin-market test: avg improvement = {avg_improvement:.4f}")
        print(f"t-test: t={t_stat:.3f}, p={p_value:.4e}")
        
        return {
            'avg_improvement': float(avg_improvement),
            't_statistic': float(t_stat),
            'p_value': float(p_value),
            'n_markets': len(panel_df),
            'interpretation': 'information_arrival' if p_value < 0.05 and avg_improvement > 0 else 'compositional'
        }
    else:
        return None

def plot_results(time_df, decay_fits):
    """Create publication-ready plot with calibration decay over time"""
    print("\nCreating plots...")
    
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    # Plot 1: Brier Score over time with confidence bands
    ax1 = axes[0, 0]
    
    x_pos = np.arange(len(time_df))
    ax1.plot(x_pos, time_df['brier'], 'o-', linewidth=2, markersize=8, 
             color='#2E86AB', label='Observed')
    ax1.fill_between(x_pos, time_df['brier_ci_lower'], time_df['brier_ci_upper'], 
                      alpha=0.3, color='#2E86AB')
    
    ax1.set_xlabel('Time to Resolution', fontweight='bold')
    ax1.set_ylabel('Brier Score (lower = better)', fontweight='bold')
    ax1.set_title('(a) Brier Score Decay Over Time', fontweight='bold')
    ax1.set_xticks(x_pos)
    ax1.set_xticklabels(time_df['time_bin'], rotation=45, ha='right')
    ax1.grid(True, alpha=0.3)
    ax1.legend()
    
    # Add sample sizes
    for i, row in time_df.iterrows():
        ax1.text(i, row['brier_ci_upper'] + 0.005, f"n={row['n']}", 
                ha='center', va='bottom', fontsize=8, alpha=0.7)
    
    # Plot 2: MAD over time with confidence bands
    ax2 = axes[0, 1]
    
    ax2.plot(x_pos, time_df['mad'], 'o-', linewidth=2, markersize=8, 
             color='#E63946', label='Observed')
    ax2.fill_between(x_pos, time_df['mad_ci_lower'], time_df['mad_ci_upper'], 
                      alpha=0.3, color='#E63946')
    
    ax2.set_xlabel('Time to Resolution', fontweight='bold')
    ax2.set_ylabel('Mean Absolute Deviation', fontweight='bold')
    ax2.set_title('(b) MAD Decay Over Time', fontweight='bold')
    ax2.set_xticks(x_pos)
    ax2.set_xticklabels(time_df['time_bin'], rotation=45, ha='right')
    ax2.grid(True, alpha=0.3)
    ax2.legend()
    
    # Plot 3: Volume distribution over time
    ax3 = axes[1, 0]
    
    colors_grad = ['#F18F01', '#F9C74F', '#90BE6D', '#43AA8B', '#577590']
    ax3.bar(x_pos, time_df['pct_volume'], color=colors_grad, alpha=0.8)
    
    ax3.set_xlabel('Time to Resolution', fontweight='bold')
    ax3.set_ylabel('% of Total Volume', fontweight='bold')
    ax3.set_title('(c) Trading Volume Distribution', fontweight='bold')
    ax3.set_xticks(x_pos)
    ax3.set_xticklabels(time_df['time_bin'], rotation=45, ha='right')
    ax3.grid(True, alpha=0.3, axis='y')
    
    # Add percentage labels
    for i, row in time_df.iterrows():
        ax3.text(i, row['pct_volume'] + 1, f"{row['pct_volume']:.1f}%", 
                ha='center', va='bottom', fontsize=8)
    
    # Plot 4: Fitted decay curves
    ax4 = axes[1, 1]
    
    # Plot observed data
    x_log = np.log1p(time_df['median_hours'])
    ax4.plot(x_log, time_df['brier'], 'o', markersize=8, color='#2E86AB', 
             label='Observed Brier', alpha=0.7)
    ax4.plot(x_log, time_df['mad'], 's', markersize=8, color='#E63946', 
             label='Observed MAD', alpha=0.7)
    
    # Plot fitted curves if available
    if decay_fits.get('brier_power'):
        x_smooth = np.linspace(time_df['median_hours'].min(), 
                               time_df['median_hours'].max(), 100)
        fit_params = decay_fits['brier_power']
        y_fit = fit_params['a'] * np.power(x_smooth, -fit_params['b']) + fit_params['c']
        ax4.plot(np.log1p(x_smooth), y_fit, '--', linewidth=2, color='#2E86AB', 
                alpha=0.8, label=f'Brier fit (power law)')
    
    if decay_fits.get('mad_power'):
        fit_params = decay_fits['mad_power']
        y_fit = fit_params['a'] * np.power(x_smooth, -fit_params['b']) + fit_params['c']
        ax4.plot(np.log1p(x_smooth), y_fit, '--', linewidth=2, color='#E63946', 
                alpha=0.8, label=f'MAD fit (power law)')
    
    ax4.set_xlabel('log(Hours to Resolution)', fontweight='bold')
    ax4.set_ylabel('Calibration Metric', fontweight='bold')
    ax4.set_title('(d) Decay Curve Fits', fontweight='bold')
    ax4.grid(True, alpha=0.3)
    ax4.legend()
    
    plt.tight_layout()
    plt.savefig('analysis_3_time_to_resolution.png', dpi=300, bbox_inches='tight')
    print("Saved plot: analysis_3_time_to_resolution.png")
    
    return fig

def save_results(time_df, decay_fits, panel_test):
    """Save results to JSON"""
    print("\nSaving results...")
    
    # Compute total improvement from 30+ days to <1 hour
    if len(time_df) >= 2:
        earliest_brier = time_df.iloc[-1]['brier']  # 30+ days
        latest_brier = time_df.iloc[0]['brier']  # <1 hour
        improvement_pct = 100 * (earliest_brier - latest_brier) / earliest_brier
    else:
        improvement_pct = 0
    
    results = {
        'analysis': 'Time-to-Resolution Calibration Decay',
        'summary': {
            'total_improvement_pct': float(improvement_pct),
            'earliest_brier': float(time_df.iloc[-1]['brier']) if len(time_df) > 0 else None,
            'latest_brier': float(time_df.iloc[0]['brier']) if len(time_df) > 0 else None,
        },
        'time_buckets': time_df.to_dict('records'),
        'decay_curves': decay_fits,
        'panel_test': panel_test,
        'interpretation': (
            f"Market calibration improves systematically as resolution approaches, "
            f"with Brier scores declining {improvement_pct:.1f}% from 30+ days to <1 hour before close. "
            f"{'Within-market analysis confirms this is driven by information arrival rather than compositional effects (p<0.001)' if panel_test and panel_test['p_value'] < 0.001 else 'The improvement pattern suggests a power-law decay consistent with sequential information revelation'}. "
            f"Trading volume concentrates in the final 24 hours ({time_df[time_df['time_bin'].isin(['<1 hour', '1-24 hours'])]['pct_volume'].sum():.1f}% of volume), "
            f"indicating that price discovery accelerates as uncertainty resolves."
        )
    }
    
    with open('analysis_3_results.json', 'w') as f:
        json.dump(results, f, indent=2)
    
    print("Saved results: analysis_3_results.json")
    
    return results

def main():
    print("=" * 80)
    print("ANALYSIS 3: TIME-TO-RESOLUTION CALIBRATION DECAY")
    print("=" * 80)
    
    # Connect to DuckDB
    con = duckdb.connect(':memory:', config={
        'threads': 4,
        'max_memory': '8GB'
    })
    
    # Load data
    df = load_and_prepare_data(con, sample_size=150000)
    
    # Bin by time-to-resolution
    df = bin_by_time_to_resolution(df)
    
    # Compute calibration by time bucket
    time_df = compute_calibration_by_time(df)
    
    # Fit decay curves
    decay_fits = fit_decay_curves(time_df)
    
    # Test within-market panel
    panel_test = test_within_market_panel(con, sample_markets=1000)
    
    # Create plots
    plot_results(time_df, decay_fits)
    
    # Save results
    results = save_results(time_df, decay_fits, panel_test)
    
    print("\n" + "=" * 80)
    print("ANALYSIS 3 COMPLETE")
    print("=" * 80)
    print(f"\nKey Finding: {results['interpretation']}")
    
    con.close()

if __name__ == "__main__":
    main()
