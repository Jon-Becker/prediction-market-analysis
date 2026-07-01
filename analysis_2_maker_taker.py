"""
Analysis 2: Maker-Taker Divergence by Category
Examines whether informed makers price better than retail takers
"""

import duckdb
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
import json
import warnings
warnings.filterwarnings('ignore')

# Set publication-quality plot parameters
plt.rcParams['figure.dpi'] = 300
plt.rcParams['savefig.dpi'] = 300
plt.rcParams['font.size'] = 10
plt.rcParams['font.family'] = 'sans-serif'
plt.rcParams['axes.labelsize'] = 10
plt.rcParams['axes.titlesize'] = 11
plt.rcParams['xtick.labelsize'] = 9
plt.rcParams['ytick.labelsize'] = 9
plt.rcParams['legend.fontsize'] = 8

def extract_category(event_ticker):
    """Extract category from event ticker"""
    if pd.isna(event_ticker):
        return 'other'
    
    ticker_upper = event_ticker.upper()
    
    # Sports categories
    if any(x in ticker_upper for x in ['NFL', 'NBA', 'MLB', 'NHL', 'NCAA', 'FIFA', 'SOCCER', 'TENNIS']):
        return 'sports'
    # Politics
    elif any(x in ticker_upper for x in ['PRES', 'SENATE', 'HOUSE', 'ELECTION', 'CONGRESS', 'GOV']):
        return 'politics'
    # Finance/Economics
    elif any(x in ticker_upper for x in ['FED', 'GDP', 'CPI', 'INFLATION', 'RATE', 'STOCK', 'SPX', 'NASDAQ']):
        return 'finance'
    # Weather
    elif any(x in ticker_upper for x in ['TEMP', 'WEATHER', 'SNOW', 'RAIN', 'HURRICANE']):
        return 'weather'
    # Crypto
    elif any(x in ticker_upper for x in ['BTC', 'ETH', 'CRYPTO', 'BITCOIN', 'ETHEREUM']):
        return 'crypto'
    # Entertainment
    elif any(x in ticker_upper for x in ['MOVIE', 'OSCAR', 'GRAMMY', 'EMMY', 'BOX', 'ALBUM']):
        return 'entertainment'
    else:
        return 'other'

def load_and_prepare_data(con, sample_size=None):
    """Load trades and market data with maker/taker information"""
    print("Loading trades and market data...")
    
    # First, get a sample of tickers with resolved markets
    ticker_query = """
    SELECT ticker
    FROM read_parquet('data/kalshi/markets/*.parquet')
    WHERE result IN ('yes', 'no')
      AND yes_bid IS NOT NULL
      AND yes_ask IS NOT NULL
    ORDER BY RANDOM()
    LIMIT 10000
    """
    
    sample_tickers = con.execute(ticker_query).df()['ticker'].tolist()
    
    # Now load trades for those tickers
    query = f"""
    WITH resolved_markets AS (
        SELECT 
            ticker,
            event_ticker,
            result,
            yes_bid,
            yes_ask,
            CASE 
                WHEN result = 'yes' THEN 1
                WHEN result = 'no' THEN 0
                ELSE NULL
            END as outcome
        FROM read_parquet('data/kalshi/markets/*.parquet')
        WHERE ticker IN ({','.join("'" + t + "'" for t in sample_tickers[:1000])})
          AND result IN ('yes', 'no')
          AND yes_bid IS NOT NULL
          AND yes_ask IS NOT NULL
    ),
    trades_with_outcome AS (
        SELECT 
            t.ticker,
            t.yes_price,
            t.no_price,
            t.taker_side,
            t.count,
            m.outcome,
            m.yes_bid,
            m.yes_ask,
            m.event_ticker
        FROM read_parquet('data/kalshi/trades/*.parquet') t
        INNER JOIN resolved_markets m ON t.ticker = m.ticker
        WHERE t.taker_side IS NOT NULL
    )
    SELECT *
    FROM trades_with_outcome
    """
    
    if sample_size:
        query += f" ORDER BY RANDOM() LIMIT {sample_size}"
    
    df = con.execute(query).df()
    print(f"Loaded {len(df)} trades from resolved markets")
    
    # Extract category
    print("Categorizing markets...")
    df['category'] = df['event_ticker'].apply(extract_category)
    
    print("\nCategory distribution:")
    print(df['category'].value_counts())
    
    return df

def compute_maker_taker_prices(df):
    """Compute effective prices for makers and takers"""
    print("\nComputing maker/taker effective prices...")
    
    # Normalize prices to 0-1 range (they are stored as 0-100)
    df['yes_price'] = df['yes_price'] / 100.0
    df['no_price'] = df['no_price'] / 100.0
    df['yes_bid'] = df['yes_bid'] / 100.0
    df['yes_ask'] = df['yes_ask'] / 100.0
    
    # For each trade:
    # - If taker_side = 'yes', taker bought yes (paid yes_price), maker sold yes
    # - If taker_side = 'no', taker bought no (paid no_price), maker sold no
    
    # Taker's effective probability is the price they paid
    df['taker_prob'] = df.apply(
        lambda row: row['yes_price'] if row['taker_side'] == 'yes' else row['no_price'],
        axis=1
    )
    
    # Maker's effective probability is the opposite side
    # If taker bought yes at yes_price, maker sold yes, so maker's implied prob is 1 - yes_price
    # But actually, maker's position is the other side of the book
    # If taker bought yes, maker sold yes from the ask side
    df['maker_prob'] = df.apply(
        lambda row: 1 - row['yes_price'] if row['taker_side'] == 'yes' else 1 - row['no_price'],
        axis=1
    )
    
    return df

def compute_calibration_metrics(df):
    """Compute Brier Score and MAD for makers and takers by category"""
    print("\nComputing calibration metrics by category and role...")
    
    results = []
    
    categories = ['sports', 'politics', 'finance', 'weather', 'crypto', 'entertainment']
    
    for category in categories:
        cat_data = df[df['category'] == category].copy()
        
        if len(cat_data) < 100:  # Skip categories with too few samples
            print(f"Skipping {category}: only {len(cat_data)} samples")
            continue
        
        # Taker metrics
        taker_brier = np.mean((cat_data['taker_prob'] - cat_data['outcome']) ** 2)
        taker_mad = np.mean(np.abs(cat_data['taker_prob'] - cat_data['outcome']))
        taker_n = len(cat_data)
        
        # Maker metrics  
        maker_brier = np.mean((cat_data['maker_prob'] - cat_data['outcome']) ** 2)
        maker_mad = np.mean(np.abs(cat_data['maker_prob'] - cat_data['outcome']))
        maker_n = len(cat_data)
        
        # Spread
        avg_spread = np.mean(cat_data['yes_ask'] - cat_data['yes_bid'])
        
        results.append({
            'category': category,
            'role': 'taker',
            'brier': taker_brier,
            'mad': taker_mad,
            'spread': avg_spread,
            'n': taker_n
        })
        
        results.append({
            'category': category,
            'role': 'maker',
            'brier': maker_brier,
            'mad': maker_mad,
            'spread': avg_spread,
            'n': maker_n
        })
        
        print(f"{category}: Taker Brier={taker_brier:.4f}, Maker Brier={maker_brier:.4f}, Spread={avg_spread:.4f}")
    
    return pd.DataFrame(results)

def compute_spread_correlation(df):
    """Correlate bid-ask spreads with calibration quality"""
    print("\nComputing spread vs accuracy correlation...")
    
    # Aggregate by market (ticker)
    market_stats = df.groupby('ticker').agg({
        'yes_bid': 'mean',
        'yes_ask': 'mean',
        'taker_prob': lambda x: np.mean((x - df.loc[x.index, 'outcome']) ** 2),  # Brier
        'outcome': 'first'
    }).reset_index()
    
    market_stats['spread'] = market_stats['yes_ask'] - market_stats['yes_bid']
    market_stats['brier'] = market_stats['taker_prob']
    
    # Remove outliers
    market_stats = market_stats[
        (market_stats['spread'] > 0) & 
        (market_stats['spread'] < 0.5) &
        (market_stats['brier'].notna())
    ]
    
    if len(market_stats) > 0:
        corr, p_value = stats.spearmanr(market_stats['spread'], market_stats['brier'])
        print(f"Spread vs Brier correlation: rho={corr:.4f}, p={p_value:.4e}")
        return market_stats, corr, p_value
    else:
        return market_stats, 0, 1

def plot_results(metrics_df, spread_data, spread_corr, spread_p):
    """Create publication-ready 2x2 grid plot"""
    print("\nCreating plots...")
    
    fig = plt.figure(figsize=(14, 10))
    gs = fig.add_gridspec(2, 2, hspace=0.3, wspace=0.3)
    
    # Plot 1: Brier Score by Category and Role
    ax1 = fig.add_subplot(gs[0, 0])
    
    categories = metrics_df['category'].unique()
    x = np.arange(len(categories))
    width = 0.35
    
    makers = metrics_df[metrics_df['role'] == 'maker'].set_index('category')
    takers = metrics_df[metrics_df['role'] == 'taker'].set_index('category')
    
    ax1.bar(x - width/2, [makers.loc[cat, 'brier'] for cat in categories], 
            width, label='Maker', color='#2E86AB', alpha=0.8)
    ax1.bar(x + width/2, [takers.loc[cat, 'brier'] for cat in categories], 
            width, label='Taker', color='#E63946', alpha=0.8)
    
    ax1.set_xlabel('Category', fontweight='bold')
    ax1.set_ylabel('Brier Score (lower = better)', fontweight='bold')
    ax1.set_title('(a) Calibration by Role: Brier Score', fontweight='bold')
    ax1.set_xticks(x)
    ax1.set_xticklabels(categories, rotation=45, ha='right')
    ax1.legend()
    ax1.grid(True, alpha=0.3, axis='y')
    
    # Plot 2: MAD by Category and Role
    ax2 = fig.add_subplot(gs[0, 1])
    
    ax2.bar(x - width/2, [makers.loc[cat, 'mad'] for cat in categories], 
            width, label='Maker', color='#2E86AB', alpha=0.8)
    ax2.bar(x + width/2, [takers.loc[cat, 'mad'] for cat in categories], 
            width, label='Taker', color='#E63946', alpha=0.8)
    
    ax2.set_xlabel('Category', fontweight='bold')
    ax2.set_ylabel('Mean Absolute Deviation', fontweight='bold')
    ax2.set_title('(b) Calibration by Role: MAD', fontweight='bold')
    ax2.set_xticks(x)
    ax2.set_xticklabels(categories, rotation=45, ha='right')
    ax2.legend()
    ax2.grid(True, alpha=0.3, axis='y')
    
    # Plot 3: Spread vs Accuracy
    ax3 = fig.add_subplot(gs[1, 0])
    
    if len(spread_data) > 1000:
        # Sample for visualization
        spread_sample = spread_data.sample(n=1000, random_state=42)
    else:
        spread_sample = spread_data
    
    ax3.scatter(spread_sample['spread'], spread_sample['brier'], 
                alpha=0.3, s=5, color='#06D6A0')
    
    # Add trend line
    if len(spread_sample) > 1:
        z = np.polyfit(spread_sample['spread'], spread_sample['brier'], 1)
        p = np.poly1d(z)
        x_line = np.linspace(spread_sample['spread'].min(), spread_sample['spread'].max(), 100)
        ax3.plot(x_line, p(x_line), "r--", linewidth=2, alpha=0.8, 
                label=f'ρ={spread_corr:.3f}, p<0.001' if spread_p < 0.001 else f'ρ={spread_corr:.3f}')
    
    ax3.set_xlabel('Bid-Ask Spread', fontweight='bold')
    ax3.set_ylabel('Brier Score', fontweight='bold')
    ax3.set_title('(c) Spread vs. Calibration Quality', fontweight='bold')
    ax3.legend()
    ax3.grid(True, alpha=0.3)
    
    # Plot 4: Summary Table
    ax4 = fig.add_subplot(gs[1, 1])
    ax4.axis('off')
    
    # Create summary statistics table
    summary_data = []
    for cat in categories:
        maker_row = makers.loc[cat]
        taker_row = takers.loc[cat]
        
        brier_diff = taker_row['brier'] - maker_row['brier']
        mad_diff = taker_row['mad'] - maker_row['mad']
        
        summary_data.append([
            cat.capitalize(),
            f"{maker_row['brier']:.4f}",
            f"{taker_row['brier']:.4f}",
            f"{brier_diff:+.4f}",
            f"{maker_row['spread']:.3f}"
        ])
    
    table = ax4.table(cellText=summary_data,
                     colLabels=['Category', 'Maker\nBrier', 'Taker\nBrier', 
                               'Diff\n(T-M)', 'Avg\nSpread'],
                     cellLoc='center',
                     loc='center',
                     bbox=[0, 0, 1, 1])
    
    table.auto_set_font_size(False)
    table.set_fontsize(9)
    table.scale(1, 2)
    
    # Color header
    for i in range(5):
        table[(0, i)].set_facecolor('#2E86AB')
        table[(0, i)].set_text_props(weight='bold', color='white')
    
    # Color difference column based on sign
    for i, row in enumerate(summary_data, start=1):
        diff_val = float(row[3])
        if diff_val > 0:
            table[(i, 3)].set_facecolor('#FFE5E5')  # Light red - taker worse
        else:
            table[(i, 3)].set_facecolor('#E5FFE5')  # Light green - maker worse
    
    ax4.set_title('(d) Summary Statistics by Category', fontweight='bold', pad=20)
    
    plt.savefig('analysis_2_maker_taker_divergence.png', dpi=300, bbox_inches='tight')
    print("Saved plot: analysis_2_maker_taker_divergence.png")
    
    return fig

def save_results(metrics_df, spread_corr, spread_p):
    """Save results to JSON"""
    print("\nSaving results...")
    
    # Analyze maker vs taker performance
    maker_avg_brier = metrics_df[metrics_df['role'] == 'maker']['brier'].mean()
    taker_avg_brier = metrics_df[metrics_df['role'] == 'taker']['brier'].mean()
    
    results = {
        'analysis': 'Maker-Taker Divergence by Category',
        'summary': {
            'maker_avg_brier': float(maker_avg_brier),
            'taker_avg_brier': float(taker_avg_brier),
            'brier_difference': float(taker_avg_brier - maker_avg_brier),
            'spread_correlation': {
                'rho': float(spread_corr),
                'p_value': float(spread_p)
            }
        },
        'by_category': metrics_df.to_dict('records'),
        'interpretation': (
            f"Makers demonstrate {'superior' if maker_avg_brier < taker_avg_brier else 'inferior'} "
            f"calibration compared to takers (ΔBrier={taker_avg_brier - maker_avg_brier:.4f}), "
            f"suggesting that {'informed market makers price more accurately than retail takers, particularly in high-spread markets (ρ={spread_corr:.3f})' if maker_avg_brier < taker_avg_brier else 'retail takers may be better informed or makers are providing liquidity at suboptimal prices'}. "
            f"This asymmetry varies by category, with the largest divergence in "
            f"{metrics_df.groupby('category').apply(lambda x: (x[x['role']=='taker']['brier'].values[0] - x[x['role']=='maker']['brier'].values[0])).idxmax() if len(metrics_df) > 0 else 'N/A'} markets."
        )
    }
    
    with open('analysis_2_results.json', 'w') as f:
        json.dump(results, f, indent=2)
    
    print("Saved results: analysis_2_results.json")
    
    return results

def main():
    print("=" * 80)
    print("ANALYSIS 2: MAKER-TAKER DIVERGENCE BY CATEGORY")
    print("=" * 80)
    
    # Connect to DuckDB
    con = duckdb.connect(':memory:', config={
        'threads': 4,
        'max_memory': '8GB'
    })
    
    # Load data (reduced sample for performance)
    df = load_and_prepare_data(con, sample_size=50000)
    
    # Compute maker/taker prices
    df = compute_maker_taker_prices(df)
    
    # Compute calibration metrics
    metrics_df = compute_calibration_metrics(df)
    
    # Compute spread correlation
    spread_data, spread_corr, spread_p = compute_spread_correlation(df)
    
    # Create plots
    plot_results(metrics_df, spread_data, spread_corr, spread_p)
    
    # Save results
    results = save_results(metrics_df, spread_corr, spread_p)
    
    print("\n" + "=" * 80)
    print("ANALYSIS 2 COMPLETE")
    print("=" * 80)
    print(f"\nKey Finding: {results['interpretation']}")
    
    con.close()

if __name__ == "__main__":
    main()
