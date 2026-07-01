"""Run Analysis 1: Cross-Platform Price Agreement"""
import sys
from pathlib import Path

# Add project to path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

from src.analysis.cross_platform_price_agreement import CrossPlatformPriceAgreementAnalysis

print('='*80)
print('ANALYSIS 1: Cross-Platform Price Agreement')
print('='*80)

# Create analysis with explicit paths
base_dir = project_root
analysis = CrossPlatformPriceAgreementAnalysis(
    kalshi_trades_dir=base_dir / "data" / "kalshi" / "trades",
    kalshi_markets_dir=base_dir / "data" / "kalshi" / "markets",
    poly_trades_dir=base_dir / "data" / "polymarket" / "trades",
    poly_markets_dir=base_dir / "data" / "polymarket" / "markets",
)

output = analysis.run()

# Save outputs
output_dir = base_dir / "output"
output_dir.mkdir(exist_ok=True, parents=True)

if output.figure:
    output.figure.savefig(output_dir / 'cross_platform_price_agreement.png', dpi=300, bbox_inches='tight')
    output.figure.savefig(output_dir / 'cross_platform_price_agreement.pdf', bbox_inches='tight')
    print(f'\nFigures saved to {output_dir}')

if output.data is not None and not output.data.empty:
    output.data.to_csv(output_dir / 'cross_platform_summary_stats.csv', index=False)
    print(f'Summary statistics saved to {output_dir}')

print('\n✓ Analysis 1 complete!')
