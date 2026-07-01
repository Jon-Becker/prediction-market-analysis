"""Run Analysis 3: External Forecasts Comparison"""
import sys
from pathlib import Path

# Add project to path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

from src.analysis.external_forecasts_comparison import ExternalForecastsComparisonAnalysis

print('='*80)
print('ANALYSIS 3: External Forecasts Comparison')
print('='*80)

# Create analysis with explicit paths
base_dir = project_root
analysis = ExternalForecastsComparisonAnalysis(
    trades_dir=base_dir / "data" / "kalshi" / "trades",
    markets_dir=base_dir / "data" / "kalshi" / "markets",
)

output = analysis.run()

# Save outputs
output_dir = base_dir / "output"
output_dir.mkdir(exist_ok=True, parents=True)

if output.figure:
    output.figure.savefig(output_dir / 'external_forecasts_comparison.png', dpi=300, bbox_inches='tight')
    output.figure.savefig(output_dir / 'external_forecasts_comparison.pdf', bbox_inches='tight')
    print(f'\nFigures saved to {output_dir}')

if output.data is not None and not output.data.empty:
    output.data.to_csv(output_dir / 'external_forecasts_summary.csv', index=False)
    print(f'Summary statistics saved to {output_dir}')

print('\n✓ Analysis 3 complete!')
