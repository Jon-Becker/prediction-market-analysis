#!/bin/bash

cd /Users/alexlei/Desktop/CS\ code_GIT_AI/prediction-market-analysis
source .venv/bin/activate

analyses=(
    "win_rate_by_price"
    "maker_win_rate_by_direction"
    "win_rate_by_trade_size"
    "maker_taker_gap_over_time"
    "polymarket_calibration_by_bucket"
    "vwap_by_hour"
    "longshot_volume_share_over_time"
    "returns_by_hour"
    "market_types"
    "yes_vs_no_by_price"
    "maker_taker_returns_by_category"
    "mispricing_by_price"
    "kalshi_calibration_deviation_over_time"
    "trade_size_by_role"
    "maker_vs_taker_returns"
    "maker_returns_by_direction"
    "polymarket_volume_over_time"
    "ev_yes_vs_no"
    "statistical_tests"
    "volume_over_time"
    "meta_stats"
    "win_rate_by_price_animated"
)

for analysis in "${analyses[@]}"; do
    echo "Running: $analysis"
    gtimeout 600 uv run main.py analyze "$analysis"
    if [ $? -eq 124 ]; then
        echo "Timeout for $analysis"
    fi
done