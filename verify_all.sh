#!/bin/bash
set -e

echo "=== VERIFICATION SUITE ==="
echo ""

echo "1. Linting..."
uv run ruff check cross_platform_validation.py rigorous_cross_platform_analysis.py polymarket_exploratory_analysis.py
echo "✓ Linting passed"
echo ""

echo "2. Compilation..."
python -m py_compile cross_platform_validation.py rigorous_cross_platform_analysis.py polymarket_exploratory_analysis.py
echo "✓ Compilation passed"
echo ""

echo "3. Import..."
python -c "import cross_platform_validation; import rigorous_cross_platform_analysis; import polymarket_exploratory_analysis"
echo "✓ Import passed"
echo ""

echo "4. Execution..."
python polymarket_exploratory_analysis.py > /dev/null 2>&1
echo "✓ Execution passed"
echo ""

echo "5. Scientific claims..."
python -c "import json; r=json.load(open('POLYMARKET_EXPLORATORY_RESULTS.json')); assert r['q1']['test']['replicates']; assert r['q2']['replicates']; assert sum(r['multiple_testing']['significant']) == 3"
echo "✓ Scientific claims verified"
echo ""

echo "=== ALL CHECKS PASSED ✅ ==="
