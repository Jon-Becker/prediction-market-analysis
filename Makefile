.PHONY: backfill backfill-trades analysis setup teardown

backfill:
	uv run main.py backfill

backfill-trades:
	uv run main.py backfill-trades

analysis:
	uv run main.py analysis

setup:
	uv run main.py setup

teardown:
	uv run main.py teardown
