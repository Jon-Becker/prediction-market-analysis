.PHONY: backfill backfill-trades analysis analyze setup teardown build

RUN = uv run main.py

backfill:
	$(RUN) backfill

backfill-trades:
	$(RUN) backfill-trades

setup:
	$(RUN) setup

teardown:
	$(RUN) teardown

analysis:
	$(RUN) setup
	$(RUN) analysis
	$(RUN) teardown

analyze:
	$(RUN) setup
	$(RUN) analysis $(filter-out $@,$(MAKECMDGOALS))
	$(RUN) teardown

%:
	@:
