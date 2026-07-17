.PHONY: analyze run index record package lint format test setup

RUN = uv run main.py

analyze:
	$(RUN) analyze

run:
	$(RUN) analyze $(filter-out $@,$(MAKECMDGOALS))

index:
	$(RUN) index

record:
	$(RUN) record $(filter-out $@,$(MAKECMDGOALS))

package:
	$(RUN) package

lint:
	uv run ruff check .
	uv run ruff format --check .

format:
	uv run ruff check --fix .
	uv run ruff format .

test:
	uv run pytest tests/ -v

setup:
	bash scripts/install-tools.sh
	bash scripts/download.sh

%:
	@:
