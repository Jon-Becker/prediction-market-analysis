.PHONY: analyze run index package lint format setup

RUN = uv run main.py

analyze:
	$(RUN) analyze

run:
	$(RUN) analyze $(filter-out $@,$(MAKECMDGOALS))

index:
	$(RUN) index

package:
	$(RUN) package

lint:
	uv run ruff check .
	uv run ruff format --check .

format:
	uv run ruff check --fix .
	uv run ruff format .

setup:
	bash scripts/install-tools.sh
	bash scripts/download.sh

%:
	@:
