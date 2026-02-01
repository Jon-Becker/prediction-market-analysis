.PHONY: analyze index package lint format

RUN = uv run main.py

analyze:
	$(RUN) analyze

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

%:
	@:
