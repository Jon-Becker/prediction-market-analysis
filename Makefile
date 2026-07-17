.PHONY: analyze run index record package lint format test e2e setup

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

e2e:
	docker compose -f compose.e2e.yaml up --build --abort-on-container-exit --exit-code-from runner; \
	status=$$?; docker compose -f compose.e2e.yaml down -v --remove-orphans; exit $$status

setup:
	bash scripts/install-tools.sh
	bash scripts/download.sh

%:
	@:
