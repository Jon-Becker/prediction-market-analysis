.PHONY: analyze index package

RUN = uv run main.py

analyze:
	$(RUN) analyze

index:
	$(RUN) index

package:
	$(RUN) package

%:
	@:
