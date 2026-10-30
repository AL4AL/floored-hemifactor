.PHONY: preflight install freeze lint-example phase1 test run-all

install:
	python -m pip install -e ".[dev]"

preflight:
	bash scripts/preflight.sh

freeze:
	python scripts/freeze_model.py $(GGUF_ARGS)

lint-example:
	PYTHONPATH=src python -m hemifactor.cli lint-claim "we losslessly fit exact GGUF into 6GB with spare FLOPs"

phase1:
	PYTHONPATH=src python -m hemifactor.cli run 1

test:
	PYTHONPATH=src python -m pytest -q

run-all:
	PYTHONPATH=src python -m hemifactor.cli run all
