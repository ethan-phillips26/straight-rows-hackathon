# Single entry points. Recipes are the same commands the README lists, so nothing here is hidden.
#   make submission HANDLE=<github-handle> REPO=https://github.com/<github-handle>/<repo>
#   make test | make dev | make sweep | make validate
PYTHON ?= .venv/bin/python
DATA   ?= starter/data
OUT    ?= submission.json

.PHONY: submission validate test dev sweep explorer

submission:
	@test -n "$(HANDLE)" -a -n "$(REPO)" || (echo "usage: make submission HANDLE=<github-handle> REPO=<repo-url>"; exit 1)
	$(PYTHON) scripts/run_all.py --data $(DATA) --out $(OUT) --github-handle $(HANDLE) --repo-url $(REPO)
	$(PYTHON) starter/submission/validate_submission.py $(OUT)

validate:
	$(PYTHON) starter/submission/validate_submission.py $(OUT)

test:
	$(PYTHON) -m pytest -q

dev:
	$(PYTHON) scripts/run_dev.py

sweep:
	$(PYTHON) scripts/sweep_thresholds.py

explorer:
	$(PYTHON) scripts/make_explorer.py --data $(DATA) --detections $(OUT)
