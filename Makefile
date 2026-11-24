PY ?= python
.PHONY: install install-research test test-all reproduce app

install:            ## runtime + dev
	$(PY) -m pip install -r requirements-dev.txt && $(PY) -m pip install -e . --no-deps
install-research:   ## + TensorFlow (comparativas con autoencoder, app legacy)
	$(PY) -m pip install -r requirements-research.txt && $(PY) -m pip install -e . --no-deps
test:               ## suite rápida (sin TF ni navegador)
	$(PY) -m pytest -m "not research and not e2e" --cov=metro_anomaly --cov-report=term-missing --cov-fail-under=80
test-all:
	$(PY) -m pytest
reproduce:          ## pipeline completo: datos -> selección -> holdout -> artefacto -> informes
	$(PY) scripts/reproduce.py
app:
	$(PY) -m streamlit run app.py

reproduce-research: ## incluye autoencoders (requiere install-research)
	$(PY) scripts/reproduce.py --research
determinism:        ## D9: dos ejecuciones en copias aisladas -> reports/determinismo.md
	$(PY) scripts/check_determinism.py
e2e:                ## requiere node + playwright
	$(PY) -m pytest -m e2e -s
