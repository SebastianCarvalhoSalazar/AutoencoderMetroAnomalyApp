PY ?= python
.PHONY: install install-research test test-all reproduce app

install:            ## runtime + dev
	$(PY) -m pip install -r requirements-dev.txt && $(PY) -m pip install -e . --no-deps
install-research:   ## + TensorFlow (comparativas con autoencoder, app legacy)
	$(PY) -m pip install -r requirements-research.txt && $(PY) -m pip install -e . --no-deps
test:               ## suite rápida (sin TF ni navegador)
	$(PY) -m pytest -m "not research and not e2e" --cov=metro_anomaly --cov-report=term-missing
test-all:
	$(PY) -m pytest
reproduce:          ## pipeline completo: datos -> selección -> holdout -> artefacto -> informes
	$(PY) scripts/reproduce.py
app:
	$(PY) -m streamlit run app.py
