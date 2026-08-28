.PHONY: help venv build build-debug install-dev clean test verify-docs fuzz fuzz-long benchmark benchmark-report verify \
        dist sdist wheel upload-test upload check-dist \
        clean-build clean-pyc clean-venv clean-all

VENV := .venv
VENV_READY := $(VENV)/.ready
PYTHON := $(VENV)/bin/python
PIP := $(VENV)/bin/pip
BUILD := $(PYTHON) -m build
TWINE := $(PYTHON) -m twine

help:
	@echo "spork-pds - Makefile targets"
	@echo ""
	@echo "Setup:"
	@echo "  venv             - Create the development virtual environment"
	@echo "  build            - Build the C extension in place"
	@echo "  build-debug      - Build with debug flags and available sanitizers"
	@echo "  install-dev      - Install the project in editable mode"
	@echo ""
	@echo "Testing:"
	@echo "  test             - Run the pytest suite"
	@echo "  verify-docs      - Execute documentation examples"
	@echo "  fuzz             - Run 1,000 vector fuzz examples"
	@echo "  fuzz-long        - Run 50,000 vector fuzz examples"
	@echo "  benchmark        - Run the benchmark suite (BENCH_ARGS='...')"
	@echo "  benchmark-report - Generate a Markdown benchmark report (REPORT_ARGS='...')"
	@echo "  verify           - Run tests and validate distributions"
	@echo ""
	@echo "Packaging:"
	@echo "  dist             - Build source and wheel distributions"
	@echo "  sdist            - Build the source distribution"
	@echo "  wheel            - Build a wheel"
	@echo "  check-dist       - Validate distributions with twine"
	@echo "  upload-test      - Upload distributions to TestPyPI"
	@echo "  upload           - Upload distributions to PyPI"
	@echo ""
	@echo "Cleanup:"
	@echo "  clean            - Remove build artifacts and caches"
	@echo "  clean-venv       - Remove the virtual environment"
	@echo "  clean-all        - Remove all generated files"

# ============================================================================
# Setup
# ============================================================================

$(VENV_READY): pyproject.toml setup.py
	@echo "Creating virtual environment..."
	python3 -m venv $(VENV)
	$(PIP) install --upgrade pip
	$(PIP) install -e ".[dev]"
	@touch $(VENV_READY)
	@echo "✓ Virtual environment ready"

venv: $(VENV_READY)

build: $(VENV_READY)
	@set -e; \
	extension=$$(find . -maxdepth 1 -type f \( -name 'spork_pds*.so' -o -name 'spork_pds*.pyd' \) -print -quit); \
	if [ -z "$$extension" ] || [ pds.c -nt "$$extension" ]; then \
		echo "Building C extension..."; \
		$(PYTHON) setup.py build_ext --inplace; \
		echo "✓ C extension built"; \
	else \
		echo "C extension is up to date."; \
	fi

build-debug: $(VENV_READY) clean-build
	@echo "Building C extension with debug instrumentation..."
	@if $(CC) -fsanitize=address -fsanitize=undefined -x c -E - < /dev/null > /dev/null 2>&1; then \
		DEBUG_BUILD=1 \
		CFLAGS="-O1 -g -fno-omit-frame-pointer -fsanitize=address -fsanitize=undefined" \
		LDFLAGS="-fsanitize=address -fsanitize=undefined" \
		$(PYTHON) setup.py build_ext --inplace; \
	elif $(CC) -fsanitize=address -x c -E - < /dev/null > /dev/null 2>&1; then \
		DEBUG_BUILD=1 \
		CFLAGS="-O1 -g -fno-omit-frame-pointer -fsanitize=address" \
		LDFLAGS="-fsanitize=address" \
		$(PYTHON) setup.py build_ext --inplace; \
	elif $(CC) -fsanitize=undefined -x c -E - < /dev/null > /dev/null 2>&1; then \
		DEBUG_BUILD=1 \
		CFLAGS="-O1 -g -fno-omit-frame-pointer -fsanitize=undefined" \
		LDFLAGS="-fsanitize=undefined" \
		$(PYTHON) setup.py build_ext --inplace; \
	else \
		DEBUG_BUILD=1 CFLAGS="-O1 -g -fno-omit-frame-pointer" \
		$(PYTHON) setup.py build_ext --inplace; \
	fi
	@echo "✓ Debug C extension built"

install-dev: $(VENV_READY) build
	$(PIP) install -e ".[dev]"
	@echo "✓ Installed in development mode"

# ============================================================================
# Testing and benchmarks
# ============================================================================

test: build
	$(PYTHON) -m pytest

verify-docs: build
	@$(PYTHON) tools/verify_docs.py

fuzz: build
	$(PYTHON) -m tests.fuzzing --examples 1000 --steps 200

fuzz-long: build
	$(PYTHON) -m tests.fuzzing --examples 50000 --steps 200

benchmark: build
	$(PYTHON) tools/benchmark_pds.py $(BENCH_ARGS)

benchmark-report: build
	$(PYTHON) tools/generate_benchmark_report.py $(REPORT_ARGS)

verify: test verify-docs check-dist

# ============================================================================
# Packaging
# ============================================================================

dist: $(VENV_READY) clean-build
	$(BUILD)
	@ls -lh dist/

sdist: $(VENV_READY) clean-build
	$(BUILD) --sdist
	@ls -lh dist/*.tar.gz

wheel: $(VENV_READY) clean-build
	$(BUILD) --wheel
	@ls -lh dist/*.whl

check-dist: dist
	$(TWINE) check dist/*

upload-test: check-dist
	$(TWINE) upload --repository testpypi dist/*

upload: check-dist
	$(TWINE) upload dist/*

# ============================================================================
# Cleanup
# ============================================================================

clean-build:
	rm -rf build/ dist/ *.egg-info/
	rm -f spork_pds*.so spork_pds*.pyd
	find . -name '*.o' -delete 2>/dev/null || true

clean-pyc:
	find . -type d -name __pycache__ -exec rm -rf {} + 2>/dev/null || true
	find . -type f \( -name '*.pyc' -o -name '*.pyo' \) -delete 2>/dev/null || true
	rm -rf .pytest_cache/ .hypothesis/

clean-venv:
	rm -rf $(VENV)
	@echo "✓ Virtual environment removed"

clean: clean-build clean-pyc

clean-all: clean clean-venv
