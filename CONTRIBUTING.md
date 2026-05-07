# Contributing

## Development setup

```bash
python -m venv .venv
source .venv/bin/activate
pip install -e ".[dev]"
```

## Code style

This project uses Black and isort:

```bash
black .
isort .
```

## Run tests

```bash
pytest
```

## Reproduce paper results

1. Download BigCloneBench from CodeXGLUE.
2. Place the JSONL file at `data/data.jsonl`.
3. Run:
   - `python indexing.py`
   - `python performance.py`
   - `python plots.py`
   - `python testcodebert.py`

## Add a new search backend

1. Implement index build/search functions in the benchmark scripts.
2. Add benchmark hooks using `benchmark_search` from `src/similarity_search/utils.py`.
3. Update docs (`README.md`, `docs/ARCHITECTURE.md`, `docs/RESULTS.md`).
4. Add or update tests.

## Pull request checklist

- [ ] Code formatted with Black/isort
- [ ] Tests pass locally
- [ ] Documentation updated
- [ ] Reproducibility steps verified
- [ ] No hardcoded secrets/credentials
