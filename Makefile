.RECIPEPREFIX = >
.PHONY: up down logs health test lint fmt

up:
> docker compose up -d qdrant postgres

down:
> docker compose down

logs:
> docker compose logs -f

health:
> python scripts/healthcheck.py

test:
> pytest -q

lint:
> ruff check src tests

fmt:
> ruff format src tests