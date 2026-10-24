.PHONY: install lint test build eval up down
install:  ## dev setup
	pip install -r requirements-dev.txt && pre-commit install
lint:
	ruff check . && ruff format --check .
test:
	pytest -q
build:    ## simulate, train, score -> data/
	python -m fleetpilot build
eval:     ## offline (scripted) evaluation; add LIVE=1 for real Claude
	python -m fleetpilot eval $(if $(LIVE),--live,) --out eval_report.json
up:
	docker compose up -d --build
down:
	docker compose down
