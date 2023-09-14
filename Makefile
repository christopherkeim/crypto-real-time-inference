install:
	poetry install

data:
	poetry run python src/data.py

features:
	poetry run python src/feature_pipeline.py -l

test:
	#poetry run pytest .

format:	
	poetry run black .

lint:
	poetry run ruff .

refactor: format lint

deploy:
	#deploy goes here

all: install lint test format deploy
