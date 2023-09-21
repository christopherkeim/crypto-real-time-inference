install:
	poetry install

rawdata:
	poetry run python src/data.py

features:
	poetry run python src/feature_pipeline.py

train:
	poetry run python src/train.py

nntrain:
	poetry run python src/nn_train.py

predict:
	poetry run python src/ml_inference.py

nnpredict:
	poetry run python src/nn_inference.py

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
