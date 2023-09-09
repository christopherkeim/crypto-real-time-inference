install:
	poetry install

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
