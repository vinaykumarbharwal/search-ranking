.PHONY: install train run run-advanced test clean web benchmark evaluate

install:
	pip install -r requirements.txt

train:
	python src/train_native.py

run:
	uvicorn api.main_native:app --reload --host 0.0.0.0 --port 8000

run-advanced:
	uvicorn api.advanced_api:app --reload --host 0.0.0.0 --port 8001

test:
	pytest tests/ -v

clean:
	rm -rf models/*.pkl models/*.json
	rm -rf __pycache__ */__pycache__

web:
	open http://localhost:8000/docs

benchmark:
	python src/benchmark.py

evaluate:
	python src/evaluation.py
