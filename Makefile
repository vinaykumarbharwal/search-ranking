.PHONY: install train run test docker-up docker-down clean

install:
	pip install -r requirements.txt

train:
	python src/train.py

run:
	uvicorn api.main:app --reload --host 0.0.0.0 --port 8000

test:
	pytest tests/ -v

docker-up:
	docker-compose up -d

docker-down:
	docker-compose down

clean:
	rm -rf models/*.pkl models/*.json
	rm -rf __pycache__ */__pycache__

web:
	open http://localhost:8000/docs

benchmark:
	python src/benchmark.py

evaluate:
	python src/evaluation.py
