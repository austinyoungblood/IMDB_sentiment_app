IMAGE_NAME=sentiment-fastapi
PORT?=8001
APP_PORT=8001

build:
	docker build -t $(IMAGE_NAME) .

run:
	docker run --rm -p $(PORT):$(APP_PORT) $(IMAGE_NAME)

dev:
	uvicorn main:app --host 127.0.0.1 --port $(PORT) --reload

clean:
	docker rmi $(IMAGE_NAME)
