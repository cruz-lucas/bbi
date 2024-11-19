CUDA := 12.6.0
TAG := devel
OS := ubuntu20.04
IMAGE_NAME := bbi
CONTAINER_NAME := bbi
WORKDIR_PATH := /workspace

# Build the Docker image
build:
	docker build \
		--build-arg CUDA=$(CUDA) \
		--build-arg TAG=$(TAG) \
		--build-arg OS=$(OS) \
		--build-arg WORKDIR_PATH=$(WORKDIR_PATH) \
		-t $(IMAGE_NAME) .

# Run the Docker container with the current folder attached as a volume
run:
	docker run --rm -it \
		--name $(CONTAINER_NAME) \
		-v $(shell pwd):$(WORKDIR_PATH) \
		-u $(USER_ID):$(GROUP_ID) \
		$(IMAGE_NAME)

train:
	python train.py --verbose=600

# Clean up by removing the Docker image
clean:
	docker rmi $(IMAGE_NAME)
