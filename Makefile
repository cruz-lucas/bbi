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

test:
	python train.py --verbose=600

q-learning:
	python train.py --verbose=600 --config="bbi/config/goright_q_learning.yaml"

perfect:
	python train.py --verbose=600 --config="bbi/config/goright_perfect.yaml"

expect-2:
	python train.py --verbose=600 --config="bbi/config/goright_expected_h2.yaml"

expect-5:
	python train.py --verbose=600 --config="bbi/config/goright_expected_h5.yaml"

sampling-2:
	python train.py --verbose=600 --config="bbi/config/goright_sampling_h2.yaml"

sampling-5:
	python train.py --verbose=600 --config="bbi/config/goright_sampling_h5.yaml"

bounding-box:
	python train.py --verbose=600 --config="bbi/config/goright_bbi.yaml"

linear-bbi:
	python train.py --verbose=600 --config="bbi/config/goright_bbi_linear.yaml"

tree-bbi:
	python train.py --verbose=600 --config="bbi/config/goright_bbi_tree.yaml"

neural-bbi:
	python train.py --verbose=600 --config="bbi/config/goright_bbi_neural.yaml"

# Clean up by removing the Docker image
clean:
	docker rmi $(IMAGE_NAME)
