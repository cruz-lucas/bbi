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

q-learning:
	python train.py --config_file="goright_qlearning" --n_seeds=20 --start_seed=30

perfect:
	python train.py --config_file="goright_perfect" --n_seeds=50 --start_seed=0

expect-2:
	python train.py --config_file="goright_expected_h2" --n_seeds=50 --start_seed=0

expect-5:
	python train.py --config_file="goright_expected_h5" --n_seeds=50 --start_seed=0

sampling-2:
	python train.py --config_file="goright_sampling_h2" --n_seeds=50 --start_seed=0

sampling-5:
	python train.py --config_file="goright_sampling_h5" --n_seeds=50 --start_seed=0

bounding-box:
	python train.py --config_file="goright_bbi" --n_seeds=10 --start_seed=10

linear-bbi:
	python train.py --config_file="goright_bbi_linear" --n_seeds=50 --start_seed=0

tree-bbi:
	python train.py --config_file="goright_bbi_tree" --n_seeds=50 --start_seed=0

neural-bbi:
	python train.py --config_file="goright_bbi_neural" --n_seeds=50 --start_seed=0

# Clean up by removing the Docker image
clean:
	docker rmi $(IMAGE_NAME)
