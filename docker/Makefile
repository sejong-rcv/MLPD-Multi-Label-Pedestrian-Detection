IMAGE_NAME = mlpd
IMAGE_TAG = latest
PYTHON_VERSION = 3.7
CONDA_ENV_NAME = mlpd
UID := $(shell id -u)
USER_NAME := $(shell whoami)

docker-make:
	$(eval IMAGE_NAME := $(IMAGE_NAME):$(IMAGE_TAG))
	@nvidia-docker build -t ${IMAGE_NAME} \
  	--build-arg PYTHON_VERSION=${PYTHON_VERSION} \
  	--build-arg CONDA_ENV_NAME=${CONDA_ENV_NAME} \
  	-f ./Dockerfile \
  	.
