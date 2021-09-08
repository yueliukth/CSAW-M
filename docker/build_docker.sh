#!/bin/bash

IMAGE_NAME=""  # your desired docker image name
WANDB_KEY=""  # your WANDB key
USER_NAME=""  # username to be used inside docker
Dockerfile=""  # Dockerfile specified in arg

helper()
{
    echo
    echo -e "$(tput bold)Helper for the docker builder$(tput sgr0)"
    echo "-----------------------------"
    echo "  -h  --help            Print this usage and exit."
    echo "  -d  --docker_file     Specify the Dockerfile to be used."
    echo "  -i  --image_name      Specify the Image name to be create."
    echo
}

# get args
while [[ $# -gt 0 ]]
do
    key="$1"
    case $key in
        -h|--help)
            helper
            exit 0
            ;;
        -d|--docker_file)
            Dockerfile="$2"
            shift
            shift
            ;;
         -i|--image_name)
            IMAGE_NAME="$2"
            shift
            shift
            ;;
        *)
            echo "Argument '$key' is not defined. Terminating..."
            exit 1
            ;;
    esac
done

# do some checks
if [[ $Dockerfile == "" ]]; then
    echo "Please specify Dockerfile properly. Terminating..."
    exit 1
fi
if [[ ! -f $Dockerfile ]]; then
    echo "Dockerfile does not exist. Terminating..."
    exit 1
fi
if [[ $IMAGE_NAME  == "" ]]; then
    echo "Please specify Image name properly. Terminating..."
    exit 1
fi

echo
echo "Dockerfile: $(tput bold)$Dockerfile$(tput sgr0)"
echo "IMAGE_NAME: $(tput bold)$IMAGE_NAME$(tput sgr0)"
echo

docker build -f $Dockerfile -t $IMAGE_NAME \
                  --build-arg UID=$(id -u) \
                  --build-arg GID=$(id -g) \
                  --build-arg USER=$USER_NAME \
                  --build-arg GROUP=$(id -g -n) \
                  --build-arg WANDB_KEY=$WANDB_KEY .
