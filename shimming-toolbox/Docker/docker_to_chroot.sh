#!/bin/bash
# This script creates chroot image (.img) from the Docker file.

if [[ "$#" -ne 2 && "$#" -ne 3 ]]; then
    echo "Wrong number of arguments"
    echo "Syntax: ./docker_to_chroot.sh <docker_image_name> <chroot_file_name> [buffer_mb]"
    exit 1
fi

SCRIPT_DIR=$(dirname "$(realpath "$0")")
DOCKER_NAME=${1}
CHROOT_FILE=${2}
EXPORT_FILE=docker-export.tar
BUFFER_MB=${3:-50}

# Check if Docker is installed
if ! command -v docker &> /dev/null; then
    echo "Docker is not installed. Please install Docker and try again."
    exit 1
fi

# Create a Docker image from the Dockerfile
echo ----------------------------------------------------------------------
echo Creating Docker image ${DOCKER_NAME} from Dockerfile
echo ----------------------------------------------------------------------

docker build --platform linux/amd64 -t ${DOCKER_NAME} ${SCRIPT_DIR}

# Create a Docker container and export to a .tar file
echo ----------------------------------------------------------------------
echo Exporting Docker image ${DOCKER_NAME} to ${EXPORT_FILE}
echo ----------------------------------------------------------------------

docker create --name tmpimage ${DOCKER_NAME}
docker export -o ${EXPORT_FILE} tmpimage
docker rm tmpimage

# Run a privileged Docker to create the chroot file
docker run -it --rm          \
           --privileged=true \
           -v $(pwd):/share  \
           ubuntu            \
           /bin/bash -c "sed -i -e 's/\r//g' /share/docker_tar_to_chroot.sh && /share/docker_tar_to_chroot.sh /share/${EXPORT_FILE} /share/${CHROOT_FILE} ${BUFFER_MB}"

rm ${EXPORT_FILE}
