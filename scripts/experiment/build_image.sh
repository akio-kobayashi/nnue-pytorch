#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"

IMAGE_NAME="${IMAGE_NAME:-nnue-pytorch-cu124}"
IMAGE_TAG="${IMAGE_TAG:-latest}"

cd "${REPO_ROOT}"
docker build -f docker/Dockerfile.cuda124 -t "${IMAGE_NAME}:${IMAGE_TAG}" .
echo "Built Docker image: ${IMAGE_NAME}:${IMAGE_TAG}"
