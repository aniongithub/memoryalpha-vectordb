#!/bin/bash

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

set -euo pipefail

pushd "$SCRIPT_DIR"
trap popd EXIT

docker build -t memoryalpha-pipeline -f pipeline/pipeline.Dockerfile .
docker run --rm -it -v "$SCRIPT_DIR/data":/data memoryalpha-pipeline