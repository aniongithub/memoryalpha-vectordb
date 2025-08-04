#!/bin/bash

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

set -euo pipefail

# By default 100 pages, can be overridden by setting MAX_PAGES in the environment.
# Set to -1 to process all pages.
MAX_PAGES="${MAX_PAGES:-100}"

pushd "$SCRIPT_DIR"
trap popd EXIT

docker build -t memoryalpha-pipeline -f pipeline/pipeline.Dockerfile .

# Use -it only if running in a TTY
if [ -t 1 ]; then
  DOCKER_RUN_FLAGS="-it"
else
  DOCKER_RUN_FLAGS=""
fi

docker run --rm $DOCKER_RUN_FLAGS -e MAX_PAGES="$MAX_PAGES" -v "$SCRIPT_DIR/data":/data memoryalpha-pipeline