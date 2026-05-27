#!/bin/bash
# Build all MoleculeForge Docker images
set -euo pipefail

IMAGES=(
  "mf-base:infra/docker/base/Dockerfile.base"
  "mf-chem:infra/docker/base/Dockerfile.chem"
  "mf-generator:infra/docker/base/Dockerfile.generator"
  "mf-oracle:infra/docker/base/Dockerfile.oracle"
  "mf-agent:infra/docker/base/Dockerfile.agent"
)

for img_def in "${IMAGES[@]}"; do
  tag="${img_def%%:*}"
  dockerfile="${img_def##*:}"
  echo "Building $tag from $dockerfile..."
  docker build -f "$dockerfile" -t "moleculeforge/$tag:latest" .
done

echo "All images built successfully."
