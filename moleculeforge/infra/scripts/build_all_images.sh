#!/bin/bash
# Build all MoleculeForge Docker images
set -euo pipefail

IMAGES=(
  "base:infra/docker/base/Dockerfile.base"
  "chem:infra/docker/base/Dockerfile.chem"
  "generator:infra/docker/base/Dockerfile.generator"
  "oracle:infra/docker/base/Dockerfile.oracle"
  "agent-runtime:infra/docker/base/Dockerfile.agent"
)

publish_registry="$(printf '%s' "${PUBLISH_REGISTRY:-}" | tr '[:upper:]' '[:lower:]')"
publish_tag="${PUBLISH_TAG:-latest}"

for img_def in "${IMAGES[@]}"; do
  tag="${img_def%%:*}"
  dockerfile="${img_def##*:}"
  local_image="moleculeforge/$tag:latest"
  printf 'Building %s from %s...\n' "$local_image" "$dockerfile"
  docker build -f "$dockerfile" -t "$local_image" .

  if [[ -n "$publish_registry" ]]; then
    published_image="$publish_registry/$tag:$publish_tag"
    docker tag "$local_image" "$published_image"
    docker push "$published_image"
  fi
done

printf '%s\n' "All images built successfully."
