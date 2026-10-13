#!/bin/bash
# Builds the Docker (OCI) image of the processor, as the ICD asks (PDGS-ID-XPR-IRIDE-0209 v2.4,
# Appendices D and F):
#     docker/build_image.sh           build the image <IMAGE_NAME>:<version>
#     docker/build_image.sh --save    build it and save it for delivery in dist/, with its SHA-256
# <version> is the processor version in task_table.xml (e.g. 01.00): the same in the image tag, the
# archive name and the Task Table. Needs Docker, internet (base image, Python packages) and the
# model weights in hf_cache/omnicloudmask (python download_models.py). The image never goes online.
set -euo pipefail

REPO=$(cd "$(dirname "$0")/.." && pwd)
cd "$REPO"

# TODO: image name to agree with PDGS (naming convention in [GSNCD], ICD Appendix E, item 1)
IMAGE_NAME=${IMAGE_NAME:-ipf-ime-msi-l1d}
VERSION=$(sed -n 's:^ *<Version>\([0-9][0-9]\.[0-9][0-9]\)</Version>.*:\1:p' task_table.xml | head -n 1)
if [ -z "$VERSION" ]; then
    echo "No processor <Version> found in task_table.xml" >&2
    exit 1
fi
IMAGE="$IMAGE_NAME:$VERSION"

if ! ls hf_cache/omnicloudmask/*.safetensors >/dev/null 2>&1; then
    echo "No model weights in hf_cache/omnicloudmask: run 'python download_models.py' first" >&2
    exit 1
fi

echo "Building $IMAGE"
docker build --pull --build-arg PROCESSOR_VERSION="$VERSION" -t "$IMAGE" .

mkdir -p dist
FREEZE="dist/${IMAGE_NAME}_${VERSION}.pip-freeze.txt"
docker run --rm --network none "$IMAGE" python -m pip freeze > "$FREEZE"
echo "Python packages in the image: $FREEZE"
docker image inspect "$IMAGE" --format "Image $IMAGE: {{.Id}}, {{.Size}} bytes"

if [ "${1:-}" = "--save" ]; then
    ARCHIVE="dist/$(echo "$IMAGE_NAME" | tr '[:lower:]' '[:upper:]')-${VERSION}.tar.gz"
    echo "Saving $IMAGE to $ARCHIVE"
    docker save "$IMAGE" | gzip > "$ARCHIVE"
    (cd dist && sha256sum "$(basename "$ARCHIVE")" > "$(basename "$ARCHIVE").sha256")
    ls -l "$ARCHIVE" "$ARCHIVE.sha256"
fi
