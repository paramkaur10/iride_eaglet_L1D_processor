#!/bin/bash
# Builds the Docker (OCI) image of the processor the way PDGS builds it (docker/Dockerfile-e2l1d-trixie,
# in the format of the PDGS build of the previous version), and saves it for delivery (ICD Appendices
# D and F):
#     docker/build_image.sh           build the image <IMAGE_NAME>:<version>
#     docker/build_image.sh --save    build it and save it in dist/, with its SHA-256
# The build context is made in dist/build/ as PDGS has it: the repository as a zip
# (iride_eaglet_processor_v2-main.zip, like GitHub's "Download ZIP" of main; here made from this
# checkout, with the changes not yet committed), the package list and the two model weights.
# <version> is the processor version in task_table.xml (e.g. 01.00). Needs Docker, internet (base
# image, Python packages) and the weights in hf_cache/omnicloudmask (python download_models.py).
set -euo pipefail

REPO=$(cd "$(dirname "$0")/.." && pwd)
cd "$REPO"

IMAGE_NAME=${IMAGE_NAME:-ipf-ime-msi-l1d}
VERSION=$(sed -n 's:^ *<Version>\([0-9][0-9]\.[0-9][0-9]\)</Version>.*:\1:p' task_table.xml | head -n 1)
if [ -z "$VERSION" ]; then
    echo "No processor <Version> found in task_table.xml" >&2
    exit 1
fi
IMAGE="$IMAGE_NAME:$VERSION"

WEIGHTS=(hf_cache/omnicloudmask/*.safetensors)
if [ ! -f "${WEIGHTS[0]}" ]; then
    echo "No model weights in hf_cache/omnicloudmask: run 'python download_models.py' first" >&2
    exit 1
fi

# the build context, as PDGS has it
CONTEXT=dist/build
rm -rf "$CONTEXT"
mkdir -p "$CONTEXT"
SNAPSHOT=$(git stash create)        # the checkout with its uncommitted changes, without touching it
git archive --format=zip --prefix=iride_eaglet_processor_v2-main/ \
    -o "$CONTEXT/iride_eaglet_processor_v2-main.zip" "${SNAPSHOT:-HEAD}"
cp docker/Dockerfile-e2l1d-trixie docker/Eaglet-2-L1D-requirements-trixie.txt "${WEIGHTS[@]}" "$CONTEXT/"
echo "Build context $CONTEXT:"
ls -l "$CONTEXT"

echo "Building $IMAGE"
docker build --pull -t "$IMAGE" -f "$CONTEXT/Dockerfile-e2l1d-trixie" "$CONTEXT"

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
