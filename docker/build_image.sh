#!/bin/bash
# Builds the Docker (OCI) image of the processor the way PDGS builds it (docker/Dockerfile-e2l1d-trixie,
# in the format of the PDGS build of the previous version), and saves it for delivery:
#     docker/build_image.sh           build the image <IMAGE_NAME>:<version>
#     docker/build_image.sh --save    build it and save it in dist/, with its SHA-256
# The build context is made in dist/build/ as PDGS has it: the repository as a zip
# (iride_eaglet_processor_v2-main.zip, like GitHub's "Download ZIP" of main; here made from this
# checkout, with the changes to tracked files not yet committed), the package list and the cloud model
# package (clarity_eaglet2_fp32/, the files its SHA256SUMS lists). <version> is the processor version in
# task_table.xml (e.g. 01.02). Needs Docker, internet (base image, Python packages) and the cloud model
# package unzipped in models/clarity_eaglet2_fp32 (models/README.md).
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

# the cloud model package, as delivered (checked here, again in the image build, and at every run)
MODEL=models/clarity_eaglet2_fp32
if [ ! -f "$MODEL/SHA256SUMS" ]; then
    echo "No cloud model package in $MODEL: unzip the delivered package into models/ first (models/README.md)" >&2
    exit 1
fi
if ! (cd "$MODEL" && sha256sum --quiet -c SHA256SUMS); then
    echo "$MODEL is not the delivered package: its files do not match its SHA256SUMS" >&2
    exit 1
fi
MODEL_FILES=$(sed 's/^[0-9a-fA-F]* [ *]//' "$MODEL/SHA256SUMS")

# the build context, as PDGS has it
CONTEXT=dist/build
rm -rf "$CONTEXT"
mkdir -p "$CONTEXT/clarity_eaglet2_fp32"
SNAPSHOT=$(git stash create)        # the checkout with its uncommitted changes, without touching it
git archive --format=zip --prefix=iride_eaglet_processor_v2-main/ \
    -o "$CONTEXT/iride_eaglet_processor_v2-main.zip" "${SNAPSHOT:-HEAD}"
cp docker/Dockerfile-e2l1d-trixie docker/Eaglet-2-L1D-requirements-trixie.txt "$CONTEXT/"
for f in $MODEL_FILES SHA256SUMS; do
    cp "$MODEL/$f" "$CONTEXT/clarity_eaglet2_fp32/"
done
echo "Build context $CONTEXT:"
ls -l "$CONTEXT" "$CONTEXT/clarity_eaglet2_fp32"
echo "Repository zip: $(unzip -Z1 "$CONTEXT/iride_eaglet_processor_v2-main.zip" | wc -l) entries" \
     "(docs/, tools/ and the posters are left out by .gitattributes)"

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
