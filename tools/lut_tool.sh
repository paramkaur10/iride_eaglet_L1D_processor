#!/bin/bash
# lut_tool.sh: runs one of the LUT tools (build_lut_v4.py, build_lut.py, lut_panels.py, cloud_panels.py) in the
# image eaglet-lut-tools (tools/docker: the processor's image, with its Python, packages, model weights, src and
# colour AUX file, plus opencv, requests and matplotlib). As the processor's user 1100:600, on the host's network
# (Docker's own network could not look up names on our server), for the Copernicus Data Space.
# Inside: LUT (default ~/lut) is /lut, results in /lut/work (owned by 1100:600, so put work folders in it); read
# only, TOOLS (default LUT/tools) is the processor's tools/ folder, DATA (default ~/l1d_ohb/data, the L1C zips) is
# /data and RUNS (default ~/l1d_ohb/runs, the processor's working folders made by run_l1d.sh) is /runs.
#     bash ~/lut/lut_tool.sh build_lut_v4.py --work /lut/work/v5 --products /data --since 20261017
# From a clone of the repository, with its tools/:
#     TOOLS=$PWD/tools bash tools/lut_tool.sh build_lut_v4.py --work /lut/work/v5 --products /data --since 20261017
# The CDSE client id and secret are asked for when a download is needed, or taken from CDSE_CLIENT_ID and
# CDSE_CLIENT_SECRET when they are set in this shell; they are never written to a file.
set -euo pipefail
LUT="${LUT:-$HOME/lut}"
TOOLS="${TOOLS:-$LUT/tools}"
DATA="${DATA:-$HOME/l1d_ohb/data}"
RUNS="${RUNS:-$HOME/l1d_ohb/runs}"
APP=/opt/ipf-ime-msi-l1d/iride_eaglet_processor_v2-main
TOOL=${1:?usage: bash lut_tool.sh <tool.py> [its options]}
shift
[ -f "$TOOLS/$TOOL" ] || { echo "No $TOOLS/$TOOL" >&2; exit 2; }
sudo mkdir -p "$LUT/work/.home" "$RUNS"
sudo chown -R 1100:600 "$LUT/work"
TTY=()
if [ -t 0 ] && [ -t 1 ]; then TTY=(-it); fi
exec sudo --preserve-env=CDSE_CLIENT_ID,CDSE_CLIENT_SECRET docker run --rm --network host "${TTY[@]}" \
    --user 1100:600 -e HOME=/lut/work/.home -e CDSE_CLIENT_ID -e CDSE_CLIENT_SECRET \
    -v "$LUT:/lut" -v "$TOOLS:$APP/tools:ro" -v "$DATA:/data:ro" -v "$RUNS:/runs:ro" -w /lut/work \
    eaglet-lut-tools python "$APP/tools/$TOOL" "$@"
