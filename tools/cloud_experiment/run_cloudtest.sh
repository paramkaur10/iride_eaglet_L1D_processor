#!/bin/bash
# run_cloudtest.sh: the cloud experiment from start to end. L1Ds of the scenes below with the experiment image
# (ipf-ime-msi-l1d:01.01-cloudtest) and the v4 AUX file, tag V4_CLOUD; then the panels LUT_V4 against V4_CLOUD,
# copied to ~/l1d_ohb/downloads/cloud_panels for download from Jupyter.
#     nohup bash ~/cloudtest/run_cloudtest.sh > ~/l1d_ohb/cloudtest.log 2>&1 &
set -uo pipefail
SCENES="5 12 14 18 20"
IMAGE=ipf-ime-msi-l1d:01.01-cloudtest
cd ~/l1d_ohb || exit 2
sudo docker image inspect "$IMAGE" >/dev/null 2>&1 || { echo "No image $IMAGE: build it first" >&2; exit 2; }
FREE=$(df --output=avail -BG ~ | tail -1 | tr -dc 0-9)
[ "$FREE" -ge 15 ] || { echo "Only $FREE GB free on the disk: not starting (each run copies its L1C)" >&2; exit 2; }
V4_AUX=$(ls ~/lut/work_v4/aux/IME___AXE1D_CLUT_*.xml) || exit 2
ZIPS=$(python3 - $SCENES <<'PYEND'
import json, os, sys
want = {f"#{n}" for n in sys.argv[1:]}
inv = json.load(open(os.path.expanduser("~/lut/work_v4/inventory.json")))
print(" ".join(f"data/{p['stem']}.zip" for p in inv if p["label"].split()[0] in want))
PYEND
)
[ -n "$ZIPS" ] || { echo "None of the scenes $SCENES in ~/lut/work_v4/inventory.json" >&2; exit 2; }
echo "AUX file: $V4_AUX"
echo "L1C products ($(echo $ZIPS | wc -w)):"
printf '    %s\n' $ZIPS
echo
sudo env IMAGE="$IMAGE" AUX="$V4_AUX" TAG=V4_CLOUD bash run_l1d_test.sh $ZIPS
echo
echo "== the experiment, as the processor logged it (one line per scene):"
grep -h "EXPERIMENT\|clouds not kept neutral" runs/WD_*_V4_CLOUD_*.log
echo
echo "== panels: LUT_V4 against V4_CLOUD"
bash ~/lut/lut_tool.sh cloud_panels.py --work /lut/work_v4 --runs /runs --a LUT_V4 --b V4_CLOUD $SCENES
mkdir -p ~/l1d_ohb/downloads/cloud_panels
cp ~/lut/work_v4/cloud_panels/*.png ~/l1d_ohb/downloads/cloud_panels/ && ls ~/l1d_ohb/downloads/cloud_panels
echo "CLOUD TEST FINISHED"
