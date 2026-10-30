#!/bin/bash
# Runs EAGLET-2 L1C products through the L1D processor's Docker image the way the PDGS runs it: as user
# 1100:600, with only the working directory mounted, the CPUs of the Task Table, and no network.
#
#     bash run_l1d.sh <L1C zip> [<L1C zip> ...]
#
# Run it from the folder that holds task_table.xml and the image archive IPF-IME-MSI-L1D-<version>.tar.gz
# (and its .sha256). The image is loaded from the archive once, after its SHA-256 is checked. Each product
# gets its own working directory in runs/, as the PDGS prepares one: a copy of the L1C and a Job Order
# written from the Task Table. The log goes next to it (runs/WD_....log) and the L1D zip is written in it.
# Without AUX, the Job Order has no colour AUX file and the processor uses its own. With AUX=<AUX .xml>, the
# Job Order gives that file, packed as the PDGS hands AUX files over (<ID>.zip holding <ID>/<ID>.xml and a
# header <ID>/<ID>.hdr, <ID> from the file's own File_Name). TAG=<label> goes into the folder names:
#
#     sudo env AUX=aux_test/HYBRID.xml TAG=HYBRID bash run_l1d.sh <L1C zip> ...
# IMAGE=<image> runs another image than ipf-ime-msi-l1d:<version of the Task Table>, a test image for example.
set -uo pipefail

TT=task_table.xml
[ -f "$TT" ] || { echo "No $TT in $(pwd)" >&2; exit 2; }
[ "$#" -ge 1 ] || { echo "usage: bash run_l1d.sh <L1C zip> [<L1C zip> ...]" >&2; exit 2; }
PROCESSOR=$(sed -n 's:^ *<Processor_Name>\(.*\)</Processor_Name>.*:\1:p' "$TT" | head -n 1)
VERSION=$(sed -n 's:^ *<Version>\([0-9][0-9]\.[0-9][0-9]\)</Version>.*:\1:p' "$TT" | head -n 1)
TASK=$(sed -n '/<Task>/,/<\/Task>/s:^ *<Name>\(.*\)</Name>.*:\1:p' "$TT" | head -n 1)
TASK_VERSION=$(sed -n '/<Task>/,/<\/Task>/s:^ *<Version>\(.*\)</Version>.*:\1:p' "$TT" | head -n 1)
CPUS=$(sed -n 's:^ *<Number_of_CPUs>\(.*\)</Number_of_CPUs>.*:\1:p' "$TT" | head -n 1)
IMAGE="${IMAGE:-ipf-ime-msi-l1d:$VERSION}"
ARCHIVE="IPF-IME-MSI-L1D-$VERSION.tar.gz"
REG_EXP='^IME[0-9]{2}_1DSP__RGB3.*\.zip$'
echo "Task Table: $PROCESSOR $VERSION, task $TASK $TASK_VERSION, $CPUS CPUs"
AUX=${AUX:-}
TAG=${TAG:-}
AUX_ID=""
if [ -n "$AUX" ]; then
    [ -f "$AUX" ] || { echo "No such AUX file: $AUX" >&2; exit 2; }
    AUX=$(readlink -f "$AUX")
    AUX_ID=$(sed -n 's:.*<File_Name>\(IME___AXE1D_CLUT_[^<]*\)</File_Name>.*:\1:p' "$AUX" | head -n 1)
    [ -n "$AUX_ID" ] || { echo "No IME___AXE1D_CLUT File_Name in $AUX" >&2; exit 2; }
    echo "Colour AUX file for the Job Order: $AUX_ID, method $(sed -n 's:.*<Method[^>]*>\([A-Z_]*\)</Method>.*:\1:p' "$AUX" | head -n 1)"
fi

# --- the image ------------------------------------------------------------------------------------------
if ! docker image inspect "$IMAGE" >/dev/null 2>&1; then
    [ -f "$ARCHIVE" ] || { echo "No image $IMAGE in Docker and no $ARCHIVE here" >&2; exit 2; }
    echo "Checking $ARCHIVE"
    sha256sum -c "$ARCHIVE.sha256" || { echo "The SHA-256 of $ARCHIVE does not match: not loading it" >&2; exit 2; }
    echo "Loading $ARCHIVE into Docker (a few minutes)"
    docker load -i "$ARCHIVE" || exit 2
fi
docker image inspect "$IMAGE" --format "Image $IMAGE: {{.Id}}" || exit 2
echo

mkdir -p runs
FAILED=0
for SRC in "$@"; do
    L1C=$(readlink -f "$SRC")
    [ -f "$L1C" ] || { echo "No such file: $SRC" >&2; FAILED=$((FAILED + 1)); continue; }
    NAME=$(basename "$L1C")
    WD="$(pwd)/runs/WD_${NAME:0:5}${TAG:+_$TAG}_$(date -u +%Y%m%dT%H%M%S)"
    JOB_ORDER="$WD/JobOrder.0001.xml"
    LOG="$WD.log"
    mkdir -p "$WD"
    cp "$L1C" "$WD/$NAME"              # a copy, not a link: only the working directory is mounted

    START=""; STOP=""
    if [[ $NAME =~ _([0-9]{8})T([0-9]{6})_([0-9]{8})T([0-9]{6})_ ]]; then
        START="${BASH_REMATCH[1]}_${BASH_REMATCH[2]}000000"
        STOP="${BASH_REMATCH[3]}_${BASH_REMATCH[4]}000000"
    fi
    BASELINE=A00
    if [[ $NAME =~ _([A-Z0-9]{3})\.zip$ ]]; then
        BASELINE=${BASH_REMATCH[1]}    # the input product's processing baseline, for the L1D name
    fi

    N_INPUTS=1
    AUX_INPUT=""
    if [ -n "$AUX_ID" ]; then           # the AUX file packed as the PDGS hands it over
        python3 - "$AUX" "$WD/$AUX_ID.zip" "$AUX_ID" <<'PYEOF' || { echo "Could not pack $AUX" >&2; exit 2; }
import sys, zipfile
src, out, aux_id = sys.argv[1:]
header = (f'<?xml version="1.0" encoding="UTF-8"?>\n<Earth_Explorer_Header>\n  <Fixed_Header>\n'
          f'    <File_Name>{aux_id}</File_Name>\n    <File_Type>AXE1D_CLUT</File_Type>\n  </Fixed_Header>\n'
          f'</Earth_Explorer_Header>\n')
with zipfile.ZipFile(out, "w", zipfile.ZIP_DEFLATED) as zf:
    zf.writestr(zipfile.ZipInfo(f"{aux_id}/"), b"")
    zf.write(src, f"{aux_id}/{aux_id}.xml")
    zf.writestr(f"{aux_id}/{aux_id}.hdr", header)
PYEOF
        N_INPUTS=2
        AUX_INPUT="
        <Input>
          <File_Type>AXE1D_CLUT</File_Type>
          <File_Name_Type>Physical</File_Name_Type>
          <List_of_File_Names count=\"1\">
            <File_Name>$WD/$AUX_ID.zip</File_Name>
          </List_of_File_Names>
          <List_of_Time_Intervals count=\"0\"/>
        </Input>"
    fi

    cat > "$JOB_ORDER" <<EOF
<?xml version="1.0" encoding="UTF-8"?>
<Ipf_Job_Order>
  <Ipf_Conf>
    <Processor_Name>$PROCESSOR</Processor_Name>
    <Version>$VERSION</Version>
    <Logging_Level>INFO</Logging_Level>
    <Processing_Station>IRI</Processing_Station>
    <Sensing_Time>
      <Start>$START</Start>
      <Stop>$STOP</Stop>
    </Sensing_Time>
    <List_of_Dyn_Processing_Parameters count="6">
      <Dyn_Processing_Parameter><Name>Class_ID</Name><Value>TEST</Value></Dyn_Processing_Parameter>
      <Dyn_Processing_Parameter><Name>Timeliness_Key</Name><Value></Value></Dyn_Processing_Parameter>
      <Dyn_Processing_Parameter><Name>Environment</Name><Value>O</Value></Dyn_Processing_Parameter>
      <Dyn_Processing_Parameter><Name>Processing_Baseline_1DSP__RGB3</Name><Value>$BASELINE</Value></Dyn_Processing_Parameter>
      <Dyn_Processing_Parameter><Name>Acquisition_ID</Name><Value></Value></Dyn_Processing_Parameter>
      <Dyn_Processing_Parameter><Name>Acquisition_Type</Name><Value>NOM</Value></Dyn_Processing_Parameter>
    </List_of_Dyn_Processing_Parameters>
  </Ipf_Conf>
  <List_of_Ipf_Procs count="1">
    <Ipf_Proc>
      <Task_Name>$TASK</Task_Name>
      <Task_Version>$TASK_VERSION</Task_Version>
      <Breakpoint>
        <Enable>OFF</Enable>
        <List_of_Brk_Files count="0"/>
      </Breakpoint>
      <List_of_Config_Files count="0"/>
      <List_of_Inputs count="$N_INPUTS">
        <Input>
          <File_Type>1CSP__RGB3</File_Type>
          <File_Name_Type>Physical</File_Name_Type>
          <List_of_File_Names count="1">
            <File_Name>$WD/$NAME</File_Name>
          </List_of_File_Names>
          <List_of_Time_Intervals count="0"/>
        </Input>$AUX_INPUT
      </List_of_Inputs>
      <List_of_Outputs count="1">
        <Output>
          <Destination>External</Destination>
          <Mandatory>Yes</Mandatory>
          <File_Type>1DSP__RGB3</File_Type>
          <File_Name_Type>Physical</File_Name_Type>
          <File_Name>$WD</File_Name>
          <Reg_Exp>$REG_EXP</Reg_Exp>
        </Output>
      </List_of_Outputs>
    </Ipf_Proc>
  </List_of_Ipf_Procs>
</Ipf_Job_Order>
EOF

    # the working directory belongs to the PDGS user, as in operations
    if [ "$(id -u)" = 0 ]; then
        chown -R 1100:600 "$WD"
    else
        chmod -R a+rwX "$WD"
    fi

    echo "=== $NAME"
    echo "    working directory $WD"
    T0=$(date +%s)
    docker run --rm --name "$TASK-run-$$" \
        --ulimit nofile=1024:1024 \
        --init \
        --cpus "$CPUS" \
        --network none \
        --read-only \
        -v "$WD:$WD:rw" \
        -e ENV_WORKING_DIR="$WD" \
        -w "$WD" \
        --user 1100:600 \
        "$IMAGE" "$TASK" "$JOB_ORDER" > "$LOG" 2>&1
    CODE=$?
    echo "    exit code $CODE after $(( $(date +%s) - T0 )) s, log $LOG"
    OUT=$(ls "$WD" | grep -E "$REG_EXP" | head -n 1)
    if [ "$CODE" -eq 0 ] && [ -n "$OUT" ]; then
        echo "    L1D: $WD/$OUT ($(du -h "$WD/$OUT" | cut -f1))"
    else
        FAILED=$((FAILED + 1))
        echo "    FAILED: no L1D. The errors in the log:"
        grep -E "\[E\]|Traceback|Error" "$LOG" | tail -n 15 | sed 's/^/      /'
    fi
    grep -E "Cloud stats|Colour correction: |Processing times|\[Stop\]" "$LOG" | sed 's/^.* : //; s/^/    /'
    echo
done

if [ "$FAILED" -eq 0 ]; then
    echo "ALL $# PRODUCTS DONE"
    exit 0
fi
echo "$FAILED OF $# PRODUCTS FAILED"
exit 1
