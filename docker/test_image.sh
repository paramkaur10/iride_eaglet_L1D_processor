#!/bin/bash
# Validation of the processor's Docker image, run the way PDGS runs it (ICD PDGS-ID-XPR-IRIDE-0209
# v2.4, Appendices D and E): as user 1100:600, with only the working directory mounted and the CPUs
# of the Task Table. Stricter than PDGS: no network and a read-only container file system, so any
# download, or any file written outside the working directory, makes the run fail.
#
#     docker/test_image.sh <L1C product zip> [<reference L1D zip>]
#
# Makes a new working directory under $TEST_ROOT (default /var/tmp/eaglet2_l1d_test; not under
# /root, which user 1100 cannot enter, in the container too), copies the product zip into it, writes
# the Job Order as PDGS would, runs the image and checks the result. With a reference L1D zip, made
# without Docker from the same product with the same colour method, the L1D image must have the same
# pixels (compared in the image, with its Python and rasterio).
# The Job Order also gives the colour AUX file (AXE1D_CLUT) as PDGS hands AUX files over: <ID>.zip holding
# <ID>/<ID>.xml and a header <ID>/<ID>.hdr (PDGS run of 20 Oct 2026). By default it is the newest AUX file in
# aux/; AUX=<.xml file> packs another one the same way, AUX=<.zip file> gives that zip as it is, and AUX=none
# leaves the AUX file out of the Job Order, so the processor uses its own. The Job Order's processing baseline
# for the L1D name is the input product's (BASELINE=<A01 or other> sets another).
# Unattended: prints PASS or FAIL for each check and exits with 0 only when all of them pass.
# DOCKER=podman runs it with Podman.
set -uo pipefail

if [ "$#" -lt 1 ] || [ "$#" -gt 2 ]; then
    echo "usage: docker/test_image.sh <L1C product zip> [<reference L1D zip>]" >&2
    exit 2
fi
L1C=$(readlink -f "$1")
REF=${2:+$(readlink -f "$2")}
for f in "$L1C" ${REF:+"$REF"}; do
    [ -f "$f" ] || { echo "No such file: $f" >&2; exit 2; }
done

REPO=$(cd "$(dirname "$0")/.." && pwd)
TT="$REPO/task_table.xml"
DOCKER=${DOCKER:-docker}
IMAGE_NAME=${IMAGE_NAME:-ipf-ime-msi-l1d}
VERSION=$(sed -n 's:^ *<Version>\([0-9][0-9]\.[0-9][0-9]\)</Version>.*:\1:p' "$TT" | head -n 1)
PROCESSOR=$(sed -n 's:^ *<Processor_Name>\(.*\)</Processor_Name>.*:\1:p' "$TT" | head -n 1)
TASK=$(sed -n '/<Task>/,/<\/Task>/s:^ *<Name>\(.*\)</Name>.*:\1:p' "$TT" | head -n 1)
TASK_VERSION=$(sed -n '/<Task>/,/<\/Task>/s:^ *<Version>\(.*\)</Version>.*:\1:p' "$TT" | head -n 1)
CPUS=$(sed -n 's:^ *<Number_of_CPUs>\(.*\)</Number_of_CPUs>.*:\1:p' "$TT" | head -n 1)
REG_EXP='^IME[0-9]{2}_1DSP__RGB3.*\.zip$'
IMAGE="$IMAGE_NAME:$VERSION"
if ! "$DOCKER" image inspect "$IMAGE" >/dev/null 2>&1; then
    echo "No image $IMAGE: build it first with docker/build_image.sh" >&2
    exit 2
fi

# --- the colour AUX file for the Job Order ------------------------------------------------------
OWN_AUX=$(ls "$REPO"/aux/IME___AXE1D_CLUT_*.xml 2>/dev/null | sort | tail -n 1)
[ -n "$OWN_AUX" ] || { echo "No colour AUX file in $REPO/aux" >&2; exit 2; }
AUX=${AUX:-$OWN_AUX}
if [ "$AUX" = none ]; then
    AUX_SRC=""
    AUX_ID=$(basename "$OWN_AUX" .xml)             # the processor's own, the newest in aux/
else
    AUX_SRC=$(readlink -f "$AUX")
    [ -f "$AUX_SRC" ] || { echo "No such file: $AUX" >&2; exit 2; }
    case "$AUX_SRC" in
        *.xml) AUX_ID=$(basename "$AUX_SRC" .xml) ;;
        *.zip) AUX_ID=$(basename "$AUX_SRC" .zip) ;;
        *) echo "AUX must be an .xml or .zip file, or none: $AUX" >&2; exit 2 ;;
    esac
fi

# --- the working directory, as PDGS prepares it ------------------------------------------------
NAME=$(basename "$L1C")
TEST_ROOT=${TEST_ROOT:-/var/tmp/eaglet2_l1d_test}
WD="$TEST_ROOT/WD_$(date -u +%Y%m%dT%H%M%S)"
JOB_ORDER="$WD/JobOrder.0001.xml"
LOG="$WD.log"
mkdir -p "$WD"
cp "$L1C" "$WD/$NAME"          # a copy, not a link: only the working directory is mounted

START=""; STOP=""
if [[ $NAME =~ _([0-9]{8})T([0-9]{6})_([0-9]{8})T([0-9]{6})_ ]]; then
    START="${BASH_REMATCH[1]}_${BASH_REMATCH[2]}000000"
    STOP="${BASH_REMATCH[3]}_${BASH_REMATCH[4]}000000"
fi
if [ -z "${BASELINE:-}" ] && [[ $NAME =~ _([A-Z0-9]{3})\.zip$ ]]; then
    BASELINE=${BASH_REMATCH[1]}                    # the input product's processing baseline
fi
BASELINE=${BASELINE:-A00}

# the colour AUX file, packed as PDGS packs it, unless it is a zip already
AUX_FILE=""
AUX_INPUT=""
if [ -n "$AUX_SRC" ]; then
    AUX_FILE="$AUX_ID.zip"
    if [[ $AUX_SRC == *.zip ]]; then
        cp "$AUX_SRC" "$WD/$AUX_FILE"
    else
        python3 - "$AUX_SRC" "$WD/$AUX_FILE" "$AUX_ID" <<'PYEOF' || { echo "Could not pack $AUX_SRC" >&2; exit 2; }
import sys, zipfile
src, out, aux_id = sys.argv[1:]
header = (f'<?xml version="1.0" encoding="UTF-8"?>\n<Earth_Explorer_Header>\n  <Fixed_Header>\n'
          f'    <File_Name>{aux_id}</File_Name>\n    <File_Type>AXE1D_CLUT</File_Type>\n  </Fixed_Header>\n'
          f'</Earth_Explorer_Header>\n')
with zipfile.ZipFile(out, "w", zipfile.ZIP_DEFLATED) as zf:
    zf.writestr(zipfile.ZipInfo(f"{aux_id}/"), b"")                 # the folder, then the AUX file and its header
    zf.write(src, f"{aux_id}/{aux_id}.xml")
    zf.writestr(f"{aux_id}/{aux_id}.hdr", header)
PYEOF
    fi
    AUX_INPUT="
        <Input>
          <File_Type>AXE1D_CLUT</File_Type>
          <File_Name_Type>Physical</File_Name_Type>
          <List_of_File_Names count=\"1\">
            <File_Name>$WD/$AUX_FILE</File_Name>
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
      <List_of_Inputs count="$(( ${#AUX_FILE} > 0 ? 2 : 1 ))">
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

# --- the run: the docker run of the ICD (Appendix E, item 13), plus --network none --read-only ---
echo "Image $IMAGE, task $TASK $TASK_VERSION, $CPUS CPUs"
echo "Job Order: processing baseline $BASELINE, colour AUX file ${AUX_FILE:-none (the processor uses its own, $AUX_ID)}"
echo "Working directory $WD"
echo "Log $LOG"
T0=$(date +%s)
"$DOCKER" run --rm --name "$TASK-test-$$" \
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
SECONDS_RUN=$(( $(date +%s) - T0 ))
echo "Exit code $CODE after $SECONDS_RUN s"
echo

# --- checks ------------------------------------------------------------------------------------
FAILED=0
check() {       # check <description> <command...>: PASS when the command succeeds
    local description=$1
    shift
    if "$@" >/dev/null 2>&1; then
        echo "PASS  $description"
    else
        echo "FAIL  $description"
        FAILED=$((FAILED + 1))
    fi
}

check "exit code 0 (got $CODE)" [ "$CODE" -eq 0 ]
check "log ends with [Stop] exit_code=0" grep -q "\[Stop\] exit_code=0" "$LOG"
check "the processor uses $CPUS CPU threads (Number_of_CPUs)" \
    grep -q "Inference device: cpu (the processor runs on CPU only), $CPUS threads" "$LOG"
if [ -n "$AUX_FILE" ]; then
    check "colour AUX file from the Job Order, read from the zip: $AUX_FILE" \
        grep -q "Colour correction: method .*, AUX $AUX_ID (version .*), read from $AUX_FILE (" "$LOG"
else
    check "colour AUX file of the processor: $AUX_ID.xml" \
        grep -q "Colour correction: method .*, AUX $AUX_ID (version .*), read from $AUX_ID.xml\$" "$LOG"
fi

OUTPUTS=()
LEFT=()
for path in "$WD"/* "$WD"/.[!.]*; do
    [ -e "$path" ] || continue
    name=$(basename "$path")
    if [[ $name =~ $REG_EXP ]]; then
        OUTPUTS+=("$name")
    elif [ "$name" != "$NAME" ] && [ "$name" != "$(basename "$JOB_ORDER")" ] && [ "$name" != "$AUX_FILE" ]; then
        LEFT+=("$name")
    fi
done
check "one output in the working directory matching $REG_EXP (found ${#OUTPUTS[@]})" [ "${#OUTPUTS[@]}" -eq 1 ]
check "nothing else left in the working directory${LEFT[*]:+ (found: ${LEFT[*]})}" [ "${#LEFT[@]}" -eq 0 ]
OUT=${OUTPUTS[0]:-}

if [ -n "$OUT" ]; then
    check "log says OUTPUT_READY: $OUT" grep -q "OUTPUT_READY: $OUT\$" "$LOG"
    if [ "$(id -u)" = 0 ]; then
        OWNER=$(stat -c %u:%g "$WD/$OUT")
        check "output owned by 1100:600, the user that ran it (got $OWNER)" [ "$OWNER" = "1100:600" ]
    fi

    ID=${OUT%.zip}
    check "zip holds $ID/ with its XML, STAC and IMG_DATA/$ID.tif" \
        unzip -l "$WD/$OUT" "$ID/$ID.xml" "$ID/$ID.json" "$ID/IMG_DATA/$ID.tif"
    XML=$(unzip -p "$WD/$OUT" "$ID/$ID.xml" 2>/dev/null)
    STAC=$(unzip -p "$WD/$OUT" "$ID/$ID.json" 2>/dev/null)
    GEN="(none in the name)"
    if [[ $ID =~ ^IME[0-9]{2}_1DSP__RGB3_[0-9]{8}T[0-9]{6}_[0-9]{8}T[0-9]{6}_([0-9]{4})([0-9]{2})([0-9]{2})T([0-9]{2})([0-9]{2})([0-9]{2})_ ]]; then
        GEN="${BASH_REMATCH[1]}-${BASH_REMATCH[2]}-${BASH_REMATCH[3]}T${BASH_REMATCH[4]}:${BASH_REMATCH[5]}:${BASH_REMATCH[6]}"
    fi
    check "XML PRODUCT_ID is the product name" grep -q "<PRODUCT_ID>$ID</PRODUCT_ID>" <<< "$XML"
    check "XML PROCESSING_LEVEL L1D" grep -q "<PROCESSING_LEVEL>L1D</PROCESSING_LEVEL>" <<< "$XML"
    check "XML GENERATION_TIME is the generation time in the name ($GEN)" grep -q "<GENERATION_TIME>$GEN" <<< "$XML"
    check "STAC id is the product name" grep -q "\"id\": *\"$ID\"" <<< "$STAC"
    check "STAC processing:level L1D" grep -q "\"processing:level\": *\"L1D\"" <<< "$STAC"
    check "XML AUXILIARY_FILE is the colour AUX file, without .xml or .zip" \
        grep -q "<AUXILIARY_FILE>$AUX_ID</AUXILIARY_FILE>" <<< "$XML"
    check "STAC names the colour AUX file, without .xml or .zip" grep -q "colour AUX file $AUX_ID\. " <<< "$STAC"

    image_sha() {       # the SHA-256 of the L1D image in the zip $1
        local id
        id=$(basename "$1" .zip)
        unzip -p "$1" "$id/IMG_DATA/$id.tif" 2>/dev/null | sha256sum | cut -c1-64
    }
    IMAGE_SHA=$(image_sha "$WD/$OUT")
    XML_SHA=$(tr -d '\n' <<< "$XML" \
        | sed -n 's:.*<Checksum[^>]*>[^<]*<Image>[[:space:]]*\([0-9a-fA-F]*\)[[:space:]]*</Image>.*:\1:p')
    check "XML image SHA-256 is the one of the L1D image (${XML_SHA:0:12})" [ "$XML_SHA" = "$IMAGE_SHA" ]

    if [ -n "$REF" ]; then
        cloud_cover() { # eo:cloud_cover in the STAC in the zip $1
            local id
            id=$(basename "$1" .zip)
            unzip -p "$1" "$id/$id.json" 2>/dev/null | grep -o '"eo:cloud_cover": *[0-9.eE+-]*' | head -n 1
        }
        same_pixels() { # the L1D images in the zips $1 and $2 have the same pixels, compared in the image
            local a b
            a=$(basename "$1" .zip)
            b=$(basename "$2" .zip)
            "$DOCKER" run --rm --network none --user 1100:600 \
                -v "$(dirname "$1"):/zips_a:ro" -v "$(dirname "$2"):/zips_b:ro" "$IMAGE" python -c '
import sys
import numpy as np
import rasterio
with rasterio.open(sys.argv[1]) as a, rasterio.open(sys.argv[2]) as b:
    same = a.count == b.count and a.shape == b.shape and all(
        np.array_equal(a.read(i), b.read(i)) for i in range(1, a.count + 1))
sys.exit(0 if same else 1)' "/vsizip//zips_a/$a.zip/$a/IMG_DATA/$a.tif" "/vsizip//zips_b/$b.zip/$b/IMG_DATA/$b.tif"
        }
        REF_SHA=$(image_sha "$REF")
        SAME=no
        if same_pixels "$WD/$OUT" "$REF" >/dev/null 2>&1; then
            SAME=yes
        fi
        check "L1D image has the same pixels as the one in $(basename "$REF")" [ "$SAME" = yes ]
        if [ "$IMAGE_SHA" = "$REF_SHA" ]; then
            echo "      the two image files are identical too (SHA-256 ${IMAGE_SHA:0:12})"
        elif [ "$SAME" = yes ]; then
            echo "      the two image files differ in their bytes only (file layout, e.g. another GDAL version)"
        fi
        echo "      cloud cover: this run $(cloud_cover "$WD/$OUT"), reference $(cloud_cover "$REF")"
    fi
fi

echo
grep -E "Colour correction: method|Saturated in the L1C|Quicklook stretch|Processing times" "$LOG" | sed 's/^.* : //'
echo
if [ "$FAILED" -eq 0 ]; then
    echo "ALL CHECKS PASSED"
    exit 0
fi
echo "$FAILED CHECK(S) FAILED: see $LOG"
exit 1
