#!/usr/bin/env python3
"""
process_l1c.py — Eaglet-2 L1C -> L1D processor: cloud masking and colour
correction. Task executable (repository iride_eaglet_processor_v2).

PDGS ICD compliance (PDGS-ID-XPR-IRIDE-0209 v2.4):
  * Accepts exactly one command-line argument: the absolute path to the
    Job Order XML file (per Section 3.1.2 / Appendix E). No other
    arguments, no interactive input.
  * Does NOT scan the working directory for input files. The input file
    path is read from the Job Order itself (Section 3.1.2.3), per the
    ICD's explicit instruction that a processor "must rely on the paths
    specified in the JobOrder, trusting that they exist as volumes
    mounted at container runtime."
  * Logs in the exact structured format required by Section 3.1.4:
    one-line, Syslog-style, unbuffered, on stdout (Debug/Info/Progress)
    and stderr (Warning/Error).
  * Returns an exit code per Section 3.1.5: 0 = OK, 128-255 = FAILURE.
  * Runs on CPU only: no GPU is used, even on a machine that has one.
  * Runs fully offline — OmniCloudMask's model weights are loaded from a
    local cache bundled alongside this processor (see hf_cache/ and
    download_models.py), never downloaded at runtime. Required for
    air-gapped deployment (IRIDE PDGS).
  * Generates a NEW L1D output product from the L1C input (naming: "1CSP"
    -> "1DSP", generation timestamp = this run's processing time). The
    original input zip is never modified or deleted. The L1D product keeps
    the L1C layout, renamed throughout to the new L1D product ID, with:
        IMG_DATA/<NEW_ID>.tif        the L1D image, in place of the L1C one:
                                     colour-corrected RGB (true colour), Cloud
                                     Optimized GeoTIFF, LZW (lossless), uint16
                                     = reflectance x 10000 + 1000, 0 = no data
        AUX_DATA/<NEW_ID>.CLM.tif    5-class cloud mask
        AUX_DATA/<NEW_ID>.COM.tif    cloud confidence map
        AUX_DATA/<NEW_ID>.CLM.json   class legend
        <NEW_ID>.json                updated: eo:cloud_cover, clm/com/clm_aux,
                                     the rgb asset and its renders describing the
                                     L1D image, the L1D identity (collection,
                                     description, processing:level L1D,
                                     processing:lineage = the input L1C product,
                                     processing:software = this processor and
                                     version), quicklook -> PNG and thumbnail ->
                                     JPEG, id, created/updated, asset hrefs,
                                     file checksums and sizes
        <NEW_ID>.xml                 the L1C XML as in Enrico's L1D template:
                                     PRODUCT_ID = <NEW_ID>, PROCESSING_LEVEL L1D,
                                     GENERATION_TIME = this run's generation time,
                                     AUXILIARY_FILE = only the colour AUX file,
                                     INPUT_PRODUCTS = the input L1C zip,
                                     Processor_Version = this processor's; and
                                     Rescaling_Factors (GAIN 0.0001, OFFSET -0.1)
                                     and the image SHA-256 set for the L1D image
        PREVIEW/<NEW_ID>.QLK.*       PNG and JPEG drawn from the L1D image (same
                                     names, formats and sizes as in the L1C);
                                     the PNG is RGBA, transparent where there is
                                     no data; the KML places the PNG on the map
                                     (the L1C one places the JPEG)
    packaged as <NEW_ID>.zip in the same working directory.
  * The colour correction is configured by an AUX file of type AXE1D_CLUT
    (IME___AXE1D_CLUT_....xml: the LUT, Enrico's class model, and <Method>,
    which selects LUT, CLASS_MODEL or HYBRID). The processor carries its own
    in aux/ and uses it by default (the newest, by the creation time in the
    name); an AXE1D_CLUT file given in the Job Order overrides it. Without a
    usable AUX file there is no L1D image: exit code 255.
  * Calibration (DN * gain + offset) comes from the product XML's
    Rescaling_Factors, for the cloud mask and the colour step alike.
  * Logs the time spent in each step at the end ("Processing times").

Usage (as invoked by the PDGS wrapper script, per Appendix E):
    python process_l1c.py <absolute_path_to_JobOrder.xml>
"""

import json
import os
import shutil
import sys
import time
import traceback
import zipfile
from datetime import datetime, timezone
from pathlib import Path

_PROCESSOR_ROOT = Path(__file__).resolve().parent

# --- Offline Hugging Face model cache -----------------------------------
# Must be set before importing anything that transitively imports
# huggingface_hub (i.e. before importing iride_eaglet_processor.inference).
os.environ.setdefault("HF_HOME", str(_PROCESSOR_ROOT / "hf_cache"))
os.environ["HF_HUB_OFFLINE"] = "1"

# --- CPU only -------------------------------------------------------------
# The processor runs strictly on CPU: no GPU is made visible to torch, even
# on a machine that has one. Must be set before torch is imported.
os.environ["CUDA_VISIBLE_DEVICES"] = ""

_SRC_DIR = _PROCESSOR_ROOT / "src"
sys.path.insert(0, str(_SRC_DIR))

import numpy as np

from job_order import parse_job_order, JobOrderError
from pdgs_logger import PdgsLogger
from product_naming import derive_output_product_id, rename_product_in_place, update_json_references

from iride_eaglet_processor.raster_reader import read_product, NODATA_VALUE
from iride_eaglet_processor.footprint import get_valid_footprint
from iride_eaglet_processor.resampling import (
    downsample_to_target_resolution,
    upsample_prob_stack,
)
from iride_eaglet_processor.inference import run_omnicloud_on_array
from iride_eaglet_processor.masks import (
    probs_to_four_class_map,
    probs_to_cloud_confidence,
    save_clm_output,
    save_com_output,
)
from iride_eaglet_processor.stats import compute_cloud_stats
from iride_eaglet_processor.metadata import save_clm_legend, enrich_sidecar_json, refresh_file_info
from iride_eaglet_processor.calibration import find_product_xml, set_l1d_image_in_xml
from iride_eaglet_processor.product_xml import set_l1d_header_in_xml
from iride_eaglet_processor.previews import make_l1d_previews, point_kml_at_renamed_files
from iride_eaglet_processor.colour_correction import (
    ColourCorrection, write_colour_product, L1D_GAIN, L1D_OFFSET,
)


# --- Must match task_table.xml exactly -----------------------------------
# TODO: confirm these registered names with PDGS integration.
# Named after the OUTPUT level (L1D), per PDGS's own convention — see the
# HAWK_L1B_PROC example in the ICD (Section 4.1.1), which is named after
# its output (L1B) despite consuming L0 input. No "CLM" in the names: the
# processor does the whole L1C to L1D step, cloud mask and colour
# correction (Enrico, 12 Oct 2026). PROCESSOR_VERSION is also written in
# the STAC of the product (processing:software).
PROCESSOR_NAME = "EAGLET2_L1D_PROC"
PROCESSOR_VERSION = "01.00"
TASK_NAME = "EAGLET2_L1D"
TASK_VERSION = "01.00"

# Job Order input types (must match task_table.xml)
PRODUCT_FILE_TYPE = "1CSP__RGB3"     # the L1C RGB3 product zip
CLUT_FILE_TYPE = "AXE1D_CLUT"        # colour-correction AUX file (LUT + class model + Method)
OWN_AUX_DIR = _PROCESSOR_ROOT / "aux"  # the processor's own AXE1D_CLUT file, used unless the Job Order gives one

INFERENCE_RESOLUTION = 5.0   # m — OmniCloudMask's expected input resolution

# Files already compressed are stored in the output zip as they are
# (deflating them again costs time and saves almost nothing)
STORED_IN_ZIP = (".tif", ".tiff", ".png", ".jpg", ".jpeg", ".jp2")

# Exit codes per ICD Section 3.1.5.3
EXIT_OK = 0
EXIT_FAILURE = 255      # 128-255 range: error, PDGS aborts


def is_product_root(p):
    return (p / "IMG_DATA").exists() and (p / f"{p.name}.json").exists()


def extract_zip(zip_path, extract_into, logger):
    product_name = zip_path.stem
    extract_target = extract_into / product_name

    logger.info(f"Extracting: {zip_path.name}")
    with zipfile.ZipFile(zip_path, "r") as zf:
        names = zf.namelist()
        if names and names[0].startswith(product_name + "/"):
            zf.extractall(extract_into)
        else:
            extract_target.mkdir(parents=True, exist_ok=True)
            zf.extractall(extract_target)

    if not is_product_root(extract_target):
        raise RuntimeError(
            f"Extracted archive does not contain a valid L1C product "
            f"(IMG_DATA/ + {product_name}.json): {zip_path.name}"
        )

    logger.info(f"Extracted to: {extract_target}")
    return extract_target


def zip_product(product_root, output_zip_path, logger):
    logger.info(f"Packaging new L1D output: {output_zip_path.name}")
    with zipfile.ZipFile(output_zip_path, "w", compression=zipfile.ZIP_DEFLATED) as zf:
        for path in sorted(product_root.rglob("*")):
            if path.is_file():
                arcname = path.relative_to(product_root.parent)
                stored = path.suffix.lower() in STORED_IN_ZIP
                zf.write(path, arcname, compress_type=zipfile.ZIP_STORED if stored else zipfile.ZIP_DEFLATED)
    logger.info(f"Wrote: {output_zip_path}")


INFERENCE_DEVICE = "cpu"       # the processor runs on CPU only (see CUDA_VISIBLE_DEVICES above)


def select_device(logger):
    logger.info(f"Inference device: {INFERENCE_DEVICE} (the processor runs on CPU only)")
    return INFERENCE_DEVICE


class StepTimer:
    """wall-clock seconds per processing step, for the "Processing times" log line"""

    def __init__(self):
        self.start = self.last = time.perf_counter()
        self.steps = []

    def mark(self, name, last_part=None):
        """time since the previous mark; last_part=(name, seconds) splits off the step's final part"""
        now = time.perf_counter()
        spent, self.last = now - self.last, now
        if last_part is not None:
            spent -= last_part[1]
        self.steps.append((name, spent))
        if last_part is not None:
            self.steps.append(last_part)

    def summary(self):
        parts = [f"{name} {seconds:.1f}" for name, seconds in self.steps]
        return " | ".join(parts + [f"total {time.perf_counter() - self.start:.1f}"])


def make_l1d_image(product, clut_path, logger):
    """
    Colour-corrects the product at full resolution with the method the AUX
    file selects, and puts the result in IMG_DATA in place of the L1C image,
    under the same file name: COG, LZW (lossless), uint16 = reflectance x
    10000 + 1000, 0 = no data. Returns the facts for the logs, the XML and
    the STAC entry.
    """
    cc = ColourCorrection(clut_path)
    logger.info(f"Colour correction: {cc.describe()}")
    gains = [product["calibration"][b][0] for b in (1, 2, 3)]
    offsets = [product["calibration"][b][1] for b in (1, 2, 3)]
    tif_path = product["tif_path"]
    l1d_path = tif_path.with_name(f"{tif_path.stem}.L1D.tif")
    logger.progress("Running colour correction at full resolution")
    try:
        info = write_colour_product(tif_path, gains, offsets, l1d_path, cc, logger)
        l1d_path.replace(tif_path)                 # IMG_DATA now holds only the L1D image
    finally:
        if l1d_path.exists():
            l1d_path.unlink()
    logger.info(f"L1D image: IMG_DATA/{tif_path.name} (COG, LZW), {info['valid_pixels']} valid pixels, "
                f"method {info['method']}; colour correction {info['seconds_correction']:.1f} s, "
                f"COG writing {info['seconds_cog']:.1f} s")
    return info


def process_product(input_zip_path, clut_path, device, logger, timer):
    """
    input_zip_path: absolute Path to the L1C RGB3 product zip, as supplied
    by the Job Order. Extracted and enriched under its original (L1C)
    name, then the whole product is renamed to the derived L1D product ID
    (per Enrico/Paola's confirmed naming convention) and packaged as a
    NEW output zip in the same working directory. The original input zip
    is never modified or deleted.
    clut_path: the colour-correction AUX file (the Job Order's, or the processor's own).
    Returns the output zip path.
    """
    working_dir = input_zip_path.parent
    old_product_root = extract_zip(input_zip_path, working_dir, logger)
    old_scene_id = old_product_root.name
    aux_data_path = old_product_root / "AUX_DATA"
    aux_data_path.mkdir(parents=True, exist_ok=True)
    json_path = old_product_root / f"{old_scene_id}.json"
    timer.mark("unzip")

    # --- read + calibrate ---------------------------------------------------
    product = read_product(old_product_root, json_path, logger)
    image = product["image"]
    profile = product["profile"]
    src_resolution = product["resolution"]

    orig_h, orig_w, _ = image.shape
    timer.mark("read + calibrate")

    # --- footprint + resample -------------------------------------------------
    valid_full = get_valid_footprint(image, nodata_value=NODATA_VALUE)
    logger.info(f"Valid pixels at native resolution: {int(valid_full.sum())}")

    image_chw = np.transpose(image, (2, 0, 1))

    img_inf = downsample_to_target_resolution(
        image_chw, src_res=src_resolution, dst_res=INFERENCE_RESOLUTION,
    )
    valid_inf = downsample_to_target_resolution(
        valid_full.astype("float32")[None, :, :],
        src_res=src_resolution, dst_res=INFERENCE_RESOLUTION,
    )[0] > 0.5

    img_inf[:, ~valid_inf] = 0
    logger.info(f"Inference shape: {img_inf.shape}")

    # --- inference (full-scene, no tiling) --------------------------------------
    logger.progress("Running full-scene OmniCloudMask inference")
    _, _, probs_inf = run_omnicloud_on_array(
        img_inf, device=device, divide_by_10000=True, inference_dtype="fp32",
    )

    probs_inf[:, ~valid_inf] = 0.0
    probs_inf[0, ~valid_inf] = 1.0

    probs_full = upsample_prob_stack(probs_inf, orig_h, orig_w)
    probs_full[:, ~valid_full] = 0.0
    probs_full[0, ~valid_full] = 1.0

    four_class = probs_to_four_class_map(probs_full)
    conf = probs_to_cloud_confidence(probs_full)

    four_class[~valid_full] = 0
    conf[~valid_full] = 0.0

    stats = compute_cloud_stats(four_class, valid_mask=valid_full)
    logger.info(f"Cloud stats: {stats}")
    cloud_cover_pct = stats["cloud_pct"]

    four_class_out = (four_class.astype("uint16") + 1).astype("uint8")
    four_class_out[~valid_full] = 0

    # --- write the cloud outputs, still under the OLD (L1C) name -------------------
    clm_path = aux_data_path / f"{old_scene_id}.CLM.tif"
    com_path = aux_data_path / f"{old_scene_id}.COM.tif"

    logger.info(f"Writing CLM: {clm_path.name}")
    save_clm_output(clm_path, four_class_out, profile)

    logger.info(f"Writing COM: {com_path.name}")
    save_com_output(com_path, conf, profile)

    legend_path = aux_data_path / f"{old_scene_id}.CLM.json"
    save_clm_legend(legend_path, logger)
    timer.mark("cloud mask")

    # --- the L1D image: colour correction, after the cloud mask ---------------------
    # The full-resolution arrays of the cloud step are no longer needed; free
    # them first. The colour step reads the image again tile by tile.
    del image, image_chw, img_inf, probs_inf, probs_full, four_class, conf, four_class_out, valid_full
    product.pop("image", None)
    l1d = make_l1d_image(product, clut_path, logger)
    timer.mark("colour correction", last_part=("COG writing", l1d["seconds_cog"]))

    # --- quicklooks drawn from the L1D image ---------------------------------------
    make_l1d_previews(old_product_root, product["tif_path"], logger)
    timer.mark("quicklooks")

    # --- the new L1D product ID (generation time = now): the XML header names it ---------
    generation_time = datetime.now(timezone.utc)
    new_scene_id = derive_output_product_id(old_scene_id, generation_time=generation_time)

    # --- metadata of the L1D product: XML and STAC -----------------------------------
    with open(json_path) as f:
        xml_path = find_product_xml(old_product_root, json.load(f))
    if xml_path is None:
        logger.warning("No product XML: no Rescaling_Factors, image checksum or L1D header to update")
    else:
        set_l1d_image_in_xml(xml_path, product["tif_path"], L1D_GAIN, L1D_OFFSET, logger)
        set_l1d_header_in_xml(xml_path, old_scene_id, new_scene_id, generation_time, input_zip_path.name,
                              clut_path, PROCESSOR_VERSION, logger)

    enrich_sidecar_json(
        json_path, old_product_root, cloud_cover_pct, clm_path, com_path, legend_path,
        profile, stats, logger, l1d_info=l1d,
        input_product_id=old_scene_id, processor_version=PROCESSOR_VERSION,
    )
    timer.mark("metadata")

    # --- rename the whole product to the new L1D naming ---------------------------
    new_product_root = rename_product_in_place(old_product_root, old_scene_id, new_scene_id, logger)
    new_json_path = new_product_root / f"{new_scene_id}.json"
    update_json_references(new_json_path, old_scene_id, new_scene_id, generation_time, logger)
    point_kml_at_renamed_files(new_product_root, old_scene_id, new_scene_id, logger)
    refresh_file_info(new_json_path, logger)
    timer.mark("rename + checksums")

    # --- package as a NEW output zip; input zip is never touched -------------------
    output_zip_path = working_dir / f"{new_scene_id}.zip"
    zip_product(new_product_root, output_zip_path, logger)

    shutil.rmtree(old_product_root)
    shutil.rmtree(new_product_root)
    logger.info(f"Removed temporary working folders for: {old_scene_id}")
    timer.mark("zip")

    logger.info(f"Image {orig_w} x {orig_h} pixels ({orig_w * orig_h / 1e6:.0f} Mpx), colour method "
                f"{l1d['method']}, {os.cpu_count()} CPUs")
    logger.info(f"Processing times (s): {timer.summary()}")
    return output_zip_path


def main():
    timer = StepTimer()
    if len(sys.argv) != 2:
        # No logger available yet (we don't know the working dir until we
        # parse the Job Order), so this one message goes straight to stderr.
        sys.stderr.write(
            "Usage: process_l1c.py <absolute_path_to_JobOrder.xml>\n"
        )
        sys.exit(EXIT_FAILURE)

    job_order_path = Path(sys.argv[1])

    # Bootstrap logger before Job Order is parsed, so parse errors are
    # still logged in the required format. Working dir / logging level
    # will be refined once the Job Order is actually read.
    logger = PdgsLogger(
        processor_name=PROCESSOR_NAME,
        processor_version=PROCESSOR_VERSION,
        working_dir=job_order_path.parent,
        logging_level="INFO",
    )

    logger.info(f"[Start] {PROCESSOR_NAME} {PROCESSOR_VERSION}")

    try:
        job_order = parse_job_order(job_order_path, TASK_NAME, TASK_VERSION, logger=logger)
    except JobOrderError as e:
        logger.error(f"Job Order error: {e}")
        logger.info(f"[Stop] exit_code={EXIT_FAILURE}")
        sys.exit(EXIT_FAILURE)

    # Re-create the logger with the real logging level and working dir
    # from the Job Order.
    logger = PdgsLogger(
        processor_name=PROCESSOR_NAME,
        processor_version=PROCESSOR_VERSION,
        working_dir=job_order["working_dir"],
        logging_level=job_order["logging_level"],
    )

    for f in job_order["input_files"]:
        logger.info(f"[Input] {f}")

    if job_order["output_spec"]:
        spec = job_order["output_spec"]
        logger.info(f"[Expected Output] file_type={spec.get('file_type')} "
                    f"reg_exp={spec.get('reg_exp')}")

    by_type = job_order["inputs_by_type"]
    product_inputs = [p for t, paths in by_type.items() if t != CLUT_FILE_TYPE for p in paths]
    if not product_inputs:
        logger.error(f"No {PRODUCT_FILE_TYPE} product among the Job Order inputs")
        logger.info(f"[Stop] exit_code={EXIT_FAILURE}")
        sys.exit(EXIT_FAILURE)
    if len(product_inputs) > 1:
        logger.warning(
            f"Job Order declared {len(product_inputs)} product files for this "
            f"task; expected exactly 1. Using the first: {product_inputs[0]}"
        )
    input_zip_path = product_inputs[0]

    if not input_zip_path.exists():
        logger.error(f"Declared input file does not exist: {input_zip_path}")
        logger.info(f"[Stop] exit_code={EXIT_FAILURE}")
        sys.exit(EXIT_FAILURE)

    # Colour-correction AUX file: the one in the Job Order if PDGS gives one (the
    # newest, by name, if several), otherwise the processor's own in aux/.
    clut_inputs = sorted(by_type.get(CLUT_FILE_TYPE, []), key=lambda p: p.name)
    if clut_inputs:
        clut_path = clut_inputs[-1]
        if len(clut_inputs) > 1:
            logger.warning(f"{len(clut_inputs)} {CLUT_FILE_TYPE} files in the Job Order; using the newest: {clut_path.name}")
        if not clut_path.exists():
            logger.error(f"Declared {CLUT_FILE_TYPE} file does not exist: {clut_path}")
            logger.info(f"[Stop] exit_code={EXIT_FAILURE}")
            sys.exit(EXIT_FAILURE)
        logger.info(f"Colour AUX file from the Job Order: {clut_path.name}")
    else:
        own = sorted(OWN_AUX_DIR.glob(f"IME___{CLUT_FILE_TYPE}_*.xml"), key=lambda p: p.name)
        if not own:
            logger.error(f"No colour-correction AUX file ({CLUT_FILE_TYPE}): none in the Job Order and none "
                         f"in {OWN_AUX_DIR}, so no L1D image can be made")
            logger.info(f"[Stop] exit_code={EXIT_FAILURE}")
            sys.exit(EXIT_FAILURE)
        clut_path = own[-1]
        logger.info(f"Colour AUX file: none in the Job Order, using the processor's own: {clut_path.name}")

    device = select_device(logger)

    try:
        output_path = process_product(input_zip_path, clut_path, device, logger, timer)
        logger.info(f"[Output] {output_path} OUTPUT_READY")
        logger.info(f"[Stop] exit_code={EXIT_OK}")
        sys.exit(EXIT_OK)

    except Exception:
        logger.error(f"Processing failed for {input_zip_path}")
        logger.error(traceback.format_exc())
        logger.info(f"[Stop] exit_code={EXIT_FAILURE}")
        sys.exit(EXIT_FAILURE)


if __name__ == "__main__":
    main()