# product_naming.py
"""
Derives the L1D output product ID from an L1C input product ID, per the
naming convention confirmed by Enrico/Paola (email thread):

  Input:  IME08_1CSP__RGB3_20260921T070225_20260921T070227_20260921T130751_04048_R020_O_A00
  Output: IME08_1DSP__RGB3_20260921T070225_20260921T070227_<NEW_GEN_TIME>_04048_R020_O_A00

Transformation:
  - "1CSP" -> "1DSP" (processing level token: L1C -> new L1D)
  - 3rd timestamp segment (generation time) replaced with the actual
    processing time of THIS run, not copied from the input
  - Sensing start/stop (1st, 2nd timestamps) and every trailing segment
    (orbit numbers, observation direction, baseline) are preserved as-is
"""

import re
from datetime import datetime, timezone


# IME08_1CSP__RGB3_20260921T070225_20260921T070227_20260921T130751_04048_R020_O_A00
PRODUCT_ID_PATTERN = re.compile(
    r"^(?P<mission>[A-Z0-9]+)_"
    r"(?P<level>1CSP)__RGB3_"
    r"(?P<t_start>\d{8}T\d{6})_"
    r"(?P<t_stop>\d{8}T\d{6})_"
    r"(?P<t_gen>\d{8}T\d{6})_"
    r"(?P<trailer>.+)$"
)

NEW_LEVEL_TOKEN = "1DSP"


class ProductNamingError(Exception):
    pass


def derive_output_product_id(input_product_id, generation_time=None):
    """
    input_product_id: the L1C product ID (no .zip extension), e.g.
        "IME08_1CSP__RGB3_20260921T070225_20260921T070227_20260921T130751_04048_R020_O_A00"
    generation_time: datetime to use as the new generation timestamp;
        defaults to now (UTC).

    Returns the derived L1D product ID string.
    """
    match = PRODUCT_ID_PATTERN.match(input_product_id)
    if not match:
        raise ProductNamingError(
            f"Input product ID does not match the expected L1C RGB3 "
            f"naming convention: {input_product_id}"
        )

    if generation_time is None:
        generation_time = datetime.now(timezone.utc)
    new_gen_str = generation_time.strftime("%Y%m%dT%H%M%S")

    output_id = (
        f"{match.group('mission')}_{NEW_LEVEL_TOKEN}__RGB3_"
        f"{match.group('t_start')}_{match.group('t_stop')}_"
        f"{new_gen_str}_{match.group('trailer')}"
    )
    return output_id


if __name__ == "__main__":
    example_input = "IME08_1CSP__RGB3_20260921T070225_20260921T070227_20260921T130751_04048_R020_O_A00"
    test_gen_time = datetime(2026, 9, 21, 13, 7, 51, tzinfo=timezone.utc)
    result = derive_output_product_id(example_input, generation_time=test_gen_time)
    expected = "IME08_1DSP__RGB3_20260921T070225_20260921T070227_20260921T130751_04048_R020_O_A00"
    assert result == expected, f"Mismatch:\n  got:      {result}\n  expected: {expected}"
    print("Self-test passed:", result)


# ---------------------------------------------------------------------------
# Full product rename: folder, every file whose name carries the product's
# own ID, and every internal JSON reference to that ID.
#
# Files that do NOT contain the exact old product ID as a substring (e.g.
# AUX_DATA/*.anc.xml, which references a different upstream ancillary
# product with its own distinct ID) are left completely untouched — this
# falls out naturally from doing a substring match/replace per filename,
# no special-casing needed.
# ---------------------------------------------------------------------------

import json
import shutil
from pathlib import Path


def rename_product_in_place(extracted_product_root, old_scene_id, new_scene_id, logger):
    """
    Copies extracted_product_root (named old_scene_id) into a sibling
    folder named new_scene_id, renaming every file whose name contains
    old_scene_id as a substring to use new_scene_id instead. Files that
    don't contain old_scene_id (e.g. ancillary references to a different
    product) are copied with their names unchanged.

    Returns the new product root Path. The original extracted folder is
    left untouched by this function (caller is responsible for cleanup).
    """
    extracted_product_root = Path(extracted_product_root)
    new_product_root = extracted_product_root.parent / new_scene_id

    if new_product_root.exists():
        raise FileExistsError(f"Rename target already exists: {new_product_root}")

    logger.info(f"Renaming product: {old_scene_id} -> {new_scene_id}")

    for src_path in sorted(extracted_product_root.rglob("*")):
        if not src_path.is_file():
            continue

        rel_path = src_path.relative_to(extracted_product_root)
        renamed_parts = [
            part.replace(old_scene_id, new_scene_id) for part in rel_path.parts
        ]
        dst_path = new_product_root.joinpath(*renamed_parts)

        dst_path.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(src_path, dst_path)

    logger.info(f"Renamed product written to: {new_product_root}")
    return new_product_root


def update_json_references(json_path, old_scene_id, new_scene_id, generation_time, logger):
    """
    Updates the sidecar JSON in place (now living at its renamed path)
    so its own content matches the renamed product:
      - "id" -> new_scene_id
      - every asset's "href" / "file:local_path" containing old_scene_id
        -> new_scene_id (assets that don't reference old_scene_id, e.g.
        the ancillary asset, are left untouched)
      - "properties.created" / "properties.updated" -> generation_time,
        reflecting that this is a newly generated product
    """
    json_path = Path(json_path)

    with open(json_path, "r") as f:
        stac = json.load(f)

    if stac.get("id") == old_scene_id:
        stac["id"] = new_scene_id

    for asset in stac.get("assets", {}).values():
        for key in ("href", "file:local_path"):
            if key in asset and isinstance(asset[key], str) and old_scene_id in asset[key]:
                asset[key] = asset[key].replace(old_scene_id, new_scene_id)

    gen_time_str = generation_time.strftime("%Y-%m-%dT%H:%M:%S.%f") + "Z"
    if "properties" in stac:
        stac["properties"]["created"] = gen_time_str
        stac["properties"]["updated"] = gen_time_str

    with open(json_path, "w") as f:
        json.dump(stac, f, indent=4)

    logger.info(f"Updated JSON references for renamed product: {json_path}")