# metadata.py

import hashlib
import json
import os
from pathlib import Path

from iride_eaglet_processor.colour_correction import L1D_GAIN, L1D_OFFSET


# Class definitions: (value, name, title, description). Percentages are
# filled in per-scene from the actual computed stats, not hardcoded here.
# Matches our own 5 classes only (nodata + OmniCloudMask's 4 native
# classes) -- no land-cover discrimination (vegetation/water/urban/etc),
# since this processor does not implement a land-cover classifier.
CLM_CLASS_DEFINITIONS = [
    (0, "nodata", "No Data", "NoData"),
    (1, "clear", "Clear (%)", "Clear (non-cloud, non-shadow) area percentage."),
    (2, "thick_cloud", "Thick Cloud (%)", "Thick cloud area percentage."),
    (3, "thin_cloud", "Thin Cloud (%)", "Thin cloud area percentage."),
    (4, "cloud_shadow", "Cloud Shadow (%)", "Cloud shadow area percentage."),
]

CLASSIFICATION_EXTENSION_URL = (
    "https://stac-extensions.github.io/classification/v2.0.0/schema.json"
)

# The L1D image (IMG_DATA/<SCENE>.tif, STAC asset "rgb"): colour-corrected RGB,
# COG with LZW, uint16, reflectance = value * L1D_GAIN + L1D_OFFSET, 0 = no data.
IMAGE_ASSET = "rgb"
COG_MEDIA_TYPE = "image/tiff; application=geotiff; profile=cloud-optimized"
BAND_STATISTICS_KEYS = ("statistics", "raster:statistics", "raster:histogram")   # described the L1C values


def compute_sha256(path):
    """
    Returns a multihash-formatted checksum: the "1220" prefix identifies
    the algorithm (SHA-256) and digest length (32 bytes) self-descriptively,
    matching the format already used by the product's other assets
    (metadata, thumbnail, quicklook, auxiliary, etc). Without this prefix,
    a reader would have to already know the algorithm to interpret the
    string; with it, the checksum announces its own format.

    1220 = 0x12 (SHA-256 algorithm code) + 0x20 (32-byte digest length),
    both in hex, per the Multiformats multihash specification.
    """
    sha256 = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(8192), b""):
            sha256.update(chunk)
    return "1220" + sha256.hexdigest()


def build_classification_classes(stats):
    """
    Builds the STAC classification:classes array for the "clm" asset,
    using real per-scene percentages computed by compute_cloud_stats().
    class percentages (clear/thick/thin/shadow) are over valid pixels;
    nodata's percentage is over the full scene (matches eo:cloud_cover's
    own valid-pixels-only convention, confirmed self-consistent since
    thick_cloud_pct + thin_cloud_pct == eo:cloud_cover).
    """
    pct_lookup = {
        0: stats.get("nodata_pct", 0.0),
        1: stats.get("clear_pct", 0.0),
        2: stats.get("thick_cloud_pct", 0.0),
        3: stats.get("thin_cloud_pct", 0.0),
        4: stats.get("shadow_pct", 0.0),
    }

    classes = []
    for value, name, title, description in CLM_CLASS_DEFINITIONS:
        entry = {
            "value": value,
            "name": name,
            "title": title,
            "description": description,
            "percentage": round(float(pct_lookup.get(value, 0.0)), 4),
        }
        if value == 0:
            entry["nodata"] = True
        classes.append(entry)

    return classes


def save_clm_legend(legend_path, logger):
    """
    Writes the standalone CLM class legend file (<SCENE>.CLM.json),
    referenced by the "clm_aux" asset. Simple flat value->name mapping,
    separate from the richer classification:classes array embedded
    directly in the "clm" asset entry in the sidecar JSON.
    """
    legend = {str(value): name for value, name, _, _ in CLM_CLASS_DEFINITIONS}
    with open(legend_path, "w") as f:
        json.dump(legend, f, indent=4)
    logger.info(f"Wrote CLM class legend: {legend_path}")


def _proj_fields(profile):
    """
    Extracts proj:shape ([height, width]) and proj:transform
    ([a, b, c, d, e, f]) from a rasterio profile, matching the flat
    6-element format used elsewhere in IRIDE's STAC sidecars.
    """
    transform = profile["transform"]
    shape = [profile["height"], profile["width"]]
    flat_transform = [
        transform.a, transform.b, transform.c,
        transform.d, transform.e, transform.f,
    ]
    return shape, flat_transform


def _asset_entry(product_root, file_path, title, mime_type, roles, profile=None,
                  description=None, classification_classes=None):
    entry = {
        "href": f"./AUX_DATA/{file_path.name}",
        "type": mime_type,
        "title": title,
        "file:checksum": compute_sha256(file_path),
        "file:local_path": str(file_path.relative_to(product_root.parent)),
        "file:size": os.path.getsize(file_path),
        "roles": roles,
    }
    if description:
        entry["description"] = description
    if profile is not None:
        shape, transform = _proj_fields(profile)
        entry["proj:shape"] = shape
        entry["proj:transform"] = transform
    if classification_classes is not None:
        entry["classification:classes"] = classification_classes
    return entry


def describe_l1d_image(stac, l1d_info, logger):
    """
    Points the "rgb" asset at the L1D image: COG media type, band encoding
    (raster:scale / raster:offset / nodata / data_type; band statistics of
    the L1C dropped), title and description with the colour method; and the
    display range of the renders that use it (rescale: 2nd / 98th percentile
    of each band of the L1D image).
    """
    asset = stac["assets"].get(IMAGE_ASSET)
    if asset is None:
        raise KeyError(f"No '{IMAGE_ASSET}' asset in the STAC sidecar")
    asset["type"] = COG_MEDIA_TYPE
    asset["title"] = "L1D RGB, true colour"
    asset["description"] = (
        "Colour-corrected RGB (L1D) with Sentinel-2 L1C-like colours. Method "
        f"{l1d_info['method']}, colour AUX file {l1d_info['aux_file']}. Cloud Optimized GeoTIFF, LZW (lossless). "
        f"Reflectance = value x {L1D_GAIN} {L1D_OFFSET:+}; 0 = no data."
    )
    bands = asset.get("bands") or [{"name": n, "eo:common_name": n} for n in ("red", "green", "blue")]
    dropped = set()
    for band in bands[:3]:
        band.update({"raster:scale": L1D_GAIN, "raster:offset": L1D_OFFSET, "nodata": 0, "data_type": "uint16"})
        dropped.update(k for k in BAND_STATISTICS_KEYS if band.pop(k, None) is not None)
    asset["bands"] = bands

    rescale = l1d_info["rescale"]
    for holder in (stac.get("properties", {}), stac):
        renders = holder.get("renders")
        for name, render in (renders.items() if isinstance(renders, dict) else []):
            if not isinstance(render, dict) or IMAGE_ASSET not in render.get("assets", [IMAGE_ASSET]):
                continue
            if "rescale" in render:
                one_range = isinstance(render["rescale"], list) and len(render["rescale"]) == 1
                render["rescale"] = ([[min(r[0] for r in rescale), max(r[1] for r in rescale)]] if one_range
                                     else rescale)
            if "nodata" in render:
                render["nodata"] = 0
            logger.info(f"STAC render '{name}': rescale {render.get('rescale')}")
    if dropped:
        logger.info(f"STAC '{IMAGE_ASSET}' bands: dropped {sorted(dropped)} (they described the L1C values)")


def refresh_file_info(json_path, logger):
    """
    Recomputes file:checksum and file:size of every asset that points to a
    file inside the product, from the files as they are now (the L1D image
    and the XML change; renaming does not). Each checksum keeps its format:
    multihash ("1220" + SHA-256) or plain SHA-256 hex.
    """
    json_path = Path(json_path)
    with open(json_path, "r") as f:
        stac = json.load(f)
    changed = []
    for key, asset in stac.get("assets", {}).items():
        href = asset.get("href", "")
        path = (json_path.parent / href).resolve() if href and "://" not in href else None
        if path is None or not path.is_file():
            continue
        before = (asset.get("file:checksum"), asset.get("file:size"))
        if "file:size" in asset:
            asset["file:size"] = os.path.getsize(path)
        if "file:checksum" in asset:
            checksum = compute_sha256(path)
            multihash = len(str(before[0])) == 68 and str(before[0]).startswith("1220")
            asset["file:checksum"] = checksum if multihash else checksum[4:]
        if (asset.get("file:checksum"), asset.get("file:size")) != before:
            changed.append(key)
    with open(json_path, "w") as f:
        json.dump(stac, f, indent=4)
    logger.info(f"STAC file:checksum / file:size updated for: {', '.join(changed) or 'none'}")


def enrich_sidecar_json(
    json_path, product_root, cloud_cover_pct, clm_path, com_path, legend_path,
    profile, stats, logger, l1d_info,
):
    """
    Single read-modify-write pass over the existing STAC sidecar JSON
    (in place, no new file created):
      - adds/updates "eo:cloud_cover" in properties
      - adds "clm" (5-class cloud mask, with embedded classification:classes),
        "com" (confidence map), and "clm_aux" (class legend) to assets,
        matching IRIDE's established asset-naming convention (as used in
        HEO's own L2A scene-classification product), alongside the
        existing "auxiliary" (.anc.xml) entry.
      - points the "rgb" asset and its renders at the L1D image
        (describe_l1d_image); file checksums and sizes are refreshed at
        the end, after the rename (refresh_file_info)
      - registers the classification extension in stac_extensions if not
        already present.
    """
    json_path = Path(json_path)

    with open(json_path, "r") as f:
        stac = json.load(f)

    if "properties" not in stac:
        raise KeyError(f"Expected 'properties' key in sidecar JSON: {json_path}")
    stac["properties"]["eo:cloud_cover"] = round(float(cloud_cover_pct), 4)

    if "assets" not in stac:
        raise KeyError(f"Expected 'assets' key in sidecar JSON: {json_path}")

    classification_classes = build_classification_classes(stats)

    stac["assets"]["clm"] = _asset_entry(
        product_root, clm_path, "Scene Classification",
        "image/tiff; application=geotiff",
        roles=["data-mask", "visual"],
        profile=profile,
        description="Scene Classification",
        classification_classes=classification_classes,
    )
    stac["assets"]["com"] = _asset_entry(
        product_root, com_path, "Confidence Map",
        "image/tiff; application=geotiff",
        roles=["data"],
        profile=profile,
        description="Confidence Map",
    )
    stac["assets"]["clm_aux"] = _asset_entry(
        product_root, legend_path, "Scene Classification Mapping",
        "application/json",
        roles=["metadata"],
        description="JSON mapping of scene classification values",
    )
    describe_l1d_image(stac, l1d_info, logger)

    extensions = stac.setdefault("stac_extensions", [])
    if CLASSIFICATION_EXTENSION_URL not in extensions:
        extensions.append(CLASSIFICATION_EXTENSION_URL)

    with open(json_path, "w") as f:
        json.dump(stac, f, indent=4)

    logger.info(
        f"Updated sidecar JSON in place: eo:cloud_cover, assets.clm/com/clm_aux, "
        f"assets.{IMAGE_ASSET} (L1D image) -> {json_path}"
    )