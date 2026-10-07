# raster_reader.py
"""
Eaglet-2 L1C product reader.

Reads the single 3-band GeoTIFF from IMG_DATA/ by fixed band index
(Band 1/2/3 = Red/Green/Blue per the L1C product spec), applies per-band
radiometric calibration (DN * gain + offset -> TOA reflectance, then
x10000 to match the model's expected input convention), and preserves
nodata (raw DN == 0) through calibration.

Calibration gain/offset are read live from each product, never cached or
hardcoded: from the product XML's Rescaling_Factors (exact values), with the
STAC sidecar's rounded raster:scale/offset only as a fallback (see
calibration.py).
"""

from pathlib import Path

import numpy as np
import rasterio

from iride_eaglet_processor.calibration import load_calibration


BAND_INDICES = [1, 2, 3]   # Red, Green, Blue — fixed order per L1C spec
NODATA_VALUE = 0


def find_single_tif(img_data_path):
    img_data_path = Path(img_data_path)
    tif_files = sorted([
        p for p in img_data_path.iterdir()
        if p.is_file()
        and p.suffix.lower() in (".tif", ".tiff")
        and ".CLM." not in p.name
        and ".COM." not in p.name
        and ".COL." not in p.name
    ])
    if len(tif_files) == 0:
        raise FileNotFoundError(f"No product .tif found in: {img_data_path}")
    if len(tif_files) > 1:
        raise ValueError(
            f"Expected exactly one product .tif in {img_data_path}, "
            f"found {len(tif_files)}: {[p.name for p in tif_files]}"
        )
    return tif_files[0]


def read_product(product_root, json_path, logger):
    """
    Returns dict with:
      image      HWC float32, calibrated (TOA reflectance x10000), nodata=0
      calibration {band_index: (gain, offset)} used, and calibration_source
      profile    rasterio profile of the source tif (for output georeferencing)
      resolution pixel size in metres, read from the raster's own geotransform
                 (GSD varies per acquisition — never trust a stored constant)
      tif_path   path to the source tif
    """
    product_root = Path(product_root)
    img_data_path = product_root / "IMG_DATA"
    if not img_data_path.exists():
        raise FileNotFoundError(f"IMG_DATA not found: {img_data_path}")

    tif_path = find_single_tif(img_data_path)
    logger.info(f"Product raster: {tif_path}")

    calibration, calibration_source = load_calibration(product_root, json_path, logger)
    logger.info(f"Calibration source: {calibration_source}")

    bands = []
    with rasterio.open(tif_path) as src:
        profile = src.profile.copy()
        resolution = abs(src.transform.a)
        logger.info(f"Resolution from geotransform: {resolution:.6f} m")

        for band_idx in BAND_INDICES:
            raw = src.read(band_idx)
            raw_nodata_mask = (raw == NODATA_VALUE)

            scale, offset = calibration[band_idx]
            arr = raw.astype("float32") * scale + offset   # -> reflectance 0-1
            arr = arr * 10000.0                            # -> reflectance x10000

            # Calibration shifts raw-nodata pixels away from exactly 0
            # (0*scale + offset != 0 in general; offset is not always -scale).
            # Re-zero them so nodata stays nodata downstream.
            arr[raw_nodata_mask] = 0.0

            logger.info(
                f"Band {band_idx} calibrated | min={np.nanmin(arr):.2f} | "
                f"max={np.nanmax(arr):.2f} | nodata_pixels={int(raw_nodata_mask.sum())}"
            )
            bands.append(arr)

    image = np.stack(bands, axis=-1)
    logger.info(f"Calibrated image stack: {image.shape}")

    return {
        "image": image,
        "profile": profile,
        "resolution": resolution,
        "tif_path": tif_path,
        "calibration": calibration,
        "calibration_source": calibration_source,
    }