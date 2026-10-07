# masks.py

import numpy as np
import rasterio


def probs_to_four_class_map(probs):
    """
    OmniCloudMask's own native output classes (unshifted, model-internal):
      0 = clear
      1 = thick cloud
      2 = thin cloud
      3 = cloud shadow
    """
    return np.argmax(probs, axis=0).astype("uint8")


def probs_to_cloud_confidence(probs):
    return (probs[1] + probs[2]).astype("float32")


def _strip_incompatible_metadata(profile):
    """
    reference_profile is copied from the source 3-band RGB product tif,
    which carries RGB-specific tags (photometric interpretation, color
    interpretation) that become invalid once repurposed for a single-band
    output. Left in place, some tools (SNAP, some QGIS backends) fail to
    open the resulting file with a parser error, since a 1-band raster
    claiming RGB photometric interpretation is an internally inconsistent
    GeoTIFF header. GDAL itself is often lenient about this and opens it
    anyway, which is why the issue doesn't show up in every tool.
    """
    profile = profile.copy()
    for key in ("photometric", "colorinterp"):
        profile.pop(key, None)
    return profile


def save_clm_output(out_path, clm_mask, reference_profile):
    """
    Cloud mask, IRIDE enrichment naming convention: <SCENE>.CLM.tif

    NOTE: clm_mask passed in here must already be in the OUTPUT encoding
    (0=nodata, 1=clear, 2=thick cloud, 3=thin cloud, 4=cloud shadow) -
    i.e. the model's native 0-3 classes shifted +1, with nodata pixels
    set to 0. This shift happens in process_l1c.py, not in this function.
    """
    profile = _strip_incompatible_metadata(reference_profile)
    profile.update({
        "driver": "GTiff",
        "height": clm_mask.shape[0],
        "width": clm_mask.shape[1],
        "count": 1,
        "dtype": "uint8",
        "compress": "lzw",
    })
    with rasterio.open(out_path, "w", **profile) as dst:
        dst.write(clm_mask.astype("uint8"), 1)
        dst.set_band_description(
            1, "0_nodata_1_clear_2_thick_cloud_3_thin_cloud_4_shadow"
        )


def save_com_output(out_path, cloud_conf, reference_profile):
    """
    Cloud confidence map, IRIDE enrichment naming convention: <SCENE>.COM.tif

    Encoding: uint16, scaled x10000 (0-10000 representing confidence 0.0-1.0).
    """
    profile = _strip_incompatible_metadata(reference_profile)
    conf_uint16 = np.clip(cloud_conf * 10000, 0, 10000).astype("uint16")
    profile.update({
        "driver": "GTiff",
        "height": cloud_conf.shape[0],
        "width": cloud_conf.shape[1],
        "count": 1,
        "dtype": "uint16",
        "compress": "lzw",
    })
    with rasterio.open(out_path, "w", **profile) as dst:
        dst.write(conf_uint16, 1)
        dst.set_band_description(1, "cloud_confidence_x10000")