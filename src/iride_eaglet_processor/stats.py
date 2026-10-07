# stats.py

import numpy as np


def compute_cloud_stats(four_class_mask, valid_mask=None):
    """
    OmniCloudMask's native classes (unshifted, model-internal):
      0 = clear
      1 = thick cloud
      2 = thin cloud
      3 = cloud shadow

    Percentages here are computed over VALID pixels only (nodata excluded
    from the denominator). nodata_pct is computed separately, over the
    FULL scene, since it describes the scene's footprint rather than a
    class within it.
    """
    total_pixels = int(four_class_mask.size)

    if valid_mask is not None:
        pixels = four_class_mask[valid_mask]
        nodata_pixels = int(total_pixels - valid_mask.sum())
    else:
        pixels = four_class_mask.ravel()
        nodata_pixels = 0

    nodata_pct = 100.0 * nodata_pixels / total_pixels if total_pixels > 0 else float("nan")

    if pixels.size == 0:
        return {
            "valid_pixels": 0,
            "nodata_pixels": nodata_pixels,
            "clear_pixels": 0,
            "thick_cloud_pixels": 0,
            "thin_cloud_pixels": 0,
            "shadow_pixels": 0,
            "cloud_pixels": 0,
            "nodata_pct": nodata_pct,
            "clear_pct": float("nan"),
            "thick_cloud_pct": float("nan"),
            "thin_cloud_pct": float("nan"),
            "shadow_pct": float("nan"),
            "cloud_pct": float("nan"),
        }

    valid_pixels = int(pixels.size)
    clear_pixels = int(np.sum(pixels == 0))
    thick_cloud_pixels = int(np.sum(pixels == 1))
    thin_cloud_pixels = int(np.sum(pixels == 2))
    shadow_pixels = int(np.sum(pixels == 3))
    cloud_pixels = thick_cloud_pixels + thin_cloud_pixels

    return {
        "valid_pixels": valid_pixels,
        "nodata_pixels": nodata_pixels,
        "clear_pixels": clear_pixels,
        "thick_cloud_pixels": thick_cloud_pixels,
        "thin_cloud_pixels": thin_cloud_pixels,
        "shadow_pixels": shadow_pixels,
        "cloud_pixels": cloud_pixels,
        "nodata_pct": nodata_pct,
        "clear_pct": 100.0 * clear_pixels / valid_pixels,
        "thick_cloud_pct": 100.0 * thick_cloud_pixels / valid_pixels,
        "thin_cloud_pct": 100.0 * thin_cloud_pixels / valid_pixels,
        "shadow_pct": 100.0 * shadow_pixels / valid_pixels,
        "cloud_pct": 100.0 * cloud_pixels / valid_pixels,
    }