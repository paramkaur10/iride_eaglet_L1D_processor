# resampling.py
"""
Native GSD <-> inference resolution resampling.

Uses the same GDAL/rasterio bilinear resampling as before — no change to
the resampling algorithm or numerical output. The only change is batching:
all bands are written into and read from a single in-memory dataset per
call, instead of opening a fresh dataset per band in a loop. Same accuracy,
less redundant open/write/read overhead.
"""

import numpy as np
import rasterio
from rasterio.enums import Resampling


def downsample_to_target_resolution(img, src_res, dst_res):
    """
    img: CHW array
    """
    scale = src_res / dst_res
    bands, h, w = img.shape
    new_h = max(1, int(round(h * scale)))
    new_w = max(1, int(round(w * scale)))

    with rasterio.io.MemoryFile() as memfile:
        profile = {
            "driver": "GTiff",
            "height": h,
            "width": w,
            "count": bands,
            "dtype": str(img.dtype),
        }
        with memfile.open(**profile) as ds:
            ds.write(img)
            out = ds.read(
                out_shape=(bands, new_h, new_w),
                resampling=Resampling.bilinear,
            ).astype("float32")

    return out


def upsample_stack(arr, target_h, target_w, resampling=Resampling.bilinear):
    """
    arr: (bands, H, W) array — resamples all bands in a single dataset.
    Replaces the old per-band upsample_array loop.
    """
    bands = arr.shape[0]

    with rasterio.io.MemoryFile() as memfile:
        profile = {
            "driver": "GTiff",
            "height": arr.shape[1],
            "width": arr.shape[2],
            "count": bands,
            "dtype": str(arr.dtype),
        }
        with memfile.open(**profile) as ds:
            ds.write(arr)
            out = ds.read(
                out_shape=(bands, target_h, target_w),
                resampling=resampling,
            )

    return out


def upsample_prob_stack(probs, target_h, target_w):
    """
    probs: (4, H, W) probability stack -> upsampled to (4, target_h, target_w).
    """
    return upsample_stack(
        probs.astype("float32"), target_h, target_w, resampling=Resampling.bilinear
    ).astype("float32")