# previews.py
"""
Quicklooks of the L1D product.

The L1C product's PREVIEW folder holds downsampled copies of the image (a
PNG and a JPEG) and a KML that places the JPEG on the map. In the L1D product
the images are redrawn from the L1D image, with the same names, formats and
sizes; the PNG gets an alpha channel, transparent where there is no data
(the JPEG format has no transparency, so no data stays black there), and the
KML places the PNG on the map instead of the JPEG. The KML is pointed at the
renamed files after the product is renamed. In the STAC of the L1D the PNG is
the "quicklook" and the JPEG the "thumbnail", the reverse of the L1C
(metadata.point_quicklook_at_png).

The stretch is the display range of the STAC renders, from the colour step
(the whole image at full resolution, saturated pixels left out;
colour_correction.display_range): each band starts at its own dark level, its
2nd percentile, and the three bands share one span, the 98th percentile of
the three after their dark levels; then a gamma of 1.6 brightens it. One
stretch for the three bands, as before 01.02, left the quicklooks bluish: the
haze of top-of-atmosphere colour, which the LUT keeps, as Sentinel-2 L1C has
it, and which is largest in blue. Taking each band's dark level away removes
it; the shared span keeps the colours of the correction. Clouds saturated in
the L1C are white in the L1D image, so they show white.
"""

import re
from pathlib import Path

import numpy as np

from iride_eaglet_processor.colour_correction import RENDER_PERCENTILES

PREVIEW_IMAGES = (".jpg", ".jpeg", ".png")
STRETCH_PERCENTILES = RENDER_PERCENTILES   # (2, 98): dark level of each band, span of the three (no range given)
GAMMA = 1.6                                # display brightness, after the stretch
JPEG_QUALITY = 90
KML_HREF = re.compile(rb"(<href>)([^<]*)(</href>)", re.I)


def make_l1d_previews(product_root, image_path, logger, limits=None):
    """
    Redraws every JPEG / PNG in PREVIEW from the L1D image (image_path), at
    each file's own size and in its own format: each band stretched between
    its limits, then GAMMA. limits: [[low, high]] per band in L1D values, the
    render range of the colour step; without it, the same kind of range from
    the largest quicklook (each band's 2nd percentile, the 98th percentile of
    the three after it as the span). A PNG is written as RGBA:
    alpha 0 (transparent) where there is no data, 255 elsewhere. A JPEG keeps
    its colour mode, with no data black. Then the KML places the PNG on the
    map (point_kml_at_png). Returns the image files written.
    """
    import rasterio
    from PIL import Image
    from rasterio.enums import Resampling

    paths = sorted(p for p in (Path(product_root) / "PREVIEW").glob("*") if p.suffix.lower() in PREVIEW_IMAGES)
    if not paths:
        logger.info("No quicklook images in PREVIEW to redraw")
        return []
    specs = {}
    for p in paths:
        with Image.open(p) as im:
            specs[p] = (im.size, im.mode, im.format)

    lo = hi = None
    if limits is not None:
        lo, hi = np.asarray(limits, dtype="float64").T
        hi = np.maximum(hi, lo + 1.0)
    with rasterio.open(image_path) as src:
        for p in sorted(paths, key=lambda q: -specs[q][0][0] * specs[q][0][1]):   # largest first (sets lo, hi
            (w, h), mode, fmt = specs[p]
            a = src.read([1, 2, 3], out_shape=(3, h, w), resampling=Resampling.average).astype("float32")
            valid = (a > 0).all(0)
            if lo is None:                         # when no limits are given), per band
                if valid.any():
                    lo = np.percentile(a[:, valid], STRETCH_PERCENTILES[0], axis=1)
                    hi = lo + np.percentile((a[:, valid] - lo[:, None]).ravel(), STRETCH_PERCENTILES[1])
                else:
                    lo, hi = np.zeros(3), np.ones(3)
                hi = np.maximum(hi, lo + 1.0)
            x = np.clip((a - lo[:, None, None]) / (hi - lo)[:, None, None], 0, 1) ** (1 / GAMMA)
            rgb = np.clip(x * 255 + 0.5, 0, 255).astype("uint8")
            rgb[:, ~valid] = 0
            img = Image.fromarray(np.ascontiguousarray(np.moveaxis(rgb, 0, -1)), "RGB")
            if fmt == "PNG":
                mode = "RGBA"                      # transparent where there is no data
            if mode in ("RGBA", "LA"):
                img.putalpha(Image.fromarray(valid.astype("uint8") * 255, "L"))
            if img.mode != mode:
                img = img.convert(mode)
            tmp = p.with_name(p.name + ".part")
            img.save(tmp, format=fmt, **({"quality": JPEG_QUALITY} if fmt == "JPEG" else {}))
            tmp.replace(p)
            logger.info(f"Quicklook drawn from the L1D image: PREVIEW/{p.name} ({fmt} {img.mode} {w} x {h})")
    logger.info("Quicklook stretch, each band from its dark level over one span (L1D values): "
                + ", ".join(f"{band} {low:.0f} to {high:.0f}" for band, low, high in zip("RGB", lo, hi))
                + f", gamma {GAMMA}")
    point_kml_at_png(product_root, logger)
    return paths


def point_kml_at_png(product_root, logger):
    """
    The KML in PREVIEW places one image on the map: in the L1C the JPEG, which
    cannot be transparent. In the L1D it places the PNG of the same name (same
    footprint, transparent where there is no data; Enrico, 13 Oct 2026). Each
    KML must have exactly one <href>, naming a JPEG or a PNG that has a PNG of
    the same name in PREVIEW; a KML that already places the PNG is left as it
    is. Anything else stops the processor.
    """
    preview = Path(product_root) / "PREVIEW"
    for kml in sorted(preview.glob("*.kml")):
        data = kml.read_bytes()
        hrefs = KML_HREF.findall(data)
        if len(hrefs) != 1:
            raise ValueError(f"PREVIEW/{kml.name}: expected one <href>, found {len(hrefs)}")
        href = hrefs[0][1].decode("utf-8").strip()
        stem, _, ext = href.rpartition(".")
        if ext.lower() not in ("jpg", "jpeg", "png"):
            raise ValueError(f"PREVIEW/{kml.name}: expected it to place the JPEG or the PNG; it places {href!r}")
        png = f"{stem}.png"
        if not (preview / Path(png).name).is_file():
            raise ValueError(f"PREVIEW/{kml.name}: it places {href!r}, but there is no {Path(png).name} in PREVIEW")
        if ext.lower() == "png":
            logger.info(f"PREVIEW/{kml.name}: already places {Path(png).name} on the map")
            continue
        kml.write_bytes(KML_HREF.sub(lambda m: m.group(1) + png.encode("utf-8") + m.group(3), data, count=1))
        logger.info(f"PREVIEW/{kml.name}: places {Path(png).name} (transparent) on the map instead of the JPEG")


def point_kml_at_renamed_files(product_root, old_id, new_id, logger):
    """After the rename: the KML in PREVIEW names the quicklook by the old product ID; use the new one"""
    for kml in sorted((Path(product_root) / "PREVIEW").glob("*.kml")):
        data = kml.read_bytes()
        if old_id.encode() in data:
            kml.write_bytes(data.replace(old_id.encode(), new_id.encode()))
            logger.info(f"PREVIEW/{kml.name}: points to the renamed quicklook")
