#!/usr/bin/env python3
"""
l1d_figure.py: one L1D product in one picture, as in the README: its quicklook, its cloud mask (CLM) over the
L1D image, and its cloud confidence (COM). Cloud mask colours: thick cloud red, thin cloud yellow, cloud shadow
blue; clear pixels show the image. Confidence: 0 dark purple to 1 yellow.

    python tools/l1d_figure.py <L1D> <out .jpg>

The L1D is a zip, or a folder holding one (a working directory of docker/test_image.sh). Needs numpy, rasterio
and Pillow, as in the processor's image (with tools/ mounted, for compare_panels.py next to this file):
    docker run --rm --network none -u 0 -v "$PWD/tools:/tools:ro" -v <folder>:/w ipf-ime-msi-l1d:01.02 \
        python /tools/l1d_figure.py /w/<L1D> /w/l1d_example.jpg
"""
import sys
from pathlib import Path

import rasterio
from PIL import Image, ImageDraw
from rasterio.enums import Resampling

sys.path.insert(0, str(Path(__file__).resolve().parent))
import compare_panels as lp  # noqa: E402


def figure(path, out):
    p = lp.L1D(path)
    with rasterio.open(p.image) as src:
        H, W = src.height, src.width
        scale = min(lp.BOX / W, lp.BOX / H)
        shape = (max(1, round(H * scale)), max(1, round(W * scale)))
        img = src.read(out_shape=(3, *shape), resampling=Resampling.average)
    rgb = lp.show(img, p.display_range())
    with rasterio.open(p.clm) as src:
        clm = src.read(1, out_shape=shape, resampling=Resampling.mode)
    with rasterio.open(p.com) as src:
        com = src.read(1, out_shape=shape, resampling=Resampling.average) / 1e4
    com_rgb = lp.ramp(com)
    com_rgb[clm == 0] = lp.BG
    cover = f"{p.cover:.1f}%" if p.cover is not None else "?"
    cells = [(lp.quicklook(p), "Quicklook"),
             (lp.fit(lp.overlay(rgb, clm)), f"Cloud mask: {cover} cloud (thick red, thin yellow, shadow blue)"),
             (lp.fit(com_rgb), "Cloud confidence: 0 purple to 1 yellow")]
    sat, start = p.id[:5], p.id[17:25]
    title = f"Eaglet-2 L1D, {sat}, {start[6:8]}.{start[4:6]}.{start[:4]}, processor {p.version}"
    width = 3 * lp.BOX + 4 * lp.GAP
    height = lp.GAP + 44 + 30 + 30 + max(im.height for im, _ in cells) + lp.GAP
    page = Image.new("RGB", (width, height), lp.BG)
    d = ImageDraw.Draw(page)
    d.text((lp.GAP, lp.GAP), title, fill=lp.INK, font=lp.font(30))
    d.text((lp.GAP, lp.GAP + 44), p.id, fill=lp.SOFT, font=lp.font(20))
    y = lp.GAP + 44 + 30
    for i, (im, caption) in enumerate(cells):
        x = lp.GAP + i * (lp.BOX + lp.GAP)
        d.text((x, y), caption, fill=lp.SOFT, font=lp.font(20))
        page.paste(im, (x + (lp.BOX - im.width) // 2, y + 30))
    Path(out).parent.mkdir(parents=True, exist_ok=True)
    page.save(out, quality=88, subsampling=0)
    print(f"{p.id}: cloud {cover} -> {out}")


if __name__ == "__main__":
    if len(sys.argv) != 3:
        sys.exit(__doc__)
    figure(sys.argv[1], sys.argv[2])
