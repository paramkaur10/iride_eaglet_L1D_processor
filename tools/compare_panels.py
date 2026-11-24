#!/usr/bin/env python3
"""
compare_panels.py: what changed between two L1D versions of the same L1C products, one picture per product.
A is the older version (e.g. 01.01), B the newer (e.g. 01.02).

  1  whole scene    A's quicklook           | B's quicklook           | what changed in the L1D image
  2  cloud mask     A's CLM over the image  | B's CLM over the image  | where the two cloud masks differ
  3  confidence     A's COM                 | B's COM                 | B minus A
  4  2 km at full resolution where only A sees cloud the most: the L1D image | A's CLM | B's CLM
  5  the same, where only B sees cloud the most

Colours. Cloud masks: thick cloud red, thin cloud yellow, cloud shadow blue; clear pixels show the image.
Where the masks differ: white cloud in both, orange cloud only in A, cyan cloud only in B; the numbered boxes
are the 2 km of rows 4 and 5. Confidence: 0 dark purple to 1 yellow. B minus A: red where B is more confident,
blue where A is, black where they agree. What changed in the L1D image: magenta where B's pixels differ from
A's (from 01.02, pixels saturated in the L1C are written white). The images of rows 2, 4 and 5 are B's L1D
image with B's display range (its STAC render range and a gamma of 1.6, as B's quicklook).

    python tools/compare_panels.py <out folder> <L1D A> <L1D B> [<L1D A> <L1D B> ...]

Each L1D is a zip, or a folder holding one (a working directory of docker/test_image.sh). Writes
<out folder>/<L1C product ID>.jpg and prints one line per product. Needs numpy, rasterio and Pillow, as in
the processor's image:
    docker run --rm -i --network none -u 0 -v <folder>:/w ipf-ime-msi-l1d:01.02 \
        python - /w/panels /w/<A> /w/<B> < tools/compare_panels.py
"""
import io
import json
import re
import sys
import warnings
import zipfile
from pathlib import Path

import numpy as np
import rasterio
from PIL import Image, ImageDraw, ImageFont
from rasterio.enums import Resampling
from rasterio.errors import NotGeoreferencedWarning
from rasterio.warp import transform as warp_transform
from rasterio.windows import Window

warnings.filterwarnings("ignore", category=NotGeoreferencedWarning)

BOX = 720                      # each picture fits in BOX x BOX pixels
GAP = 16
ZOOM = 1000                    # the zooms: ZOOM x ZOOM L1D pixels (2 km at 2 m)
GAMMA = 1.6
ALPHA = 0.6                    # cloud mask colours over the image
BG, INK, SOFT = (22, 22, 24), (238, 238, 238), (170, 170, 170)
CLASS_RGB = {2: (255, 45, 45), 3: (255, 215, 0), 4: (40, 120, 255)}
BOTH, ONLY_A, ONLY_B = (245, 245, 245), (255, 140, 0), (0, 205, 255)
MAGENTA = (255, 0, 255)
UP, DOWN = np.array([255, 60, 60.0]), np.array([60, 140, 255.0])     # B minus A: B higher, A higher
DIFF_FULL = 0.5                # a confidence difference this big shows at full colour
VIRIDIS = np.array([(68, 1, 84), (59, 82, 139), (33, 145, 140), (94, 201, 98), (253, 231, 37)], float)
MODEL = {"01.02": "Clarity"}          # the cloud model of a version, where it is worth naming
NAMES = {1: "clear", 2: "thick", 3: "thin", 4: "shadow"}


def font(size):
    return ImageFont.load_default(size=size)


class L1D:
    """an L1D product: its zip, ID, STAC facts, quicklook and the paths of its rasters"""

    def __init__(self, path):
        path = Path(path)
        if path.is_dir():
            found = sorted(path.glob("IME*_1DSP__RGB3_*.zip"))
            if len(found) != 1:
                sys.exit(f"{path}: {len(found)} L1D zips in it, not one")
            path = found[0]
        self.zip, self.id = path, path.name[:-4]
        with zipfile.ZipFile(path) as z:
            self.stac = json.loads(z.read(f"{self.id}/{self.id}.json"))
            png = sorted(n for n in z.namelist() if re.search(r"/PREVIEW/[^/]+\.QLK\.png$", n))
            self.quicklook = Image.open(io.BytesIO(z.read(png[0]))).convert("RGBA") if png else None
        props = self.stac.get("properties", {})
        self.version = ", ".join(props.get("processing:software", {}).values()) or "?"
        self.model = f"{self.version} {MODEL[self.version]}" if self.version in MODEL else self.version
        self.l1c = props.get("processing:lineage") or self.id
        self.cover = props.get("eo:cloud_cover")
        self.image = self.member("IMG_DATA", ".tif")
        self.clm = self.member("AUX_DATA", ".CLM.tif")
        self.com = self.member("AUX_DATA", ".COM.tif")

    def member(self, folder, suffix):
        return f"/vsizip/{self.zip}/{self.id}/{folder}/{self.id}{suffix}"

    def display_range(self):
        """[[low, high]] per band, from the STAC render range (one range for the three bands, or one each)"""
        for holder in (self.stac.get("properties", {}), self.stac):
            for render in (holder.get("renders") or {}).values():
                if isinstance(render, dict) and render.get("rescale"):
                    r = np.asarray(render["rescale"], float)
                    return np.repeat(r, 3, axis=0)[:3] if len(r) == 1 else r[:3]
        return None


def show(v, rng):
    """L1D values (3, h, w) as RGB (h, w, 3), with the display range and the gamma; no data dark"""
    if rng is None:
        ok = (v > 0).all(0)
        rng = np.array([np.percentile(b[ok], (2, 98)) if ok.any() else (0, 1) for b in v])
    lo, hi = rng[:, 0][:, None, None], rng[:, 1][:, None, None]
    x = np.clip((v.astype("float32") - lo) / np.maximum(hi - lo, 1), 0, 1) ** (1 / GAMMA)
    rgb = np.rint(x * 255).astype("uint8").transpose(1, 2, 0)
    rgb[(v == 0).all(0)] = BG
    return rgb


def overlay(rgb, clm):
    """the cloud mask's classes in colour over the image; clear pixels show the image, a little darker"""
    out = rgb.astype("float32") * 0.85
    for k, colour in CLASS_RGB.items():
        m = clm == k
        out[m] = out[m] * (1 - ALPHA) + np.array(colour, float) * ALPHA
    out[clm == 0] = BG
    return out.astype("uint8")


def agreement(rgb, clm_a, clm_b):
    """white cloud in both, orange only in A, cyan only in B, the image darkened elsewhere"""
    out = np.repeat(rgb.astype("float32").mean(2, keepdims=True) * 0.35, 3, axis=2).astype("uint8")
    ca, cb = np.isin(clm_a, (2, 3)), np.isin(clm_b, (2, 3))
    out[ca & cb], out[ca & ~cb], out[~ca & cb] = BOTH, ONLY_A, ONLY_B
    out[(clm_a == 0) | (clm_b == 0)] = BG
    return out


def ramp(x):
    t = np.clip(x, 0, 1) * (len(VIRIDIS) - 1)
    i = np.minimum(t.astype(int), len(VIRIDIS) - 2)
    f = (t - i)[..., None]
    return (VIRIDIS[i] * (1 - f) + VIRIDIS[i + 1] * f).astype("uint8")


def difference(com_a, com_b, nodata):
    d = (com_b - com_a)[..., None]
    out = np.clip(d / DIFF_FULL, 0, 1) * UP + np.clip(-d / DIFF_FULL, 0, 1) * DOWN
    out = out.astype("uint8")
    out[nodata] = BG
    return out


def changed_pixels(a, b, shape, H, W):
    """where B's L1D image differs from A's: (pixels that differ, of them not white in B, a map of shape that
    is True in every cell holding one), read band by band in strips of rows"""
    cells = np.zeros(shape, bool)
    n = not_white = 0
    rows_to = (np.arange(H) * shape[0]) // H
    cols_to = (np.arange(W) * shape[1]) // W
    with rasterio.open(a.image) as ra, rasterio.open(b.image) as rb:
        for r0 in range(0, H, 1024):
            win = Window(0, r0, W, min(1024, H - r0))
            va, vb = ra.read(window=win), rb.read(window=win)
            differ = (va != vb).any(0)
            if differ.any():
                rr, cc = np.nonzero(differ)
                n += rr.size
                band = vb[:, rr, cc]
                not_white += int(((band[0] != band[1]) | (band[1] != band[2])).sum())
                cells[rows_to[rr + r0], cols_to[cc]] = True
    return n, not_white, cells


def busiest(mask, size):
    """top-left (row, col) of the size x size window holding most of mask, and how many it holds"""
    H, W = mask.shape
    size = min(size, H, W)
    step = max(1, size // 10)
    h, w = H // step, W // step
    blocks = mask[: h * step, : w * step].reshape(h, step, w, step).sum((1, 3))
    k = size // step
    c = np.pad(blocks.cumsum(0).cumsum(1), ((1, 0), (1, 0)))
    sums = c[k:, k:] - c[:-k, k:] - c[k:, :-k] + c[:-k, :-k]
    i, j = np.unravel_index(np.argmax(sums), sums.shape)
    return (int(i * step), int(j * step)), int(sums[i, j]), size


def fit(img):
    """an RGB array or a PIL image, shrunk to fit BOX x BOX"""
    im = img if isinstance(img, Image.Image) else Image.fromarray(img)
    if im.width > BOX or im.height > BOX:
        im = im.copy()
        im.thumbnail((BOX, BOX), Image.LANCZOS)
    return im


def quicklook(product):
    if product.quicklook is None:
        return Image.new("RGB", (BOX, BOX // 2), BG)
    q = product.quicklook
    s = min(BOX / q.width, BOX / q.height)            # to the size of the other pictures, up or down
    q = q.resize((max(1, round(q.width * s)), max(1, round(q.height * s))), Image.LANCZOS)
    out = Image.new("RGB", q.size, BG)
    out.paste(q, mask=q.split()[3])
    return out


def latlon(transform, crs, row, col):
    x, y = transform * (col, row)
    lon, lat = warp_transform(crs, "EPSG:4326", [x], [y])
    return f"{lat[0]:.4f} {'N' if lat[0] >= 0 else 'S'}, {lon[0]:.4f} {'E' if lon[0] >= 0 else 'W'}"


def shares(clm):
    v = clm[clm > 0]
    return {k: float(np.mean(v == k)) if v.size else float("nan") for k in (1, 2, 3, 4)}


def panel(a, b, out_dir):
    with rasterio.open(b.image) as src:
        H, W, transform, crs = src.height, src.width, src.transform, src.crs
        scale = min(BOX / W, BOX / H)
        shape = (max(1, round(H * scale)), max(1, round(W * scale)))
        img = src.read(out_shape=(3, *shape), resampling=Resampling.average)
    rng = b.display_range()
    rgb = show(img, rng)

    def small(path, how):
        with rasterio.open(path) as src:
            return src.read(1, out_shape=shape, resampling=how)

    clm_a_s, clm_b_s = small(a.clm, Resampling.mode), small(b.clm, Resampling.mode)
    com_a_s, com_b_s = (small(p, Resampling.average) / 1e4 for p in (a.com, b.com))
    nodata_s = (clm_a_s == 0) | (clm_b_s == 0)

    # the whole cloud masks, at full resolution: shares, agreement, where they differ the most
    with rasterio.open(a.clm) as sa, rasterio.open(b.clm) as sb:
        if (sa.height, sa.width) != (sb.height, sb.width):
            sys.exit(f"{a.id} and {b.id}: cloud masks of different sizes")
        clm_a, clm_b = sa.read(1), sb.read(1)
    valid = (clm_a > 0) & (clm_b > 0)
    ca, cb = np.isin(clm_a, (2, 3)) & valid, np.isin(clm_b, (2, 3)) & valid
    nv = max(int(valid.sum()), 1)
    same_class = float(np.sum((clm_a == clm_b) & valid)) / nv
    cloud_same = float(np.sum((ca == cb) & valid)) / nv
    both = int(np.sum(ca & cb))
    sh_a, sh_b = shares(clm_a), shares(clm_b)
    with rasterio.open(a.com) as sa, rasterio.open(b.com) as sb:
        sub = (max(1, H // 4), max(1, W // 4))
        ma = sa.read(1, out_shape=sub, resampling=Resampling.nearest) / 1e4
        mb = sb.read(1, out_shape=sub, resampling=Resampling.nearest) / 1e4
        with rasterio.open(a.clm) as s:
            vsub = s.read(1, out_shape=sub, resampling=Resampling.nearest) > 0
    ma, mb = ma[vsub], mb[vsub]
    corr = float(np.corrcoef(ma, mb)[0, 1]) if ma.size > 1 and ma.std() > 0 and mb.std() > 0 else float("nan")
    zooms = [busiest(ca & ~cb, ZOOM), busiest(~ca & cb & valid, ZOOM)]
    n_changed, not_white, cells = changed_pixels(a, b, shape, H, W)

    # the pictures
    grey = np.repeat(rgb.astype("float32").mean(2, keepdims=True) * 0.5, 3, axis=2).astype("uint8")
    grey[(img == 0).all(0)] = BG
    change = grey.copy()
    change[cells] = MAGENTA
    agree = Image.fromarray(agreement(rgb, clm_a_s, clm_b_s))
    draw = ImageDraw.Draw(agree)
    for k, ((r0, c0), count, size) in enumerate(zooms, 1):
        if count:
            box = [c0 * scale, r0 * scale, (c0 + size) * scale, (r0 + size) * scale]
            draw.rectangle(box, outline=(255, 255, 255), width=2)
            draw.text((box[2] + 4, box[1]), str(k), fill=(255, 255, 255), font=font(22))
    com_a_rgb, com_b_rgb = ramp(com_a_s), ramp(com_b_s)
    com_a_rgb[nodata_s] = BG
    com_b_rgb[nodata_s] = BG

    def pct(x):
        return f"{100 * x:.1f}%"

    def cover(p, sh):
        return pct(sh[2] + sh[3]) if p.cover is None else f"{p.cover:.1f}%"

    rows = [
        ("Whole scene: the quicklooks, as in each product",
         [(quicklook(a), f"A {a.version}"), (quicklook(b), f"B {b.version}"),
          (fit(change), (f"L1D image: {n_changed:,} pixels differ"
                         + (f", {not_white:,} not white in B" if not_white else
                            (", all white in B (saturated in the L1C)" if n_changed else ""))))]),
        ("Cloud mask (CLM) over the image: thick red, thin yellow, shadow blue",
         [(fit(overlay(rgb, clm_a_s)), f"A {a.model}: cloud {cover(a, sh_a)}"),
          (fit(overlay(rgb, clm_b_s)), f"B {b.model}: cloud {cover(b, sh_b)}"),
          (fit(agree), "cloud in both white, only A orange, only B cyan")]),
        ("Cloud confidence (COM): 0 purple to 1 yellow",
         [(fit(com_a_rgb), f"A {a.model}"), (fit(com_b_rgb), f"B {b.model}"),
          (fit(difference(com_a_s, com_b_s, nodata_s)),
           "B minus A: red B higher, blue A higher" + (" (identical)" if np.array_equal(com_a_s, com_b_s) else ""))]),
    ]
    with rasterio.open(b.image) as src:
        for k, ((r0, c0), count, size) in enumerate(zooms, 1):
            who = "A" if k == 1 else "B"
            if not count:
                rows.append((f"{k}: nowhere is there cloud only in {who}", []))
                continue
            v = src.read(window=Window(c0, r0, size, size))
            z = show(v, rng)
            za, zb = clm_a[r0:r0 + size, c0:c0 + size], clm_b[r0:r0 + size, c0:c0 + size]
            km = size * abs(transform.a) / 1000
            rows.append((f"{k}: {km:.0f} km at full resolution where only {who} sees cloud the most "
                         f"(centre {latlon(transform, crs, r0 + size / 2, c0 + size / 2)})",
                         [(fit(z), "L1D image (B)"), (fit(overlay(z, za)), f"A {a.model}"),
                          (fit(overlay(z, zb)), f"B {b.model}")]))

    # the page
    width = 3 * BOX + 4 * GAP
    head = [
        (font(30), INK, f"{b.l1c}"),
        (font(21), SOFT, f"A  {a.id}  (processor {a.model})"),
        (font(21), SOFT, f"B  {b.id}  (processor {b.model})"),
        (font(22), INK, f"Valid pixels: same class {pct(same_class)}; cloud or not the same {pct(cloud_same)}; "
                        f"cloud in both {pct(both / max(int(ca.sum()), 1))} of A's cloud and "
                        f"{pct(both / max(int(cb.sum()), 1))} of B's; COM correlation {corr:.3f}"),
        (font(22), INK, "A: " + " · ".join(f"{NAMES[k]} {pct(sh_a[k])}" for k in (1, 2, 3, 4)) + "      B: "
                        + " · ".join(f"{NAMES[k]} {pct(sh_b[k])}" for k in (1, 2, 3, 4))),
    ]
    heights = [sum(f.getbbox("Ag")[3] + 10 for f, _, _ in head) + GAP]
    for title, cells_ in rows:
        heights.append(34 + 28 + max([im.height for im, _ in cells_] or [0]) + GAP)
    page = Image.new("RGB", (width, sum(heights) + GAP), BG)
    d = ImageDraw.Draw(page)
    y = GAP
    for f, colour, text in head:
        d.text((GAP, y), text, fill=colour, font=f)
        y += f.getbbox("Ag")[3] + 10
    y += GAP
    for (title, cells_), h in zip(rows, heights[1:]):
        d.text((GAP, y), title, fill=INK, font=font(24))
        for i, (im, caption) in enumerate(cells_):
            x = GAP + i * (BOX + GAP)
            d.text((x, y + 34), caption, fill=SOFT, font=font(20))
            page.paste(im, (x + (BOX - im.width) // 2, y + 62))
        y += h
    out_dir.mkdir(parents=True, exist_ok=True)
    path, k = out_dir / f"{b.l1c}.jpg", 1
    while path.exists():                               # the same L1C twice: keep both pictures
        k += 1
        path = out_dir / f"{b.l1c}_{k}.jpg"
    page.save(path, quality=90, subsampling=0)
    print(f"{b.l1c}: cloud A {cover(a, sh_a)}, B {cover(b, sh_b)}; same class {pct(same_class)}; cloud or not "
          f"{pct(cloud_same)}; COM correlation {corr:.3f}; L1D pixels changed {n_changed:,}"
          f"{f' ({not_white:,} not white in B)' if not_white else ''} -> {path}")


def main(argv):
    if len(argv) < 3 or len(argv) % 2 == 0:
        sys.exit(__doc__)
    out_dir = Path(argv[0])
    failed = 0
    for pa, pb in zip(argv[1::2], argv[2::2]):
        try:                                           # one product that cannot be read does not stop the rest
            a, b = L1D(pa), L1D(pb)
            if a.l1c != b.l1c:
                print(f"warning: {a.id} and {b.id} come from different L1Cs ({a.l1c}, {b.l1c})")
            panel(a, b, out_dir)
        except (Exception, SystemExit) as e:
            failed += 1
            print(f"skipped {pa} and {pb}: {e}")
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
