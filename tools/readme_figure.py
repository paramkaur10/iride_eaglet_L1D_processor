#!/usr/bin/env python3
"""
readme_figure.py: the pictures of the README. Each L1D product in one row:
    L1C as delivered | Sentinel-2 L1C | L1D | cloud mask (CLM) | cloud confidence (COM)
Every image is drawn like the L1D quicklook (each band from its own dark level, one span for the three bands, then a
gamma of 1.6; the drawing of tools/l1d_panels.py), inside the L1D footprint.

Sentinel-2 L1C of the same place comes from a LUT build on disk (tools/build_lut_v4.py and build_lut.py keep, for
each L1C, <work>/s2/<L1C>_composite.tif and the passes it was made of), found by satellite and sensing time under
the --s2 folders: the pass that gave the composite the most, whole, or else the composite. With --fetch, a product
without one gets the clearest Sentinel-2 L1C pass within 6 days from the Copernicus Data Space instead
(CDSE_CLIENT_ID and CDSE_CLIENT_SECRET, or asked for, hidden; never written to disk), kept in --cache
(dist/s2_cache, not in git) for the next run.

    python tools/readme_figure.py <L1D zips or folders> [--l1c <L1C zips or folders>] [--s2 <folders>] [--fetch]
                                  [--out docs/images]

The L1C of each L1D is found by satellite and sensing time among the --l1c items and the L1D folders (a working
directory of docker/test_image.sh holds both). Writes <out>/l1d_<satellite>_<sensing start>.jpg. Needs numpy,
rasterio, scipy and Pillow, as in the processor's image; --fetch also needs requests (in the tools image,
tools/docker, or installed for the run). From the repository folder, for example:
    docker run --rm -it --network host -u 0 -e PYTHONDONTWRITEBYTECODE=1 -v "$PWD:/repo" \
        -v <working directories>:/runs:ro -v <LUT work folders>:/s2:ro ipf-ime-msi-l1d:01.02 \
        sh -c "pip install -q requests && python /repo/tools/readme_figure.py /runs/*/WD_*/ --s2 /s2 --fetch"
"""
import argparse
import getpass
import json
import os
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path

import numpy as np
import rasterio
from PIL import Image, ImageDraw, ImageFont
from rasterio.enums import Resampling
from rasterio.warp import reproject
from rasterio.warp import transform as warp_transform

sys.path.insert(0, str(Path(__file__).resolve().parent))
import l1d_panels as lp  # noqa: E402   the poster's reading and drawing: the same stretch and colours

REPO = Path(__file__).resolve().parents[1]
BOX, GAP, PAD = 520, 22, 30                   # pixels: the side of a cell, between cells, around the picture
WHITE, INK, SOFT = (255, 255, 255), (11, 11, 11), (82, 81, 78)
FETCH_RES = 20.0                              # metres: Sentinel-2 downloaded with --fetch
FETCH_DAYS = 6                                # ... the clearest pass up to 6 days from the Eaglet-2 image
SCORE_PER_DAY = 0.03                          # ... a pass one day further away needs 3% more of its box clear


def font(size):
    return ImageFont.load_default(size=size)


def to_image(rgb):
    return Image.fromarray((np.clip(rgb, 0, 1) * 255 + 0.5).astype("uint8"))


def when(start):
    """the sensing start of a product name (YYYYMMDDTHHMMSS) as a UTC time"""
    return datetime.strptime(start, "%Y%m%dT%H%M%S").replace(tzinfo=timezone.utc)


# ---- Sentinel-2 ---------------------------------------------------------------------------------------------------

def s2_on_disk(roots, key):
    """Sentinel-2 of the acquisition key = (satellite, start, stop) in the LUT builds under roots, as
    (path, {"passes": [...]}), or (None, None). The composite of the build with the most of the footprint clear; of it,
    the pass that gave the most (the whole pass, its clouds included, so the picture has no holes) when it is still on
    disk, else the composite itself (white where every pass was cloudy)."""
    found = []
    for root in roots:
        for comp in Path(root).rglob("*_composite.tif"):
            stem = comp.name[:-len("_composite.tif")]
            m = lp.NAME.match(stem)
            if comp.parent.name != "s2" or not m or (m.group(1), m.group(3), m.group(4)) != key:
                continue
            info = comp.with_name(stem + ".json")
            meta = json.loads(info.read_text()) if info.exists() else {}
            found.append((meta.get("filled", 0.0), str(comp), stem, meta))
    if not found:
        return None, None
    _, comp, stem, meta = max(found, key=lambda f: (f[0], f[1]))
    passes = [q for q in meta.get("passes", []) if q.get("day")]
    if passes:
        main = max(passes, key=lambda q: q.get("added", 0.0))
        whole = Path(comp).parent / stem / f"{main['day']}.tif"
        if whole.exists():
            return whole, dict(passes=[main])
    return Path(comp), dict(meta, composite=True)


_CDSE = None


def cdse():
    """the Copernicus Data Space client of tools/build_lut_v4.py; the client id and secret are asked for, hidden,
    when they are not in the environment, and kept in memory only"""
    global _CDSE
    if _CDSE is None:
        import build_lut_v4 as bv4
        for var, what in (("CDSE_CLIENT_ID", "client id"), ("CDSE_CLIENT_SECRET", "client secret")):
            if not os.environ.get(var):
                os.environ[var] = getpass.getpass(f"CDSE {what} (hidden): ").strip()
        _CDSE = bv4.CDSE()
    return _CDSE


def s2_fetch(key, g, cache):
    """the clearest Sentinel-2 L1C pass over the L1D within FETCH_DAYS days, downloaded once into cache as
    reflectance R, G, B (float32, NaN no data): (path, {"passes": [...]}), or (None, None)"""
    import build_lut_v4 as bv4
    sat, start, _ = key
    cache.mkdir(parents=True, exist_ok=True)
    done = sorted(cache.glob(f"{sat}_{start}_*.json"))
    if done:
        meta = json.loads(done[-1].read_text())
        return done[-1].with_suffix(".tif"), meta
    H, W = g["full"]
    tf = g["transform"]
    xs, ys = (tf.c, tf.c + W * tf.a), (tf.f, tf.f + H * tf.e)
    epsg = g["crs"].to_epsg()
    (left, bottom, right, top), w, h, s2_tf = bv4.grid_of((min(xs), min(ys), max(xs), max(ys)), FETCH_RES)
    lon, lat = warp_transform(g["crs"], "EPSG:4326", [left, right, right, left], [bottom, bottom, top, top])
    t_e = when(start)
    best = None
    for t, _ in cdse().days((min(lon), min(lat), max(lon), max(lat)), t_e - timedelta(days=FETCH_DAYS),
                            t_e + timedelta(days=FETCH_DAYS)):
        day = t.date().isoformat()
        scl, data = cdse().fetch("sentinel-2-l2a", bv4.EVAL_SCL, (left, bottom, right, top), epsg, 120.0, day, 2,
                                 "uint8")
        ok = data > 0
        if ok.mean() < 0.5:                       # the pass covers less than half of the box
            continue
        clear = float((ok & ~np.isin(scl, bv4.CLOUDY_SCL)).mean())
        dt = (t - t_e).total_seconds() / 86400
        score = clear - SCORE_PER_DAY * abs(dt)
        if best is None or score > best[0]:
            best = (score, day, clear, dt)
    if best is None:
        return None, None
    _, day, clear, dt = best
    dn = cdse().fetch("sentinel-2-l1c", bv4.EVAL_L1C, (left, bottom, right, top), epsg, FETCH_RES, day, 5, "uint16")
    refl = np.where(dn[4] > 0, dn[:3] / 10000.0, np.nan).astype("float32")
    path = cache / f"{sat}_{start}_{day}.tif"
    with rasterio.open(path, "w", driver="GTiff", width=w, height=h, count=3, dtype="float32", crs=f"EPSG:{epsg}",
                       transform=s2_tf, nodata=float("nan"), compress="deflate") as dst:
        dst.write(refl)
    meta = dict(passes=[dict(day=day, dt_days=round(dt, 2), clear=round(clear, 3))], source="Copernicus Data Space")
    path.with_suffix(".json").write_text(json.dumps(meta, indent=1))
    return path, meta


def s2_on_grid(path, g):
    """Sentinel-2 reflectance R, G, B of path averaged onto the drawn grid of the L1D. path: a composite or a
    download (float32 reflectance, NaN no data) or one pass of a LUT build (uint16 DN = reflectance x 10000, 0 no
    data)"""
    h, w = g["shape"]
    out = np.full((3, h, w), np.nan, "float32")
    with rasterio.open(path) as src:
        for i in range(3):
            band = src.read(i + 1).astype("float32")
            if src.dtypes[i].startswith("uint"):
                band = np.where(band > 0, band / 10000.0, np.nan).astype("float32")
            reproject(band, out[i], src_transform=src.transform, src_crs=src.crs, src_nodata=np.nan,
                      dst_transform=g["draw"], dst_crs=g["crs"], dst_nodata=np.nan, resampling=Resampling.average)
    return out


def s2_caption(meta):
    days = sorted(datetime.strptime(q["day"], "%Y-%m-%d") for q in (meta or {}).get("passes", []) if q.get("day"))
    if not days:
        return "Sentinel-2 L1C"
    if days[0] == days[-1]:
        return f"Sentinel-2 L1C, {days[0]:%d %b %Y}"
    return f"Sentinel-2 L1C, clear parts, {days[0]:%d %b} to {days[-1]:%d %b %Y}"


# ---- the picture --------------------------------------------------------------------------------------------------

def ramp(values, valid):
    """the cloud confidence in the poster's blue ramp (0 light, 1 dark), white outside the footprint"""
    cols = np.array([[int(c[i:i + 2], 16) for i in (1, 3, 5)] for c in lp.COM_RAMP], "float32") / 255
    x = np.clip(np.nan_to_num(values), 0, 1) * (len(cols) - 1)
    i = np.clip(np.floor(x).astype(int), 0, len(cols) - 2)
    f = (x - i)[..., None]
    rgb = cols[i] * (1 - f) + cols[i + 1] * f
    rgb[~valid] = lp.NODATA_RGB
    return rgb


def row(p, s2, s2_text):
    """the five images of one product, with their captions"""
    cover = p["stac"]["properties"].get("eo:cloud_cover")
    cells = [(to_image(p["l1c"]["rgb"]) if p["l1c"] else None, "L1C, as delivered")]
    if s2 is not None:
        ok = np.isfinite(s2).all(0) & p["valid"]
        cells.append((to_image(lp.to_rgb(s2, ok, lp.stretch_limits(s2, ok))), s2_text))
    else:
        cells.append((None, "Sentinel-2 L1C: none found"))
    cells += [(to_image(p["rgb_d"]), f"L1D, processor {p['version']}"),
              (to_image(lp.clm_overlay(p)), "Cloud mask" + (f": {cover:.1f}% cloud" if cover is not None else "")),
              (to_image(ramp(p["com"], p["valid"])), "Cloud confidence")]
    return cells


def legend_clm(d, x, y):
    for v, name, col in ((1, "clear", "#9a9a9a"),) + lp.CLM_CLASSES:
        d.rectangle([x, y + 3, x + 16, y + 19], fill=col)
        d.text((x + 22, y), name, fill=SOFT, font=font(17))
        x += 30 + int(d.textlength(name, font=font(17)))


def legend_com(page, d, x, y, width=220):
    bar = ramp(np.tile(np.linspace(0, 1, width), (14, 1)), np.ones((14, width), bool))
    d.text((x, y), "0", fill=SOFT, font=font(17))
    page.paste(to_image(bar), (x + 18, y + 4))
    d.text((x + 26 + width, y), "1 (cloud)", fill=SOFT, font=font(17))


def figure(p, cells, out):
    sat, start = lp.NAME.match(p["id"]).group(1), lp.NAME.match(p["id"]).group(3)
    t = when(start)
    width = 5 * BOX + 4 * GAP + 2 * PAD
    top = PAD + 38 + 30 + 18
    height = top + 30 + BOX + 16 + 24 + PAD
    page = Image.new("RGB", (width, height), WHITE)
    d = ImageDraw.Draw(page)
    d.text((PAD, PAD), f"Eaglet-2 {sat}, {t:%d %b %Y}, {t:%H:%M} UTC", fill=INK, font=font(30))
    d.text((PAD, PAD + 38), p["id"], fill=SOFT, font=font(19))
    for i, (im, caption) in enumerate(cells):
        x = PAD + i * (BOX + GAP)
        d.text((x, top), caption, fill=INK, font=font(21))
        if im is None:
            d.rectangle([x, top + 30, x + BOX - 1, top + 30 + BOX - 1], outline=(200, 200, 200))
            continue
        page.paste(im, (x + (BOX - im.width) // 2, top + 30 + (BOX - im.height) // 2))
    legend_clm(d, PAD + 3 * (BOX + GAP), top + 30 + BOX + 12)
    legend_com(page, d, PAD + 4 * (BOX + GAP), top + 30 + BOX + 12)
    out.parent.mkdir(parents=True, exist_ok=True)
    page.save(out, quality=88, subsampling=0)


def main(argv=None):
    global BOX
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("l1d", nargs="+", help="L1D zips, or folders holding them")
    ap.add_argument("--l1c", nargs="*", default=[], help="L1C zips, or folders holding them")
    ap.add_argument("--s2", nargs="*", default=[], help="folders with LUT builds (their s2/ composites)")
    ap.add_argument("--fetch", action="store_true", help="download Sentinel-2 when none is on disk")
    ap.add_argument("--out", type=Path, default=REPO / "docs" / "images", help="default: %(default)s")
    ap.add_argument("--cache", type=Path, default=REPO / "dist" / "s2_cache",
                    help="Sentinel-2 downloaded with --fetch (default: %(default)s, not in git)")
    ap.add_argument("--box", type=int, default=BOX, help="side of each image in pixels (default: %(default)s)")
    a = ap.parse_args(argv)
    BOX = a.box
    l1ds, l1cs = lp.find_products(a.l1d, "D"), lp.find_products(a.l1c + a.l1d, "C")
    if not l1ds:
        sys.exit("no L1D product found")
    for key in sorted(l1ds, key=lambda k: (k[0], k[1])):
        p = lp.load(l1ds[key], l1cs.get(key), BOX)
        comp, meta = s2_on_disk(a.s2, key)
        source = f"on disk, {comp}" if comp else "none on disk"
        if comp is None and a.fetch:
            comp, meta = s2_fetch(key, p["g"], a.cache)
            source = f"Copernicus Data Space, {comp.name}" if comp else f"none found within {FETCH_DAYS} days"
        s2 = s2_on_grid(comp, p["g"]) if comp else None
        out = a.out / f"l1d_{key[0]}_{key[1]}.jpg"
        figure(p, row(p, s2, s2_caption(meta)), out)
        print(f"{key[0]} {key[1]}: L1C {'found' if p['l1c'] else 'NOT FOUND'}; Sentinel-2 {source} -> {out}")


if __name__ == "__main__":
    main()
