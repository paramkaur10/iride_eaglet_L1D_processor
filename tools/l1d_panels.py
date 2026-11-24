#!/usr/bin/env python3
"""
l1d_panels.py: poster panels of L1D products made by the processor.

For each L1D product, one panel:
    (a) the L1C as delivered   (b) the L1D, colour corrected   (c) the cloud mask (CLM)   (d) the cloud confidence (COM)
    (e) where the scene is      (f) the product: flight model, sensing, orbit, sun, AOI, image, cloud cover, L1C and L1D
                                    names, processing, colour AUX
With --zoom, the second row starts with a part of the L1C and of the L1D at full resolution (--zoom-m wide, 2 km
by default): (e) L1C, (f) L1D, (g) where, (h) the product.
Every map has latitude and longitude on its axes, a scale bar and north. Plus one overview with a row per product,
and the scene images alone in tiles/ for the poster. PNG (200 dpi) and PDF of each.

Colours as in the L1D quicklook from processor 01.02 on: each band from its own dark level (its 2nd percentile: the
haze, largest in blue), one span for the three bands (the 98th percentile of the three after it), then a gamma of 1.6.
The L1C is drawn the same way (its saturated pixels left out of the percentiles), so what differs between (a) and (b)
is the colour correction alone.

    python tools/l1d_panels.py <L1D zips or folders> --l1c <L1C zips or folders> [--out DIR] [--zoom] [--offline]

The L1C of each L1D is found by satellite and sensing time. The place name of each scene (AOI) comes from
OpenStreetMap (Nominatim) and the country outlines of the map from Natural Earth, downloaded once; --offline skips
both. The names are kept in <out>/aoi_names.json: edit them there and run again to change what the panels say.
"""

import argparse
import json
import math
import re
import sys
import tempfile
import time
import urllib.request
import zipfile
from datetime import datetime
from pathlib import Path

import numpy as np
import rasterio
from rasterio.enums import Resampling
from rasterio.transform import Affine
from rasterio.warp import transform as warp_transform
from rasterio.windows import Window
from scipy import ndimage

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))
from iride_eaglet_processor.calibration import read_xml_calibration  # noqa: E402

NAME = re.compile(r"^(IME\d{2})_1([CD])SP__RGB3_(\d{8}T\d{6})_(\d{8}T\d{6})_(\d{8}T\d{6})_(\d{5})_(R\d{3})_"
                  r"([A-Z])_([A-Z0-9]{3})$")
INK, INK2, NODATA_RGB = "#0b0b0b", "#52514e", (1.0, 1.0, 1.0)
CLM_CLASSES = ((2, "thick cloud", "#e34948"), (3, "thin cloud", "#eda100"), (4, "cloud shadow", "#2a78d6"))
COM_RAMP = ("#cde2fb", "#b7d3f6", "#9ec5f4", "#86b6ef", "#6da7ec", "#5598e7", "#3987e5", "#2a78d6", "#256abf",
            "#1c5cab", "#184f95", "#104281", "#0d366b")
LAND, LAND_EDGE, SEA, MARK = "#e6e4df", "#9a9893", "#f2f6fa", "#e34948"
COUNTRIES_URL = ("https://raw.githubusercontent.com/nvkelso/natural-earth-vector/master/geojson/"
                 "ne_50m_admin_0_countries.geojson")


# ---- products -----------------------------------------------------------------------------------------------------

def find_products(items, level):
    """{(satellite, sensing start, sensing stop): zip} of the products of level C or D among items (zips, folders);
    the newest generation when there are several"""
    out = {}
    for item in items:
        p = Path(item).expanduser()
        for z in (sorted(p.glob("*.zip")) if p.is_dir() else [p]):
            m = NAME.match(z.stem)
            if m and m.group(2) == level:
                key = (m.group(1), m.group(3), m.group(4))
                if key not in out or z.stem > out[key].stem:
                    out[key] = z
    return out


class Product:
    """a product zip, read in place through GDAL (/vsizip/)"""

    def __init__(self, path):
        self.path = Path(path).resolve()
        self.id = self.path.stem
        with zipfile.ZipFile(self.path) as zf:
            self.names = [n for n in zf.namelist() if not n.endswith("/")]

    def member(self, test, what):
        hits = [n for n in self.names if test(n)]
        if len(hits) != 1:
            raise ValueError(f"{self.path.name}: {len(hits)} {what}")
        return hits[0]

    def file(self, basename, folder=None):
        return self.member(lambda n: n.rsplit("/", 1)[-1] == basename and (folder is None or f"{folder}/" in n),
                           basename)

    def image(self):
        return self.member(lambda n: "IMG_DATA/" in n and n.lower().endswith((".tif", ".tiff")), "image in IMG_DATA")

    def gdal(self, member):
        return f"/vsizip/{self.path}/{member}"

    def read(self, member):
        with zipfile.ZipFile(self.path) as zf:
            return zf.read(member)


def local(tag):
    return tag.rsplit("}", 1)[-1]


def xml_values(data):
    """{element name: text} of the leaves of an XML (first occurrence), namespaces dropped"""
    import xml.etree.ElementTree as ET
    out = {}
    for el in ET.fromstring(data).iter():
        if len(el) == 0 and el.text and el.text.strip():
            out.setdefault(local(el.tag), el.text.strip())
    return out


def calibration(l1c):
    """(gain, offset) per band from the L1C XML, read with the processor's own code"""
    with tempfile.NamedTemporaryFile(suffix=".xml") as tmp:
        tmp.write(l1c.read(l1c.file(f"{l1c.id}.xml")))
        tmp.flush()
        cal = read_xml_calibration(tmp.name)
    return np.array([cal[b][0] for b in (1, 2, 3)]), np.array([cal[b][1] for b in (1, 2, 3)])


# ---- reading ------------------------------------------------------------------------------------------------------

def read_scaled(path, shape, resampling, bands=None):
    with rasterio.open(path) as src:
        idx = bands or list(range(1, src.count + 1))
        return src.read(idx, out_shape=(len(idx),) + shape, resampling=resampling)


def grid(path, max_px):
    """the full raster's size and georeference, and the shape and transform of the copy that is drawn"""
    with rasterio.open(path) as src:
        f = max(1.0, max(src.width, src.height) / max_px)
        h, w = max(1, round(src.height / f)), max(1, round(src.width / f))
        return dict(full=(src.height, src.width), shape=(h, w), crs=src.crs, transform=src.transform,
                    draw=src.transform * Affine.scale(src.width / w, src.height / h), tags=src.tags())


def valid_of(dn_nearest):
    """valid where no band is 0, one drawn pixel in from the edge of the data (averaged edge pixels are darker)"""
    return ndimage.binary_erosion((dn_nearest > 0).all(0), iterations=1)


STRETCH = (2, 98)          # dark level of each band, span of the three (percentiles), then GAMMA: as the quicklook
GAMMA = 1.6
SATURATED_DN = 65535       # an L1C band at the top of the uint16 range is clipped


def stretch_limits(refl, use):
    """(low, high), arrays of 3, over the pixels use: each band from its dark level, its 2nd percentile, and one
    span for the three bands, the 98th percentile of the three after their dark levels (as the L1D quicklook)"""
    vals = refl[:, use]
    if vals.shape[1] < 100:
        return np.zeros(3), np.full(3, 0.3)
    lo = np.percentile(vals, STRETCH[0], axis=1)
    span = float(np.percentile((vals - lo[:, None]).ravel(), STRETCH[1]))
    return lo, lo + max(span, 1e-6)


def to_rgb(refl, valid, limits):
    lo, hi = limits
    rgb = np.clip((np.moveaxis(refl, 0, -1) - lo) / (hi - lo), 0, 1) ** (1 / GAMMA)
    rgb[~valid] = NODATA_RGB
    return rgb


# ---- maps ---------------------------------------------------------------------------------------------------------

def nice_step(span, n=5):
    for s in (0.0005, 0.001, 0.002, 0.005, 0.01, 0.02, 0.05, 0.1, 0.2, 0.5, 1, 2, 5, 10, 20, 30):
        if span / s <= n:
            return s
    return 45


def deg(v, step, pos, neg):
    d = max(0, -int(math.floor(math.log10(step) + 1e-9)))
    return f"{abs(v):.{d}f}°{pos if v >= 0 else neg}"


def geo_axes(ax, tr, crs, shape, fs):
    """latitude and longitude where the parallels and meridians cross the left and bottom edges of a drawn image"""
    h, w = shape
    cols = np.linspace(0, w, 300)
    lon = np.array(warp_transform(crs, "EPSG:4326", *tr * (cols, np.full_like(cols, h)))[0])
    rows = np.linspace(0, h, 300)
    lat = np.array(warp_transform(crs, "EPSG:4326", *tr * (np.zeros_like(rows), rows))[1])
    for values, along, setter, labels, pos, neg in ((lon, cols, ax.set_xticks, ax.set_xticklabels, "E", "W"),
                                                    (lat, rows, ax.set_yticks, ax.set_yticklabels, "N", "S")):
        step = nice_step(values.max() - values.min())
        ticks = np.arange(math.ceil(values.min() / step - 1e-9) * step, values.max() + 1e-12, step)
        order = np.argsort(values)
        setter(np.interp(ticks, values[order], along[order]))
        labels([deg(t, step, pos, neg) for t in ticks], fontsize=fs)
    ax.set_xlabel("Longitude", fontsize=fs + 1, color=INK2)
    ax.set_ylabel("Latitude", fontsize=fs + 1, color=INK2)
    ax.tick_params(length=3, width=0.6, colors=INK2)
    ax.set_xlim(0, w)
    ax.set_ylim(h, 0)
    for s in ax.spines.values():
        s.set_linewidth(0.6)
        s.set_color(INK2)


def halo(width=2.5):
    from matplotlib import patheffects
    return [patheffects.withStroke(linewidth=width, foreground="white")]


def scale_bar(ax, px_m, shape, fs):
    from matplotlib.patches import Rectangle
    h, w = shape
    target = w * px_m * 0.25
    length = max([v for v in (50, 100, 200, 250, 500, 1000, 2000, 2500, 5000, 10000, 20000, 50000) if v <= target]
                 or [50])
    n = length / px_m
    x0, y0, t = w * 0.96 - n, h * 0.95, max(h, w) * 0.012
    ax.add_patch(Rectangle((x0, y0 - t), n, t, facecolor=INK, edgecolor="white", linewidth=0.8, zorder=5))
    ax.text(x0 + n / 2, y0 - t * 1.6, f"{length / 1000:g} km" if length >= 1000 else f"{length:g} m",
            ha="center", va="bottom", fontsize=fs, color=INK, path_effects=halo(), zorder=6)


def north_arrow(ax, tr, crs, shape, fs):
    h, w = shape
    cx, cy = tr * (w / 2, h / 2)
    lon, lat = warp_transform(crs, "EPSG:4326", [cx], [cy])
    nx, ny = warp_transform("EPSG:4326", crs, lon, [lat[0] + 0.01])
    inv = ~tr
    c0, r0 = inv * (cx, cy)
    c1, r1 = inv * (nx[0], ny[0])
    d = np.array([c1 - c0, r1 - r0])
    d /= np.hypot(*d)
    size = max(h, w) * 0.07
    base = np.array([w * 0.94, h * 0.06 + size])
    tip = base + d * size
    ax.annotate("", xy=tip, xytext=base, zorder=6,
                arrowprops=dict(arrowstyle="-|>,head_width=0.35,head_length=0.7", color=INK, lw=1.4,
                                path_effects=halo(3)))
    ax.text(*(tip + d * size * 0.18), "N", ha="center", va="bottom" if d[1] < 0 else "top", fontsize=fs + 1,
            fontweight="bold", color=INK, path_effects=halo(), zorder=6)


def map_panel(ax, img, g, title, fs, extras=True, **kw):
    ax.imshow(img, extent=(0, g["shape"][1], g["shape"][0], 0), interpolation="antialiased", **kw)
    ax.set_title(title, loc="left", fontsize=fs + 3, color=INK, pad=6)
    geo_axes(ax, g["draw"], g["crs"], g["shape"], fs)
    if extras:
        scale_bar(ax, abs(g["draw"].a), g["shape"], fs)
        north_arrow(ax, g["draw"], g["crs"], g["shape"], fs)


# ---- the zoom (--zoom) -------------------------------------------------------------------------------------------

def pick_zoom(rgb_refl, valid, clm, px_m, side_m):
    """the part to show at full resolution, side_m wide and as tall as the scene's shape allows: all data, no cloud,
    as much vegetation and detail as possible. Returns (row, col, height, width) in drawn pixels"""
    h, w = valid.shape
    zw = max(8, min(int(round(side_m / px_m)), w // 2))
    zh = max(8, min(int(round(zw * h / w)), h // 2))
    step = max(1, min(zh, zw) // 3)
    lum = rgb_refl.mean(0)
    veg = (rgb_refl[1] > rgb_refl[0]) & (rgb_refl[1] > rgb_refl[2]) & valid
    cloud = np.isin(clm, (2, 3, 4))
    spread = float(np.std(lum[valid])) or 1.0
    best = None
    for strict in (True, False):
        for r in range(0, h - zh + 1, step):
            for c in range(0, w - zw + 1, step):
                v = valid[r:r + zh, c:c + zw].mean()
                cl = cloud[r:r + zh, c:c + zw].mean()
                if v < 0.98 or (strict and cl > 0.02):
                    continue
                score = veg[r:r + zh, c:c + zw].mean() + 0.3 * float(np.std(lum[r:r + zh, c:c + zw])) / spread - cl
                if best is None or score > best[0]:
                    best = (score, r, c)
        if best:
            break
    if best is None:
        return (h - zh) // 2, (w - zw) // 2, zh, zw
    return best[1], best[2], zh, zw


# ---- where --------------------------------------------------------------------------------------------------------

def countries(cache_dir, offline):
    path = cache_dir / "ne_50m_admin_0_countries.geojson"
    if not path.exists() and not offline:
        try:
            cache_dir.mkdir(parents=True, exist_ok=True)
            with urllib.request.urlopen(COUNTRIES_URL, timeout=60) as r:
                path.write_bytes(r.read())
        except Exception as e:
            print(f"   no country outlines ({type(e).__name__}): the map shows the coordinates only")
    if not path.exists():
        return []
    rings = []
    for feat in json.loads(path.read_text())["features"]:
        geom = feat.get("geometry") or {}
        polys = geom.get("coordinates", [])
        if geom.get("type") == "Polygon":
            polys = [polys]
        for poly in polys:
            if poly:
                rings.append(np.array(poly[0], "float64"))
    return rings


def place_name(lat, lon, offline):
    """'City, Region, Country' from OpenStreetMap (Nominatim), or None"""
    if offline:
        return None
    url = (f"https://nominatim.openstreetmap.org/reverse?format=jsonv2&lat={lat:.5f}&lon={lon:.5f}&zoom=10"
           "&accept-language=en")
    try:
        req = urllib.request.Request(url, headers={"User-Agent": "eaglet-l1d-panels/1.0"})
        with urllib.request.urlopen(req, timeout=20) as r:
            a = json.loads(r.read().decode("utf-8")).get("address", {})
        time.sleep(1.1)                                     # Nominatim allows one request a second
    except Exception as e:
        print(f"   no place name ({type(e).__name__}): the panels give the coordinates")
        return None
    town = next((a[k] for k in ("city", "town", "village", "municipality", "county") if a.get(k)), None)
    parts = [p for p in (town, a.get("state"), a.get("country")) if p]
    return ", ".join(dict.fromkeys(parts)) or None


def locator(ax, rings, lon0, lat0, footprint, fs, shape_ratio=1.0):
    from matplotlib.patches import Polygon
    half_lat = 6.0
    half_lon = half_lat * shape_ratio / max(math.cos(math.radians(lat0)), 0.2)
    x0, x1, y0, y1 = lon0 - half_lon, lon0 + half_lon, lat0 - half_lat, lat0 + half_lat
    ax.set_facecolor(SEA)
    for ring in rings:
        if ring[:, 0].max() < x0 or ring[:, 0].min() > x1 or ring[:, 1].max() < y0 or ring[:, 1].min() > y1:
            continue
        ax.add_patch(Polygon(ring, closed=True, facecolor=LAND, edgecolor=LAND_EDGE, linewidth=0.4))
    ax.add_patch(Polygon(footprint, closed=True, facecolor=MARK, edgecolor=MARK, linewidth=1.2, zorder=5))
    ax.plot(lon0, lat0, marker="o", markersize=16, markerfacecolor="none", markeredgecolor=MARK,
            markeredgewidth=1.8, zorder=6)
    ax.set_xlim(x0, x1)
    ax.set_ylim(y0, y1)
    ax.set_aspect(1 / max(math.cos(math.radians(lat0)), 0.2))
    for values, setter, labels, pos, neg in (((x0, x1), ax.set_xticks, ax.set_xticklabels, "E", "W"),
                                             ((y0, y1), ax.set_yticks, ax.set_yticklabels, "N", "S")):
        step = nice_step(values[1] - values[0], 5)
        ticks = np.arange(math.ceil(values[0] / step) * step, values[1] + 1e-9, step)
        setter(ticks)
        labels([deg(t, step, pos, neg) for t in ticks], fontsize=fs)
    ax.set_xlabel("Longitude", fontsize=fs + 1, color=INK2)
    ax.set_ylabel("Latitude", fontsize=fs + 1, color=INK2)
    ax.tick_params(length=3, width=0.6, colors=INK2)
    if not rings:
        ax.grid(color=LAND_EDGE, linewidth=0.4)


# ---- one product --------------------------------------------------------------------------------------------------

def when(text):
    try:
        return datetime.strptime(text[:19], "%Y-%m-%dT%H:%M:%S")
    except (TypeError, ValueError):
        return None


def load(l1d_path, l1c_path, max_px, zoom_m=None):
    d = Product(l1d_path)
    stac = json.loads(d.read(d.file(f"{d.id}.json")))
    xml = xml_values(d.read(d.file(f"{d.id}.xml")))
    img = d.gdal(d.image())
    g = grid(img, max_px)
    shape = g["shape"]
    dn_d = read_scaled(img, shape, Resampling.average).astype("float32")
    valid = valid_of(read_scaled(img, shape, Resampling.nearest))
    refl_d = dn_d * 0.0001 - 0.1
    clm_path, com_path = d.gdal(d.file(f"{d.id}.CLM.tif")), d.gdal(d.file(f"{d.id}.COM.tif"))
    clm = read_scaled(clm_path, shape, Resampling.nearest)[0]
    com = read_scaled(com_path, shape, Resampling.average)[0].astype("float32") / 10000.0
    with rasterio.open(clm_path) as src:                     # class shares over the whole image
        full = src.read(1)
    n_valid = max(int((full > 0).sum()), 1)
    shares = {v: 100.0 * float((full == v).sum()) / n_valid for v in (1, 2, 3, 4)}
    del full

    sw = stac["properties"].get("processing:software", {})
    version = xml.get("Processor_Version") or next(iter(sw.values()), "?")
    out = dict(id=d.id, stac=stac, xml=xml, g=g, valid=valid, refl_d=refl_d, clm=clm, com=com, shares=shares,
               l1c=None, version=version, rule_05=version != "01.00")      # 01.00: most likely class, no 0.5 rule

    sat = np.zeros(shape, bool)                       # saturated in the L1C: left out of the stretch, as 01.02 does
    if l1c_path is not None:
        c_ = Product(l1c_path)
        gains, offsets = calibration(c_)
        cimg = c_.gdal(c_.image())
        dn_c = read_scaled(cimg, shape, Resampling.average, [1, 2, 3]).astype("float32")
        cvalid = valid_of(read_scaled(cimg, shape, Resampling.nearest, [1, 2, 3])) & valid
        refl_c = dn_c * gains[:, None, None].astype("float32") + offsets[:, None, None].astype("float32")
        sat = (dn_c >= 0.98 * SATURATED_DN).any(0)
        lim_c = stretch_limits(refl_c, cvalid & ~sat)
        cx = xml_values(c_.read(c_.file(f"{c_.id}.xml")))
        out["l1c"] = dict(id=c_.id, rgb=to_rgb(refl_c, cvalid, lim_c), xml=cx)
    out["lim_d"] = stretch_limits(refl_d, valid & ~sat)
    out["rgb_d"] = to_rgb(refl_d, valid, out["lim_d"])

    if zoom_m:
        r, c, zh, zw = pick_zoom(refl_d, valid, clm, abs(g["draw"].a), zoom_m)
        fy, fx = g["full"][0] / shape[0], g["full"][1] / shape[1]
        win = Window(int(c * fx), int(r * fy), int(round(zw * fx)), int(round(zh * fy)))
        with rasterio.open(img) as src:
            zd = src.read([1, 2, 3], window=win).astype("float32")
            out["zoom"] = dict(rect=(c, r, zw, zh), g=dict(shape=zd.shape[1:], crs=src.crs,
                                                           draw=src.window_transform(win)))
        out["zoom"]["l1d"] = to_rgb(zd * 0.0001 - 0.1, (zd > 0).all(0), out["lim_d"])
        if out["l1c"]:
            with rasterio.open(cimg) as src:
                zc = src.read([1, 2, 3], window=win).astype("float32")
            zc_refl = zc * gains[:, None, None].astype("float32") + offsets[:, None, None].astype("float32")
            out["l1c"]["zoom"] = to_rgb(zc_refl, (zc > 0).all(0), lim_c)
    return out


def wrap(value, width):
    """lines of about width characters: at spaces, and product names after their underscores (a short last part
    of a name stays on the line before)"""
    pieces = []                                    # (text, glued to the piece before it, without a space)
    for word in value.split(" "):
        parts = re.split(r"(?<=_)(?=[^_])", word) if len(word) > width else [word]
        pieces += [(part, k > 0) for k, part in enumerate(parts)]
    lines, line = [], ""
    for i, (text, glued) in enumerate(pieces):
        joined = line + text if glued else (f"{line} {text}" if line else text)
        short_end = glued and len(text) <= 6 and (i + 1 == len(pieces) or not pieces[i + 1][1])
        if len(joined) > width and line and not short_end:
            lines.append(line)
            line = text
        else:
            line = joined
    return lines + ([line] if line else [])


def scene_facts(p, aoi):
    """the lines of panel (f), and the header"""
    s, x, g = p["stac"]["properties"], p["xml"], p["g"]
    m = NAME.match(p["id"])
    sat = m.group(1)
    t = when(s.get("start_datetime") or s.get("datetime") or x.get("PRODUCT_SENSING_START_TIME"))
    tstr = t.strftime("%d %b %Y, %H:%M:%S UTC") if t else "?"
    h, w = g["full"]
    px = abs(g["transform"].a)
    rows, cols = np.nonzero(p["valid"])
    cy, cx_ = rows.mean(), cols.mean()
    lon0, lat0 = (v[0] for v in warp_transform(g["crs"], "EPSG:4326", *[[q] for q in g["draw"] * (cx_ + 0.5, cy + 0.5)]))
    size_km = ((cols.max() - cols.min() + 1) * abs(g["draw"].a) / 1000, (rows.max() - rows.min() + 1) * abs(g["draw"].e) / 1000)
    epsg = g["crs"].to_epsg()
    zone = f"UTM zone {epsg % 100}{'N' if epsg // 100 == 326 else 'S'}" if epsg and epsg // 100 in (326, 327) else str(g["crs"])
    sh = p["shares"]
    cloud = s.get("eo:cloud_cover", sh[2] + sh[3])
    l1d_version = p["version"]
    rule = "cloud where COM >= 0.5" if p["rule_05"] else "each pixel in its most likely class"
    method = g["tags"].get("COLOUR_METHOD", "?")
    aux = g["tags"].get("COLOUR_AUX_FILE") or x.get("AUXILIARY_FILE", "?")
    l1c = p["l1c"]
    l1c_version = (l1c or {}).get("xml", {}).get("Processor_Version")
    where = f"{aoi}; " if aoi else ""
    facts = [
        ("Flight model", f"{sat} ({s.get('platform', x.get('SPACECRAFT_NAME', '?'))}), instrument "
                         f"{', '.join(s.get('instruments', ['MSI']))}, {s.get('iride:instrument_mode', 'TDI')}"),
        ("Sensing", f"{tstr}; orbit {s.get('sat:absolute_orbit', int(m.group(6)))} (relative "
                    f"{s.get('sat:relative_orbit', int(m.group(7)[1:]))}), {s.get('sat:orbit_state', '?')}"),
        ("Sun / view", f"sun elevation {s.get('view:sun_elevation', float('nan')):.1f}°, azimuth "
                       f"{s.get('view:sun_azimuth', float('nan')):.1f}°; off-nadir {s.get('view:off_nadir', float('nan')):.1f}°"),
        ("AOI", f"{where}centre {deg(lat0, 0.001, 'N', 'S')}, {deg(lon0, 0.001, 'E', 'W')}; about "
                f"{size_km[0]:.0f} x {size_km[1]:.0f} km"),
        ("Image", f"{w:,} x {h:,} pixels of {px:.2f} m, {zone} (EPSG:{epsg})"),
        ("Cloud", f"cloud cover {cloud:.1f}% (thick {sh[2]:.1f}%, thin {sh[3]:.1f}%), shadow {sh[4]:.1f}%, "
                  f"clear {sh[1]:.1f}%"),
        ("L1C input", f"{l1c['id'] if l1c else '(not found)'}" + (f", L1C processor {l1c_version}" if l1c_version else "")),
        ("L1D product", p["id"]),
        ("Processing", f"EAGLET2_L1D_PROC {l1d_version}: cloud mask"
                       + (" by Clarity" if str(l1d_version) >= "01.02" else "") + f", {rule}; "
                       f"colour correction {method}"),
        ("Colour AUX", aux),
    ]
    header = f"EAGLET-2 L1C to L1D  ·  {sat}  ·  {tstr}" + (f"  ·  {aoi}" if aoi else "")
    return facts, header, (lon0, lat0)


def cloud_cmap():
    from matplotlib.colors import LinearSegmentedColormap
    cmap = LinearSegmentedColormap.from_list("com", COM_RAMP)
    cmap.set_bad(NODATA_RGB)
    return cmap


def clm_overlay(p):
    """the L1D in grey, the cloud and shadow classes of the CLM in colour, white where there is no data"""
    grey = 0.32 + 0.6 * p["rgb_d"].mean(-1)
    over = np.repeat(grey[..., None], 3, -1)
    for v, _, col in CLM_CLASSES:
        over[p["clm"] == v] = [int(col[i:i + 2], 16) / 255 for i in (1, 3, 5)]
    over[~p["valid"]] = NODATA_RGB
    return over


def clm_handles(shares=None):
    from matplotlib.patches import Patch
    pct = (lambda v: f"  {shares[v]:.1f}%") if shares else (lambda v: "")
    return ([Patch(facecolor="#9a9a9a", edgecolor="none", label="clear (the image in grey)" + pct(1))]
            + [Patch(facecolor=col, edgecolor="none", label=name + pct(v)) for v, name, col in CLM_CLASSES])


def com_panel(ax, p, g, title, fs, cmap, extras, line):
    com = np.ma.masked_where(~p["valid"], p["com"])
    map_panel(ax, com, g, title, fs, extras=extras, cmap=cmap, vmin=0, vmax=1)
    inside = p["com"][p["valid"]]
    if line and (inside >= 0.5).any() and (inside < 0.5).any():
        h, w = g["shape"]
        ax.contour(np.arange(w) + 0.5, np.arange(h) + 0.5, np.where(p["valid"], p["com"], 0), levels=[0.5],
                   colors=INK, linewidths=0.6)
    return com


def fact_lines(ax, x, items, fs, width, indent):
    """label in bold and its value beside it, line by line from the top of ax; lines placed in points, so they do
    not depend on the height the layout gives the axes"""
    from matplotlib.transforms import offset_copy
    step = fs * 1.5
    k = 0.0
    for label, value in items:
        at = offset_copy(ax.transAxes, fig=ax.figure, x=0, y=-k * step, units="points")
        ax.text(x, 1.0, label, transform=at, fontsize=fs, fontweight="bold", color=INK, va="top")
        for t in wrap(value, width):
            at = offset_copy(ax.transAxes, fig=ax.figure, x=0, y=-k * step, units="points")
            ax.text(x + indent, 1.0, t, transform=at, fontsize=fs, color=INK, va="top")
            k += 1
        k += 0.5


def draw_scene(p, aoi, rings, out_dir, dpi):
    import matplotlib.pyplot as plt
    from matplotlib.patches import Rectangle

    fs = 9
    g = p["g"]
    h, w = g["shape"]
    ratio = w / h
    facts, header, (lon0, lat0) = scene_facts(p, aoi)
    zoom = p.get("zoom")
    panel_w = 4.9
    fig = plt.figure(figsize=(4 * panel_w + 1.8, 2 * panel_w / ratio + 2.7), layout="constrained")
    fig.get_layout_engine().set(w_pad=0.08, h_pad=0.1, wspace=0.05, hspace=0.08)
    gs = fig.add_gridspec(2, 4)
    ax = {k: fig.add_subplot(gs[0, i]) for i, k in enumerate("abcd")}
    if zoom:                          # (e) L1C zoom, (f) L1D zoom, (g) where, (h) the product
        ax.update({k: fig.add_subplot(gs[1, i]) for i, k in enumerate("efgh")})
        where, about = "g", "h"
    else:                             # (e) where, (f) the product, three columns wide
        ax["e"], ax["f"] = fig.add_subplot(gs[1, 0]), fig.add_subplot(gs[1, 1:])
        where, about = "e", "f"
    l1c = p["l1c"]

    def box(a):
        c, r, zw, zh = zoom["rect"]
        for col, lw in (("white", 2.6), (INK, 1.2)):
            a.add_patch(Rectangle((c, r), zw, zh, fill=False, edgecolor=col, linewidth=lw, zorder=4))
        a.text(c + zw / 2, r - zh * 0.06, "(e, f)", ha="center", va="bottom", fontsize=fs, color=INK,
               path_effects=halo(), zorder=6)

    if l1c:
        map_panel(ax["a"], l1c["rgb"], g, "(a) L1C as delivered (top-of-atmosphere reflectance)", fs)
        if zoom:
            box(ax["a"])
    else:
        ax["a"].text(0.5, 0.5, "L1C not found", ha="center", va="center", transform=ax["a"].transAxes)
        ax["a"].set_axis_off()
    map_panel(ax["b"], p["rgb_d"], g, "(b) L1D, colour corrected towards Sentinel-2", fs)
    if zoom:
        box(ax["b"])

    over = clm_overlay(p)
    map_panel(ax["c"], over, g, "(c) Cloud mask (CLM)", fs)
    ax["c"].legend(handles=clm_handles(p["shares"]), loc="lower left", fontsize=fs, framealpha=0.92,
                   edgecolor="none", title="share of the image", title_fontsize=fs)

    cmap = cloud_cmap()
    com = com_panel(ax["d"], p, g, "(d) Cloud confidence (COM)", fs, cmap, True, p["rule_05"])
    cax = ax["d"].inset_axes([1.04, 0.0, 0.045, 1.0])
    cb = fig.colorbar(ax["d"].images[0], cax=cax, ticks=[0, 0.25, 0.5, 0.75, 1])
    cb.set_label("P(thick cloud) + P(thin cloud)" + ("; the line at 0.5 is the CLM cloud" if p["rule_05"] else ""),
                 fontsize=fs, color=INK2)
    cb.ax.tick_params(labelsize=fs, colors=INK2)
    if p["rule_05"]:
        cb.ax.axhline(0.5, color=INK, linewidth=1.2)
    cb.outline.set_linewidth(0.6)

    if zoom:
        if l1c:
            map_panel(ax["e"], l1c["zoom"], zoom["g"], "(e) L1C as delivered, full resolution", fs)
        else:
            ax["e"].set_axis_off()
        map_panel(ax["f"], zoom["l1d"], zoom["g"], "(f) L1D, full resolution", fs)

    fp = np.array(p["stac"].get("geometry", {}).get("coordinates", [[[lon0, lat0]]])[0])
    locator(ax[where], rings, lon0, lat0, fp, fs, ratio)
    ax[where].set_anchor("N")                    # level with the top of the panel beside it
    ax[where].set_title(f"({where}) Where: " + (aoi or f"{deg(lat0, 0.001, 'N', 'S')}, {deg(lon0, 0.001, 'E', 'W')}"),
                        loc="left", fontsize=fs + 3, color=INK, pad=6)

    a = ax[about]
    a.set_axis_off()
    a.set_title(f"({about}) The product", loc="left", fontsize=fs + 3, color=INK, pad=6)
    if zoom:
        fact_lines(a, 0.0, facts, fs, 46, 0.27)
    else:
        half = (len(facts) + 1) // 2
        fact_lines(a, 0.0, facts[:half], fs + 1, 64, 0.08)            # the acquisition
        fact_lines(a, 0.52, facts[half:], fs + 1, 64, 0.08)           # the products and the processing

    fig.suptitle(header, x=0.004, ha="left", fontsize=fs + 9, fontweight="bold", color=INK)
    fig.text(0.004, -0.004, "The EAGLET-2 L1D processor (EAGLET2_L1D_PROC) adds a cloud mask and a cloud confidence "
             "(Clarity) to the L1C and colour-corrects its RGB towards Sentinel-2 L1C (LUT). True colour "
             "as in the L1D quicklook: each band from its own dark level (the haze), one range for the three bands, "
             "gamma 1.6; the L1C the same way; white where there is no data. Maps on the product's grid, axes in "
             "geographic coordinates where parallels and meridians cross the image edges.",
             fontsize=fs, color=INK2, ha="left", va="top", wrap=True)
    stem = out_dir / f"panel_{p['id']}"
    fig.savefig(f"{stem}.png", dpi=dpi, bbox_inches="tight")
    fig.savefig(f"{stem}.pdf", bbox_inches="tight")
    plt.close(fig)

    tiles = out_dir / "tiles"
    tiles.mkdir(exist_ok=True)
    plt.imsave(tiles / f"{p['id']}_L1D.png", p["rgb_d"])
    plt.imsave(tiles / f"{p['id']}_CLM.png", over)
    plt.imsave(tiles / f"{p['id']}_COM.png", cmap(np.ma.filled(com.astype("float64"), np.nan)))
    if l1c:
        plt.imsave(tiles / f"{p['id']}_L1C.png", l1c["rgb"])
    if zoom:
        plt.imsave(tiles / f"{p['id']}_L1D_zoom.png", zoom["l1d"])
        if l1c:
            plt.imsave(tiles / f"{p['id']}_L1C_zoom.png", l1c["zoom"])
    return over, cmap


def draw_overview(items, out_dir, dpi):
    import matplotlib.pyplot as plt
    from matplotlib.cm import ScalarMappable
    from matplotlib.colors import Normalize

    fs = 8
    panel_w = 4.6
    heights = [panel_w * it[0]["g"]["shape"][0] / it[0]["g"]["shape"][1] + 0.9 for it in items] + [1.0]
    fig = plt.figure(figsize=(4 * panel_w + 1.2, sum(heights) + 0.8), layout="constrained")
    subs = fig.subfigures(len(items) + 1, 1, height_ratios=heights)
    for sf, (p, aoi, over, cmap) in zip(subs, items):
        g, sh = p["g"], p["shares"]
        m = NAME.match(p["id"])
        t = when(p["stac"]["properties"].get("start_datetime"))
        sf.suptitle(f"{m.group(1)}  ·  {t.strftime('%d %b %Y, %H:%M UTC') if t else ''}"
                    + (f"  ·  {aoi}" if aoi else ""), x=0.004, ha="left", fontsize=fs + 5, fontweight="bold",
                    color=INK)
        axs = sf.subplots(1, 4)
        if p["l1c"]:
            map_panel(axs[0], p["l1c"]["rgb"], g, "L1C as delivered", fs)
        else:
            axs[0].set_axis_off()
        map_panel(axs[1], p["rgb_d"], g, "L1D, colour corrected", fs, extras=False)
        cloud = p["stac"]["properties"].get("eo:cloud_cover", sh[2] + sh[3])
        map_panel(axs[2], over, g, f"Cloud mask (CLM): cloud {cloud:.1f}%, shadow {sh[4]:.1f}%", fs, extras=False)
        com_panel(axs[3], p, g, "Cloud confidence (COM)", fs, cmap, False, p["rule_05"])
    la, ca = subs[-1].subplots(1, 2, gridspec_kw=dict(width_ratios=(1.3, 1)))
    la.set_axis_off()
    la.legend(handles=clm_handles(), ncol=4, loc="center", frameon=False, fontsize=fs + 1, title="Cloud mask (CLM)",
              title_fontsize=fs + 1)
    ca.set_axis_off()
    bar = ca.inset_axes([0.05, 0.45, 0.9, 0.22])
    cb = fig.colorbar(ScalarMappable(norm=Normalize(0, 1), cmap=items[0][3]), cax=bar, orientation="horizontal",
                      ticks=[0, 0.25, 0.5, 0.75, 1])
    all_05 = all(it[0]["rule_05"] for it in items)
    cb.set_label("Cloud confidence (COM): P(thick cloud) + P(thin cloud)"
                 + ("; the line at 0.5 is the CLM cloud" if all_05 else ""), fontsize=fs + 1, color=INK2)
    cb.ax.tick_params(labelsize=fs, colors=INK2)
    if all_05:
        cb.ax.axvline(0.5, color=INK, linewidth=1.2)
    fig.suptitle("EAGLET-2 L1C to L1D processor: the products", x=0.004, ha="left", fontsize=17, fontweight="bold",
                 color=INK)
    fig.savefig(out_dir / "overview.png", dpi=dpi, bbox_inches="tight")
    fig.savefig(out_dir / "overview.pdf", bbox_inches="tight")
    plt.close(fig)


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("l1d", nargs="+", help="L1D zips, or folders holding them")
    ap.add_argument("--l1c", nargs="*", default=[], help="L1C zips, or folders holding them")
    ap.add_argument("--out", type=Path, default=REPO.parent / "poster_panels", help="results (default: %(default)s)")
    ap.add_argument("--offline", action="store_true", help="no place names, no country outlines")
    ap.add_argument("--max-px", type=int, default=1600, help="long side of the scene images (default: %(default)s)")
    ap.add_argument("--zoom", action="store_true", help="add a part of the L1C and of the L1D at full resolution")
    ap.add_argument("--zoom-m", type=float, default=2000, help="width of that part in metres (default: %(default)s)")
    ap.add_argument("--dpi", type=int, default=200)
    a = ap.parse_args(argv)
    import matplotlib
    if tuple(int(v) for v in re.findall(r"\d+", matplotlib.__version__)[:2]) < (3, 6):
        sys.exit(f"needs matplotlib 3.6 or newer, this is {matplotlib.__version__}: pip install -U matplotlib")
    matplotlib.use("Agg")

    l1ds, l1cs = find_products(a.l1d, "D"), find_products(a.l1c, "C")
    if not l1ds:
        sys.exit("no L1D product found")
    a.out.mkdir(parents=True, exist_ok=True)
    names_file = a.out / "aoi_names.json"
    names = json.loads(names_file.read_text()) if names_file.exists() else {}
    rings = countries(Path.home() / ".cache" / "eaglet_l1d_panels", a.offline)
    items = []
    for key in sorted(l1ds, key=lambda k: (k[1], k[0])):
        path = l1ds[key]
        print(f"{path.name}  (L1C: {l1cs[key].name if key in l1cs else 'not found'})")
        p = load(path, l1cs.get(key), a.max_px, a.zoom_m if a.zoom else None)
        if names.get(p["id"]) is None and not a.offline:      # not looked up yet, or the lookup failed
            _, _, (lon0, lat0) = scene_facts(p, None)
            names[p["id"]] = place_name(lat0, lon0, a.offline)
            names_file.write_text(json.dumps(names, indent=2, ensure_ascii=False))
        aoi = names.get(p["id"]) or None
        over, cmap = draw_scene(p, aoi, rings, a.out, a.dpi)
        items.append((p, aoi, over, cmap))
        print(f"   {a.out / ('panel_' + p['id'] + '.png')}  AOI: {aoi or '(coordinates only)'}")
    draw_overview(items, a.out, a.dpi)
    print(f"{a.out / 'overview.png'}, PDFs next to the PNGs, single images in {a.out / 'tiles'}")


if __name__ == "__main__":
    main()
