#!/usr/bin/env python3
"""
build_lut.py: fits the colour-correction LUT of the EAGLET-2 L1D processor again, on the L1C products of one
processing baseline (A01 by default), and writes the new colour AUX file (AXE1D_CLUT).

The method is the one of LUT v3 (lut_build notebook, Cells 1 to 13); only its numbers are fitted again:
  1 inventory  the L1C products of the baseline, read inside their zips (nothing is extracted)
  2 s2         Sentinel-2 L1C over each footprint at 10 m, within +/- 3 days: clouds left out (L2A SCL plus
               OmniCloudMask, 30 m around them), each cell from the closest clear pass
  3 pairs      EAGLET fitted onto Sentinel-2 (SIFT features, RANSAC affine), averaged into the 10 m cells; clear,
               uniform cells only: up to 300,000 EAGLET / Sentinel-2 pairs per product
  4 chips      the Sentinel-2 world chips (Cell 10b): reused when on disk, downloaded otherwise
  5 fit        from the pairs: the near-infrared share of each EAGLET band and the colour balance of the
               delivered calibration. From Sentinel-2 alone (chips and footprints, turned into what EAGLET's bands
               see): the vegetation table and the table q(x, y, v)
  6 aux        the new AUX file: the current one with the balance, the shares and the tables replaced (Enrico's
               class model and <Method> untouched); read back, then loaded with the processor's own code
  7 score      every product against Sentinel-2: as delivered, current LUT, new LUT (with the product left out of
               the training, and with all of them), plus a picture of a 2 km crop per product

Usage, in the processor's environment, from the repository folder:
    python tools/build_lut.py                   # every step; downloads and per-product results are kept
    python tools/build_lut.py --redo pairs      # do a step (and the ones after it) again for every product
Sentinel-2 comes from the Copernicus Data Space (Sentinel Hub API). The client id and secret are read from
CDSE_CLIENT_ID and CDSE_CLIENT_SECRET, or asked for (the secret hidden) when a download is needed.
Needs, besides the processor's packages: opencv-python-headless, requests, matplotlib.
Results in <work> (default ../lut_a01): summary.txt, scores.csv, lut_check.png, aux/<new AUX file>.
"""

import argparse
import base64
import csv
import getpass
import hashlib
import json
import math
import os
import re
import sys
import time
import xml.etree.ElementTree as ET
import zipfile
from datetime import datetime, timedelta, timezone
from pathlib import Path

import numpy as np
import rasterio
from rasterio.transform import Affine, from_origin
from rasterio.warp import Resampling, reproject
from rasterio.warp import transform as warp_transform
from scipy import ndimage
from scipy.spatial import cKDTree

REPO = Path(__file__).resolve().parents[1]
PROJECT = REPO.parent
STAGES = ("inventory", "s2", "pairs", "chips", "fit", "aux", "score")

RES = 10.0                    # metres: the Sentinel-2 grid everything is compared on
WINDOW_DAYS = 3               # Sentinel-2 passes up to 3 days before or after the EAGLET image ...
WINDOW_DAYS_MAX = 6           # ... or up to 6 days when that leaves less than half of the footprint clear
MIN_FILLED = 0.5
MAX_PASSES = 6                # passes per footprint at most, closest in time first
MIN_PASS_CLEAR = 0.10         # a pass is worth downloading when at least 10% of the footprint is clear
CLOUDY_SCL = (0, 1, 3, 8, 9, 10, 11)   # L2A SCL: no data, defective, shadow, cloud medium/high, cirrus, snow
CLOUD_BUFFER = 3              # cells (30 m) left out around every cloud and shadow
MAX_SPREAD = 0.08             # a 10 m cell is uniform when its values vary by less than 8%
MAX_PAIRS = 300_000           # pairs per product, picked at random
MAX_LEFTOVER_M = 10.0         # ground near a 2 km tile still off by more than 10 m after the fit is left out
TILE = 200                    # leftover check in tiles of 200 x 200 cells (2 km), overlapping by half
MIN_MATCH = 15.0              # a tile counts when its match peak is 15x the background
MIN_FEATURES = 20             # a geometry fit needs 20 features that agree with it
STRETCHES = ((1, 2), (2, 1), (1, 0.5), (0.5, 1))   # tried when the image as delivered gives too few features
MIN_R2_SHARES = 0.6           # products whose green fit is this good set the near-infrared shares
MIN_SHARE = 0.03              # a share smaller than this (either sign) is taken as 0
BAL_MIN_R2 = 0.8              # products whose fit against fake EAGLET is this good set the colour balance
SHARE_JITTER = 0.10           # training varies the near-infrared shares by up to +/- 10%
MAX_PER_SOURCE = 1_000_000    # training pixels per chip or footprint, picked at random
VEG_SAMPLE = 300_000          # pixels per source for the vegetation table
SMOOTH = 1.0                  # smoothing between neighbouring table cells (half as much along "around")
K_SUPPORT = 50.0              # pixels a cell needs before its own value counts more than the fallback
SCORE_PAIRS = 100_000         # pairs per product for the scores
DISPLAY = 0.3                 # pictures and colour errors: reflectance 0 to 0.3, like a Sentinel-2 true colour
CROP = 200                    # picture crops, in 10 m cells (2 km)
NAME = re.compile(r"^(IME\d{2})_1CSP__RGB3_(\d{8}T\d{6})_(\d{8}T\d{6})_(\d{8}T\d{6})_(\d{5})_(R\d{3})_([A-Z])_"
                  r"([A-Z0-9]{3})$")
rng = np.random.default_rng(0)


def log(*parts):
    print(*parts, flush=True)


def table_text(rows, cols=None):
    """rows of dicts as an aligned text table"""
    if not rows:
        return "(none)"
    cols = cols or list(rows[0])
    cells = [[str(c) for c in cols]] + [[fmt(r.get(c, "")) for c in cols] for r in rows]
    widths = [max(len(row[i]) for row in cells) for i in range(len(cols))]
    return "\n".join("  ".join(v.rjust(w) for v, w in zip(row, widths)) for row in cells)


def fmt(v):
    if isinstance(v, float):
        return "nan" if not math.isfinite(v) else f"{v:.4g}" if abs(v) < 1e4 else f"{v:.0f}"
    return str(v)


# ---------------------------------------------------------------------------------------------------------------
# 1 inventory
# ---------------------------------------------------------------------------------------------------------------

def rescaling_factors(xml_bytes):
    """GAIN and OFFSET of bands 1, 2, 3 from <Rescaling_Factors Band_ID="n"> in the product XML"""
    found = {}
    for el in ET.fromstring(xml_bytes).iter():
        if el.tag.split("}")[-1] != "Rescaling_Factors":
            continue
        band = el.get("Band_ID")
        values = {ch.tag.split("}")[-1].upper(): (ch.text or "").strip() for ch in el}
        if band and values.get("GAIN") and values.get("OFFSET"):
            found[int(band)] = (float(values["GAIN"]), float(values["OFFSET"]))
    if sorted(found) != [1, 2, 3]:
        raise ValueError(f"Rescaling_Factors found for bands {sorted(found)}, expected 1, 2, 3")
    return [found[b][0] for b in (1, 2, 3)], [found[b][1] for b in (1, 2, 3)]


def processor_xml_reader():
    """the processor's own reader of the XML calibration (so the gains are exactly the ones it uses), or None"""
    import importlib
    sys.path.insert(0, str(REPO / "src"))
    for module, name in (("iride_eaglet_processor.calibration", "read_xml_calibration"),
                         ("iride_eaglet_processor.raster_reader", "load_xml_rescaling_factors")):
        try:
            fn = getattr(importlib.import_module(module), name)
        except Exception:
            continue
        log(f"gains read with the processor's own {module.split('.')[-1]}.{name}")
        return fn
    log("!! the processor's XML reader was not found: gains read with a simpler one")
    return None


def as_bands(cal):
    """the processor reader's result as {band: (gain, offset)}, whatever its form"""
    def pair(v):
        if isinstance(v, dict):
            low = {str(k).lower(): x for k, x in v.items()}
            return float(low["gain"]), float(low["offset"])
        return float(v[0]), float(v[1])
    if isinstance(cal, dict):
        for keys in ((1, 2, 3), ("1", "2", "3")):
            if all(k in cal for k in keys):
                return {b: pair(cal[k]) for b, k in zip((1, 2, 3), keys)}
        low = {str(k).lower(): v for k, v in cal.items()}
        for kg, ko in (("gains", "offsets"), ("gain", "offset")):
            if kg in low and ko in low:
                return {b: (float(low[kg][b - 1]), float(low[ko][b - 1])) for b in (1, 2, 3)}
    if isinstance(cal, (list, tuple)) and len(cal) == 3:
        return {b: pair(cal[b - 1]) for b in (1, 2, 3)}
    if isinstance(cal, (list, tuple)) and len(cal) == 2 and all(len(c) == 3 for c in cal):
        return {b: (float(cal[0][b - 1]), float(cal[1][b - 1])) for b in (1, 2, 3)}
    raise TypeError(f"unexpected form {type(cal).__name__}")


def read_product(zip_path, xml_dir, reader):
    stem = zip_path.stem
    with zipfile.ZipFile(zip_path) as zf:
        names = set(zf.namelist())
        stac = json.loads(zf.read(f"{stem}/{stem}.json"))
        assets = stac.get("assets", {})

        def member(asset, default):
            href = assets.get(asset, {}).get("href", default)
            return f"{stem}/{href[2:] if href.startswith('./') else href}"

        tif, xml = member("rgb", f"./IMG_DATA/{stem}.tif"), member("metadata", f"./{stem}.xml")
        missing = [m for m in (tif, xml) if m not in names]
        if missing:
            raise ValueError(f"not in the zip: {', '.join(missing)}")
        xml_path = xml_dir / f"{stem}.xml"
        xml_path.write_bytes(zf.read(xml))
    gains = offsets = None
    if reader is not None:
        try:
            cal = reader(xml_path)
        except TypeError:
            cal = None                                    # called differently: the simpler reader is used
        except Exception as e:
            raise ValueError(f"the processor's XML reader rejects it: {type(e).__name__}: {e}")
        try:
            bands = as_bands(cal) if cal is not None else None
        except (TypeError, KeyError, IndexError, ValueError):
            bands = None
        if bands is not None:
            gains, offsets = [bands[b][0] for b in (1, 2, 3)], [bands[b][1] for b in (1, 2, 3)]
    if gains is None:
        gains, offsets = rescaling_factors(xml_path.read_bytes())
    vsi = f"/vsizip/{zip_path.resolve()}/{tif}"
    stats_path = xml_dir / f"{stem}.stats.json"
    with rasterio.open(vsi) as src:
        if src.crs is None or not src.crs.is_projected:
            raise ValueError(f"the image is not in a projected (UTM) system: {src.crs}")
        info = dict(epsg=src.crs.to_epsg(), bounds=list(src.bounds), res=src.res[0], width=src.width,
                    height=src.height, count=src.count)
        if info["count"] != 3:
            raise ValueError(f"the image has {info['count']} bands, expected 3")
        if stats_path.exists():
            stats = json.loads(stats_path.read_text())
        else:                                             # median DN per band, on a small copy of the image
            f = max(1, max(src.width, src.height) // 1000)
            small = src.read(out_shape=(3, max(1, src.height // f), max(1, src.width // f)))
            valid = (small > 0).all(0)
            stats = dict(dn_median=[float(np.median(small[b][valid])) if valid.any() else float("nan")
                                    for b in range(3)])
            stats_path.write_text(json.dumps(stats))
    # reflectance as delivered: far outside 0..1 means the XML calibration is wrong for this product
    refl = [stats["dn_median"][b] * gains[b] + offsets[b] for b in range(3)]
    info.update(refl_median=[round(r, 4) for r in refl],
                calibration_ok=bool(all(math.isfinite(r) and -0.05 < r < 1.0 for r in refl) and refl[1] > 0.005))
    m = NAME.match(stem)
    sensing = m.group(2)
    when = (stac.get("properties", {}).get("datetime")
            or f"{sensing[:4]}-{sensing[4:6]}-{sensing[6:8]}T{sensing[9:11]}:{sensing[11:13]}:{sensing[13:15]}Z")
    return dict(stem=stem, zip=str(zip_path.resolve()), tif=vsi, sat=m.group(1), sensing=sensing,
                generated=m.group(4), time=when, bbox_ll=list(stac["bbox"]),
                gains=gains, offsets=offsets, **info)


def inventory_stage(a, work):
    zips = {}
    for z in sorted(a.products.rglob("IME*_1CSP__RGB3_*.zip")):
        m = NAME.match(z.stem)
        if m and m.group(8) == a.baseline:
            zips.setdefault(z.name, z)
    rows, chosen = [], {}
    for name, z in sorted(zips.items()):
        m = NAME.match(z.stem)
        if m.group(4)[:8] < a.since:
            rows.append(dict(product=z.stem, note=f"left out: generated before {a.since}"))
            continue
        key = (m.group(1), m.group(2))                  # the same acquisition processed more than once
        if key in chosen and chosen[key].stem > z.stem:
            rows.append(dict(product=z.stem, note="left out: a newer processing of the same image is used"))
            continue
        if key in chosen:
            rows.append(dict(product=chosen[key].stem, note="left out: a newer processing of the same image is used"))
        chosen[key] = z
    prods = []
    xml_dir = work / "xml"
    xml_dir.mkdir(exist_ok=True)
    reader = processor_xml_reader()
    for z in sorted(chosen.values(), key=lambda z: NAME.match(z.stem).group(2)):
        try:
            prods.append(read_product(z, xml_dir, reader))
        except zipfile.BadZipFile:
            rows.append(dict(product=z.stem, note=f"left out: not a complete zip ({z.stat().st_size / 1e6:.0f} MB): "
                                                  "an unfinished upload or download?"))
        except Exception as e:
            rows.append(dict(product=z.stem, note=f"left out: {type(e).__name__}: {e}"))
    for k, p in enumerate(prods):
        p["label"] = f"#{k} {p['sat']} {p['sensing'][:8]}"
    (work / "inventory.json").write_text(json.dumps(prods, indent=1))
    log(f"{len(prods)} {a.baseline} products in {a.products} (generated from {a.since} on):")
    log(table_text([dict(id=p["label"], size_km=f"{p['width'] * p['res'] / 1e3:.1f} x {p['height'] * p['res'] / 1e3:.1f}",
                         gsd_m=round(p["res"], 2), epsg=p["epsg"],
                         gain1=f"{p['gains'][0]:.3e}", gain2=f"{p['gains'][1]:.3e}", gain3=f"{p['gains'][2]:.3e}",
                         median_reflectance=" / ".join(f"{r:.3g}" for r in p["refl_median"]),
                         calibration="ok" if p["calibration_ok"] else "LOOKS WRONG",
                         product=p["stem"]) for p in prods]))
    for p in prods:
        if not p["calibration_ok"]:
            log(f"!! {p['label']}: with its XML gains the median reflectance is "
                f"{' / '.join(f'{r:.3g}' for r in p['refl_median'])} (expected roughly 0.02 to 0.5): its calibration "
                f"looks wrong. It is left out of the fit and the scores; its Sentinel-2 footprint still trains the table")
    for r in rows:
        log(f"   {r['product']}: {r['note']}")
    if not prods:
        raise SystemExit(f"no {a.baseline} product found in {a.products}")
    return prods


def load_inventory(work):
    path = work / "inventory.json"
    if not path.exists():
        raise SystemExit("run the inventory step first")
    return json.loads(path.read_text())


def grid_of(bounds, res=RES):
    """the bounding box snapped outwards to the 10 m grid: bounds, width, height, transform"""
    left, bottom, right, top = bounds
    left, bottom = math.floor(left / res) * res, math.floor(bottom / res) * res
    right, top = math.ceil(right / res) * res, math.ceil(top / res) * res
    w, h = int(round((right - left) / res)), int(round((top - bottom) / res))
    return (left, bottom, right, top), w, h, from_origin(left, top, res, res)


def utc(text):
    t = datetime.fromisoformat(text.replace("Z", "+00:00"))
    return t if t.tzinfo else t.replace(tzinfo=timezone.utc)


# ---------------------------------------------------------------------------------------------------------------
# Copernicus Data Space (Sentinel Hub API) and OmniCloudMask
# ---------------------------------------------------------------------------------------------------------------

EVAL_L1C = """//VERSION=3
function setup() { return { input: [{ bands: ["B04", "B03", "B02", "B08", "dataMask"], units: "DN" }],
                            output: { bands: 5, sampleType: "UINT16" } }; }
function evaluatePixel(s) { return [s.B04, s.B03, s.B02, s.B08, s.dataMask]; }"""
EVAL_SCL = """//VERSION=3
function setup() { return { input: [{ bands: ["SCL", "dataMask"] }], output: { bands: 2, sampleType: "UINT8" } }; }
function evaluatePixel(s) { return [s.SCL, s.dataMask]; }"""


class CDSE:
    TOKEN_URL = "https://identity.dataspace.copernicus.eu/auth/realms/CDSE/protocol/openid-connect/token"
    PROCESS_URL = "https://sh.dataspace.copernicus.eu/api/v1/process"
    CATALOG_URLS = ("https://sh.dataspace.copernicus.eu/api/v1/catalog/1.0.0/search",
                    "https://sh.dataspace.copernicus.eu/catalog/v1/search")

    def __init__(self):
        import requests
        self.http = requests
        self.client_id = os.environ.get("CDSE_CLIENT_ID") or input("CDSE client id: ").strip()
        self.client_secret = (os.environ.get("CDSE_CLIENT_SECRET")
                              or getpass.getpass("CDSE client secret (hidden): ").strip())
        self.token, self.expires, self.catalog_url = None, 0.0, None
        self._headers()
        log("logged in to the Copernicus Data Space")

    def _headers(self):
        if time.time() > self.expires - 60:
            r = self.http.post(self.TOKEN_URL, timeout=60, data={
                "grant_type": "client_credentials", "client_id": self.client_id, "client_secret": self.client_secret})
            if r.status_code != 200:
                raise SystemExit(f"CDSE login failed (HTTP {r.status_code}): check the client id and secret")
            j = r.json()
            self.token, self.expires = j["access_token"], time.time() + float(j.get("expires_in", 600))
        return {"Authorization": f"Bearer {self.token}"}

    def _post(self, url, body):
        for attempt in range(6):
            try:
                r = self.http.post(url, json=body, headers=self._headers(), timeout=300)
            except self.http.exceptions.RequestException:
                time.sleep(10 * (attempt + 1))
                continue
            if r.status_code not in (429, 500, 502, 503, 504):
                return r
            time.sleep(10 * (attempt + 1))
        raise RuntimeError(f"no answer from {url} after 6 tries")

    def days(self, bbox_ll, t0, t1):
        """(acquisition time, tile cloud %) of the Sentinel-2 L1C passes over bbox_ll from t0 to t1, one per day"""
        body = {"collections": ["sentinel-2-l1c"], "bbox": [float(v) for v in bbox_ll], "limit": 100,
                "datetime": f"{t0:%Y-%m-%dT%H:%M:%SZ}/{t1:%Y-%m-%dT%H:%M:%SZ}"}
        errors = []
        for url in ([self.catalog_url] if self.catalog_url else self.CATALOG_URLS):
            r = self._post(url, body)
            if r.status_code == 200:
                self.catalog_url = url
                break
            errors.append(f"{url}: HTTP {r.status_code} {r.text[:200]}")
        else:
            raise RuntimeError("Sentinel-2 catalog: " + " | ".join(errors))
        feats = []
        while True:
            if r.status_code != 200:
                raise RuntimeError(f"Sentinel-2 catalog: HTTP {r.status_code} {r.text[:300]}")
            j = r.json()
            feats += j.get("features", [])
            nxt = j.get("context", {}).get("next")
            if nxt is None:
                break
            body = dict(body, next=nxt)
            r = self._post(self.catalog_url, body)
        by_day = {}
        for f in feats:
            t = utc(f["properties"]["datetime"])
            cloud = f["properties"].get("eo:cloud_cover")
            if t.date() not in by_day or (cloud is not None and cloud < (by_day[t.date()][1] or 101)):
                by_day[t.date()] = (t, cloud)
        return [by_day[d] for d in sorted(by_day)]

    def fetch(self, collection, evalscript, bounds, epsg, res, day, nbands, dtype, max_px=2000):
        """one day of Sentinel-2 on the grid `bounds` (UTM, EPSG `epsg`) at `res` metres; tiles of max_px stitched"""
        left, bottom, right, top = bounds
        W, H = int(round((right - left) / res)), int(round((top - bottom) / res))
        out = np.zeros((nbands, H, W), dtype)
        for r0 in range(0, H, max_px):
            for c0 in range(0, W, max_px):
                h, w = min(max_px, H - r0), min(max_px, W - c0)
                body = {"input": {"bounds": {"bbox": [left + c0 * res, top - (r0 + h) * res, left + (c0 + w) * res,
                                                      top - r0 * res],
                                             "properties": {"crs": f"http://www.opengis.net/def/crs/EPSG/0/{epsg}"}},
                                  "data": [{"type": collection, "dataFilter": {
                                      "timeRange": {"from": f"{day}T00:00:00Z", "to": f"{day}T23:59:59Z"},
                                      "mosaickingOrder": "leastCC"}}]},
                        "output": {"width": w, "height": h,
                                   "responses": [{"identifier": "default", "format": {"type": "image/tiff"}}]},
                        "evalscript": evalscript}
                r = self._post(self.PROCESS_URL, body)
                if r.status_code != 200 or r.content[:4] not in (b"II*\x00", b"MM\x00*"):
                    raise RuntimeError(f"Sentinel-2 download: HTTP {r.status_code} {r.text[:300]}")
                with rasterio.MemoryFile(r.content) as mf, mf.open() as ds:
                    tile = ds.read()
                if tile.shape != (nbands, h, w):
                    raise RuntimeError(f"Sentinel-2 download: tile of shape {tile.shape}, expected {(nbands, h, w)}")
                out[:, r0:r0 + h, c0:c0 + w] = tile
        return out


class CloudMask:
    """OmniCloudMask, with the processor's own model weights when present"""

    def __init__(self, device):
        try:
            import torch
            from omnicloudmask import predict_from_array
        except ImportError as e:
            raise SystemExit(f"OmniCloudMask is needed ({e}): run this in the processor's environment")
        torch.set_num_threads(max(1, os.cpu_count() or 1))
        self.predict, self.device, self.kw = predict_from_array, device, {}
        model_dir = REPO / "hf_cache" / "omnicloudmask"
        if any(model_dir.glob("*.safetensors")):
            self.kw["destination_model_dir"] = model_dir
        sys.path.insert(0, str(REPO / "src"))
        try:                                                    # the same rule as the processor's CLM
            from iride_eaglet_processor.masks import probs_to_four_class_map
            self.rule = probs_to_four_class_map
            log("clouds: the processor's own rule (masks.probs_to_four_class_map)")
        except Exception:
            self.rule = lambda probs: np.argmax(probs, axis=0).astype("uint8")
            log("!! the processor's masks.py was not found: clouds from the most likely class")

    def classes(self, refl3):
        """0 clear, 1 thick cloud, 2 thin cloud, 3 shadow, 255 no data; refl3 = (3, h, w), NaN = no data"""
        ok = np.isfinite(refl3).all(0)
        x = np.where(ok, refl3, 0).astype("float32")
        try:
            probs = self.predict(x, export_confidence=True, softmax_output=True, inference_device=self.device,
                                 inference_dtype="fp32", **self.kw)
        except TypeError:
            probs = self.predict(x, export_confidence=True, softmax_output=True, inference_device=self.device, **self.kw)
        cls = np.asarray(self.rule(np.asarray(probs)), dtype="uint8")
        cls[~ok] = 255
        return cls


class Lazy:
    """created when first needed, so a run with everything on disk needs no login and no model"""

    def __init__(self, make):
        self.make, self.obj = make, None

    def __call__(self):
        if self.obj is None:
            self.obj = self.make()
        return self.obj


def each_product(prods, what, fn):
    """fn(p) for every product; one failing does not stop the others (a later run retries it)"""
    failed = []
    for p in prods:
        try:
            fn(p)
        except Exception as e:
            failed.append(p["label"])
            log(f"!! {p['label']}: {what} failed: {type(e).__name__}: {e}")
    if failed and len(failed) == len(prods):
        raise SystemExit(f"{what} failed for every product (see the !! lines above)")
    if failed:
        log(f"!! {what} failed for {', '.join(failed)}: run the script again later to retry")


def write_tif(path, data, epsg, tf, nodata=None):
    data = data if data.ndim == 3 else data[None]
    kw = dict(driver="GTiff", width=data.shape[2], height=data.shape[1], count=data.shape[0], dtype=data.dtype,
              crs=f"EPSG:{epsg}", transform=tf, compress="deflate", tiled=True, blockxsize=256, blockysize=256)
    if nodata is not None:
        kw["nodata"] = nodata
    if data.dtype.kind == "u":
        kw["predictor"] = 2
    tmp = path.with_name(path.name + ".part")
    with rasterio.open(tmp, "w", **kw) as dst:
        dst.write(data)
    tmp.replace(path)


# ---------------------------------------------------------------------------------------------------------------
# 2 Sentinel-2 over each footprint
# ---------------------------------------------------------------------------------------------------------------

def s2_pass(p, day, cdse, ocm, folder):
    """one Sentinel-2 day on the 10 m grid: reflectance (4, h, w) R, G, B, NIR and the clear mask; kept on disk"""
    path = folder / f"{day}.tif"
    bounds, w, h, tf = grid_of(p["bounds"])
    if path.exists():
        with rasterio.open(path) as s:
            a = s.read()
    else:
        l1c = cdse().fetch("sentinel-2-l1c", EVAL_L1C, bounds, p["epsg"], RES, day, 5, "uint16")
        l2a = cdse().fetch("sentinel-2-l2a", EVAL_SCL, bounds, p["epsg"], RES, day, 2, "uint8")
        data = (l1c[4] > 0) & (l2a[1] > 0) & (l1c[:4] > 0).all(0)
        refl = np.where(data, l1c[:4] / 10000.0, np.nan).astype("float32")
        cls = ocm().classes(refl[[0, 1, 3]])                       # OCM on red, green, NIR, as it was built for
        cloudy = ndimage.binary_dilation(np.isin(l2a[0], CLOUDY_SCL) | np.isin(cls, (1, 2, 3)),
                                         iterations=CLOUD_BUFFER)
        clear = data & ~cloudy
        a = np.concatenate([l1c[:4], l2a[:1].astype("uint16"), clear[None].astype("uint16")])
        folder.mkdir(parents=True, exist_ok=True)
        write_tif(path, a, p["epsg"], tf)
    clear = a[5] == 1
    return np.where(clear, a[:4] / 10000.0, np.nan).astype("float32"), clear


def s2_stage(prods, work, cdse, ocm, redo):
    folder = work / "s2"
    folder.mkdir(exist_ok=True)
    each_product(prods, "Sentinel-2", lambda p: s2_one(p, folder, cdse, ocm, redo))


def s2_one(p, folder, cdse, ocm, redo):
    comp_path, info_path = folder / f"{p['stem']}_composite.tif", folder / f"{p['stem']}.json"
    if info_path.exists() and not redo:
        info = json.loads(info_path.read_text())
        if info["filled"] >= MIN_FILLED or info.get("window_days", WINDOW_DAYS) >= WINDOW_DAYS_MAX:
            log(f"{p['label']}: Sentinel-2 on disk, {len(info['passes'])} passes, "
                f"{100 * info['filled']:.0f}% of the footprint clear")
            return
    work = folder.parent                                   # a new composite: the pairs made from the old one go
    for old in (work / "align" / f"{p['stem']}.json", work / "pairs" / f"{p['stem']}.npz",
                work / "eag10" / f"{p['stem']}.npz"):
        old.unlink(missing_ok=True)
    bounds, w, h, tf = grid_of(p["bounds"])
    t_e = utc(p["time"])
    S = np.full((4, h, w), np.nan, "float32")
    DT = np.full((h, w), np.nan, "float32")
    cands, used, window = [], [], WINDOW_DAYS
    for window in (WINDOW_DAYS, WINDOW_DAYS_MAX):
        seen = {c["day"] for c in cands}
        for t, tile_cloud in cdse().days(p["bbox_ll"], t_e - timedelta(days=window), t_e + timedelta(days=window)):
            day = t.date().isoformat()
            if day in seen:
                continue
            scl, dm = cdse().fetch("sentinel-2-l2a", EVAL_SCL, bounds, p["epsg"], 60.0, day, 2, "uint8")
            ok = dm > 0
            cands.append(dict(day=day, dt_days=round((t - t_e).total_seconds() / 86400, 2),
                              cover=round(float(ok.mean()), 3),
                              clear=round(float((ok & ~np.isin(scl, CLOUDY_SCL)).mean()), 3)))
        done = {u["day"] for u in used}
        usable = sorted([c for c in cands if c["clear"] >= MIN_PASS_CLEAR and c["day"] not in done],
                        key=lambda c: (abs(c["dt_days"]), -c["clear"]))
        for c in usable:
            if len(used) >= MAX_PASSES or np.isfinite(DT).mean() >= 0.97:
                break
            refl, clear = s2_pass(p, c["day"], cdse, ocm, folder / p["stem"])
            fill = np.isnan(DT) & clear
            S[:, fill], DT[fill] = refl[:, fill], c["dt_days"]
            used.append(dict(c, added=round(float(fill.mean()), 3)))
        if np.isfinite(DT).mean() >= MIN_FILLED:
            break
    filled = float(np.isfinite(DT).mean())
    if used:
        write_tif(comp_path, np.concatenate([S, DT[None]]).astype("float32"), p["epsg"], tf, nodata=np.nan)
    else:
        comp_path.unlink(missing_ok=True)
    info_path.write_text(json.dumps(dict(candidates=cands, passes=used, filled=filled, window_days=window), indent=1))
    log(f"{p['label']}: {len(cands)} Sentinel-2 days within {window} days, used "
        + (", ".join(f"{u['day']} ({u['dt_days']:+.1f} d, +{100 * u['added']:.0f}%)" for u in used) or "none")
        + f": {100 * filled:.0f}% of the footprint clear")


def load_composite(work, p):
    """Sentinel-2 composite of the footprint: (4, h, w) reflectance R, G, B, NIR, and days from the EAGLET image"""
    path = work / "s2" / f"{p['stem']}_composite.tif"
    if not path.exists():
        return None, None
    with rasterio.open(path) as s:
        a = s.read()
    return a[:4], a[4]


# ---------------------------------------------------------------------------------------------------------------
# 3 EAGLET onto Sentinel-2, and the pairs (Cells 8a and 8)
# ---------------------------------------------------------------------------------------------------------------

def to_grid(band, src_tf, crs, w, h, tf, fix=Affine.identity(), how=Resampling.average):
    """an EAGLET band (DN, 0 = no data) averaged into the 10 m cells, after the map correction `fix`"""
    out = np.full((h, w), np.nan, "float32")
    reproject(band, out, src_transform=fix * src_tf, src_crs=crs, src_nodata=0, dst_transform=tf, dst_crs=crs,
              dst_nodata=np.nan, resampling=how)
    return out


def for_features(img):
    """8-bit copy with the local contrast evened out, and where features may be taken (away from the edges)"""
    import cv2
    ok = np.isfinite(img)
    lo, hi = np.percentile(img[ok], [1, 99])
    u8 = np.where(ok, np.clip((img - lo) / (hi - lo + 1e-9) * 255, 0, 255), 0).astype("uint8")
    u8 = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(16, 16)).apply(u8)
    return u8, ndimage.binary_erosion(ok, iterations=5).astype("uint8") * 255


def fit_geometry(e_img, s_img):
    """affine taking each EAGLET cell to the Sentinel-2 cell with the same ground, from features seen in both
    (also tried with EAGLET mirrored top to bottom)"""
    import cv2
    finder, norm = cv2.SIFT_create(nfeatures=20000), cv2.NORM_L2
    ks, ds = finder.detectAndCompute(*for_features(s_img))
    best = None
    for mirrored in (False, True):
        if mirrored and best is not None and best["n"] >= 100:
            break
        e = e_img[::-1] if mirrored else e_img
        ke, de = finder.detectAndCompute(*for_features(e))
        if de is None or ds is None or len(ke) < MIN_FEATURES or len(ks) < MIN_FEATURES:
            continue
        good = [m[0] for m in cv2.BFMatcher(norm).knnMatch(de, ds, k=2)
                if len(m) == 2 and m[0].distance < 0.8 * m[1].distance]
        if len(good) < MIN_FEATURES:
            continue
        pe = np.float32([ke[m.queryIdx].pt for m in good])
        ps = np.float32([ks[m.trainIdx].pt for m in good])
        if mirrored:
            pe[:, 1] = e_img.shape[0] - 1 - pe[:, 1]
        M, inl = cv2.estimateAffine2D(pe, ps, method=cv2.RANSAC, ransacReprojThreshold=1.5, maxIters=20000,
                                      confidence=0.999, refineIters=20)
        if M is None:
            continue
        inl = inl.ravel().astype(bool)
        miss = np.hypot(*(pe[inl] @ M[:, :2].T + M[:, 2] - ps[inl]).T)
        cand = dict(M=M, n=int(inl.sum()), rms=float(np.sqrt(np.mean(miss ** 2))) * RES, mirrored=mirrored)
        if best is None or cand["n"] > best["n"]:
            best = cand
    return best


def describe(M, h, w):
    """the fit in words: where EAGLET puts the scene centre vs Sentinel-2 (m), turn (deg), size vs the ground (%)"""
    A, t = M[:, :2], M[:, 2]
    c = np.array([w / 2, h / 2])
    move = A @ c + t - c
    inv = np.linalg.inv(A)
    return (-move[0] * RES, move[1] * RES, float(np.degrees(np.arctan2(inv[1, 0], inv[0, 0]))),
            (np.hypot(*inv[:, 0]) - 1) * 100, (np.hypot(*inv[:, 1]) - 1) * 100)


def edge_map(img):
    """edge strength scaled the same way for both sensors; 0 where there is no data (and 3 cells around it)"""
    ok = np.isfinite(img)
    if ok.sum() < 100:
        return np.zeros(img.shape)
    lo, mid, hi = np.percentile(img[ok], [10, 50, 90])
    z = np.where(ok, (img - mid) / (hi - lo + 1e-9), 0.0)
    g = np.hypot(ndimage.sobel(z, 0), ndimage.sobel(z, 1))
    return np.where(ndimage.binary_erosion(ok, iterations=3), g, 0.0)


def phase_shift(a, b):
    """shift (rows, cols) between two images of the same size, and how strong the match is (peak / median)"""
    F = np.fft.fft2(a) * np.conj(np.fft.fft2(b))
    s = np.abs(np.fft.ifft2(F / np.maximum(np.abs(F), 1e-12)))
    i, j = np.unravel_index(np.argmax(s), s.shape)
    out = []
    for axis, (k, n) in enumerate(((i, s.shape[0]), (j, s.shape[1]))):
        m1 = s[(i - 1) % n, j] if axis == 0 else s[i, (j - 1) % n]
        p1 = s[(i + 1) % n, j] if axis == 0 else s[i, (j + 1) % n]
        den = m1 - 2 * s[i, j] + p1
        k = k + (0.5 * (m1 - p1) / den if abs(den) > 1e-12 else 0.0)
        out.append(k - n if k > n / 2 else k)
    return out[0], out[1], float(s[i, j] / max(np.median(s), 1e-12))


def tile_check(e_img, s_img):
    """what is left after the fit, in 2 km tiles: centre row, col, shift rows, cols (cells), match strength"""
    h, w = e_img.shape
    found = []
    for y0 in range(0, h - TILE + 1, TILE // 2):
        for x0 in range(0, w - TILE + 1, TILE // 2):
            e, s = e_img[y0:y0 + TILE, x0:x0 + TILE], s_img[y0:y0 + TILE, x0:x0 + TILE]
            if np.isfinite(e).mean() < 0.9 or np.isfinite(s).mean() < 0.5:
                continue
            a, b = edge_map(e), edge_map(s)
            if not a.any() or not b.any():
                continue
            dy, dx, strength = phase_shift(a, b)
            found.append((y0 + TILE / 2, x0 + TILE / 2, dy, dx, strength))
    return np.array(found, "float64").reshape(-1, 5)


def rank_corr(e_img, s_img):
    """how well two images agree, cell by cell (1 = perfectly), by rank, away from EAGLET's edge"""
    ok = ndimage.binary_erosion(np.isfinite(e_img), iterations=3) & np.isfinite(s_img)
    if ok.sum() < 1000:
        return float("nan")
    a, b = e_img[ok], s_img[ok]
    if a.size > 300_000:
        i = np.random.default_rng(0).choice(a.size, 300_000, replace=False)
        a, b = a[i], b[i]
    return float(np.corrcoef(a.argsort().argsort(), b.argsort().argsort())[0, 1])


def s2_spread(img):
    """relative std of each Sentinel-2 cell's 3 x 3 neighbourhood, worst band (inf where a neighbour is missing)"""
    ok = np.isfinite(img).all(0)
    out = np.zeros(img.shape[1:], "float32")
    for band in img:
        x = np.where(ok, band, 0).astype("float64")
        m = ndimage.uniform_filter(x, 3)
        m2 = ndimage.uniform_filter(x * x, 3)
        out = np.maximum(out, np.sqrt(np.clip(m2 - m * m, 0, None)) / np.maximum(m, 1e-6))
    full = ndimage.uniform_filter(ok.astype("float32"), 3) > 0.999
    return np.where(full, out, np.inf)


def align(p, dn, src_tf, crs, s2g, w, h, tf):
    """the map correction for EAGLET (an Affine on map coordinates) and what to report; None when not trusted"""
    before = to_grid(dn[1], src_tf, crs, w, h, tf)
    fit = fit_geometry(before, s2g) if np.isfinite(s2g).sum() > 10_000 else None
    pre = Affine.identity()
    if np.isfinite(s2g).sum() > 10_000 and (fit is None or fit["n"] < 50):     # too few features: squashed image?
        cx, cy = (p["bounds"][0] + p["bounds"][2]) / 2, (p["bounds"][1] + p["bounds"][3]) / 2
        for sx, sy in STRETCHES:
            q = Affine.translation(cx, cy) * Affine.scale(sx, sy) * Affine.translation(-cx, -cy)
            f2 = fit_geometry(to_grid(dn[1], src_tf, crs, w, h, tf, q), s2g)
            if f2 is not None and f2["n"] >= 100 and (fit is None or f2["n"] > fit["n"]):
                fit, pre = f2, q
    if fit is None:
        return None, dict(note="no geometry fit (too few features)")
    in_cells = Affine.translation(0.5, 0.5) * Affine(*fit["M"].ravel()) * Affine.translation(-0.5, -0.5)
    fix = tf * in_cells * ~tf * pre
    east, north, turn, size_x, size_y = describe(np.array(~tf * fix * tf)[:6].reshape(2, 3), h, w)
    after = to_grid(dn[1], src_tf, crs, w, h, tf, fix)
    t = tile_check(after, s2g)
    t = t[t[:, 4] >= MIN_MATCH]
    dev = np.hypot(t[:, 2], t[:, 3]) * RES
    c0, c1 = rank_corr(before, s2g), rank_corr(after, s2g)
    factors = (1 + size_x / 100, 1 + size_y / 100)
    plausible = abs(turn) < 10 and all(min(abs(f / g - 1) for g in (1, 0.5, 2)) < 0.15 for f in factors)
    strong = fit["n"] >= 100 and fit["rms"] <= 6
    trusted = (fit["n"] >= MIN_FEATURES and fit["rms"] <= 10 and plausible and c1 >= c0 - 0.02
               and (c1 >= 0.4 or (strong and c1 >= 0.3)))
    report = dict(east_m=round(east), north_m=round(north), turn_deg=round(turn, 2), size_x_pct=round(size_x, 2),
                  size_y_pct=round(size_y, 2), mirrored=bool(fit["mirrored"]), features=fit["n"],
                  fit_rms_m=round(fit["rms"], 1), corr_before=round(c0, 2), corr_after=round(c1, 2), tiles=len(t),
                  tiles_off_10m=int((dev > MAX_LEFTOVER_M).sum()),
                  tiles_median_m=round(float(np.median(dev)), 1) if len(t) else None,
                  trusted=bool(trusted), note="ok" if trusted else "NOT trusted, left out")
    return (fix if trusted else None), dict(report, tiles_rc=t[:, :2].tolist(), tiles_dev=dev.tolist())


def pairs_stage(prods, work, ocm, redo):
    for sub in ("pairs", "eag10", "align"):
        (work / sub).mkdir(exist_ok=True)
    resamp_rms = getattr(Resampling, "rms", None)
    each_product(prods, "alignment and pairs", lambda p: pairs_one(p, work, ocm, redo, resamp_rms))


def pairs_one(p, work, ocm, redo, resamp_rms):
    out = work / "pairs" / f"{p['stem']}.npz"
    rep_path = work / "align" / f"{p['stem']}.json"
    if not p.get("calibration_ok", True):
        out.unlink(missing_ok=True)
        rep_path.write_text(json.dumps(dict(note="calibration looks wrong, left out", pairs=0)))
        log(f"{p['label']}: calibration looks wrong, no pairs")
        return
    if rep_path.exists() and not redo:
        rep = json.loads(rep_path.read_text())
        log(f"{p['label']}: alignment and pairs on disk ({rep.get('pairs', 0)} pairs, {rep.get('note')})")
        return
    s2, dt = load_composite(work, p)
    if s2 is None:
        rep_path.write_text(json.dumps(dict(note="no clear Sentinel-2, left out", pairs=0)))
        log(f"{p['label']}: no clear Sentinel-2, left out")
        return
    bounds, w, h, tf = grid_of(p["bounds"])
    t0 = time.time()
    with rasterio.open(p["tif"]) as src:
        dn, src_tf, crs = src.read(), src.transform, src.crs
    fix, rep = align(p, dn, src_tf, crs, s2[1], w, h, tf)
    if fix is None:
        rep["pairs"] = 0
        rep_path.write_text(json.dumps(rep))
        log(f"{p['label']}: {rep['note']} {json.dumps({k: v for k, v in rep.items() if not k.startswith('tiles_')})}")
        return

    # EAGLET averaged into the Sentinel-2 cells after the fit, and how much it varies inside each cell
    mean = np.stack([to_grid(dn[b], src_tf, crs, w, h, tf, fix) for b in range(3)])
    if resamp_rms is not None:
        rms = np.stack([to_grid(dn[b], src_tf, crs, w, h, tf, fix, resamp_rms) for b in range(3)])
        msq = rms * rms
    else:
        msq = np.stack([to_grid(dn[b].astype("float32") ** 2, src_tf, crs, w, h, tf, fix) for b in range(3)])
    del dn
    spread = np.sqrt(np.clip(msq - mean * mean, 0, None)) / mean
    gains, offs = np.array(p["gains"])[:, None, None], np.array(p["offsets"])[:, None, None]
    refl = mean * gains + offs                                         # as delivered
    cls = ocm().classes(refl)                                          # EAGLET's own clouds, as the processor
    cloudy = ndimage.binary_dilation(np.isin(cls, (1, 2, 3)), iterations=CLOUD_BUFFER)
    np.savez_compressed(work / "eag10" / f"{p['stem']}.npz", refl=refl.astype("float32"), cls=cls,
                        fix=np.array(tuple(fix)[:6]))

    inner = ndimage.binary_erosion(np.isfinite(mean).all(0), iterations=2)
    clear = np.isfinite(s2).all(0) & (np.nan_to_num(s2) > 0).all(0)
    valid = np.isfinite(mean).all(0) & (np.nan_to_num(mean, nan=65535) < 60000).all(0)
    e_max = np.where(np.isfinite(spread).all(0), np.nan_to_num(spread).max(0), np.inf)
    uniform = (e_max < MAX_SPREAD) & (s2_spread(s2[:3]) < MAX_SPREAD)
    use = inner & clear & valid & uniform & ~cloudy
    rows_, cols_ = np.nonzero(use)
    left_out = 0
    dev = np.array(rep["tiles_dev"])
    if len(dev) and len(rows_):
        dist, idx = cKDTree(np.array(rep["tiles_rc"])).query(np.column_stack([rows_, cols_]))
        bad = (dist <= TILE) & (dev[idx] > MAX_LEFTOVER_M)
        left_out = int(bad.sum())
        rows_, cols_ = rows_[~bad], cols_[~bad]
    if len(rows_) > MAX_PAIRS:
        i = rng.choice(len(rows_), MAX_PAIRS, replace=False)
        rows_, cols_ = rows_[i], cols_[i]
    e_dn = mean[:, rows_, cols_].T
    np.savez_compressed(out, e_dn=e_dn.astype("float32"), e_refl=refl[:, rows_, cols_].T.astype("float32"),
                        s2=s2[:, rows_, cols_].T.astype("float32"), dt=dt[rows_, cols_].astype("float32"),
                        row=rows_.astype("int32"), col=cols_.astype("int32"))
    n_in = max(int(inner.sum()), 1)
    rep.update(pairs=int(len(rows_)), left_out_near_bad_tiles=left_out,
               s2_clear_pct=round(100 * float((inner & clear).sum()) / n_in, 1),
               eaglet_cloud_pct=round(100 * float((inner & np.isin(cls, (1, 2, 3))).sum()) / n_in, 1),
               usable_pct=round(100 * float(use.sum()) / n_in, 1), seconds=round(time.time() - t0))
    rep_path.write_text(json.dumps(rep))
    log(f"{p['label']}: EAGLET was {abs(rep['east_m'])} m {'east' if rep['east_m'] >= 0 else 'west'}, "
        f"{abs(rep['north_m'])} m {'north' if rep['north_m'] >= 0 else 'south'} of Sentinel-2, turned "
        f"{rep['turn_deg']:+.2f} deg "
        f"({rep['features']} features within {rep['fit_rms_m']} m, agreement {rep['corr_before']} -> "
        f"{rep['corr_after']}); EAGLET cloud {rep['eaglet_cloud_pct']}%, Sentinel-2 clear {rep['s2_clear_pct']}%, "
        f"{rep['pairs']} pairs ({rep['seconds']} s)")



# ---------------------------------------------------------------------------------------------------------------
# 4 the world chips (Cell 10b)
# ---------------------------------------------------------------------------------------------------------------

PLACES = [
    ("Hoggar mountains, Algeria", "desert", 23.25, 5.65), ("Great Sand Sea, Egypt", "desert", 26.5, 25.5),
    ("Hajar mountains, Oman", "desert", 23.1, 57.5), ("Atacama, Chile", "desert", -23.8, -69.2),
    ("Namib gravel plains, Namibia", "desert", -23.3, 15.3), ("Mojave, USA", "desert", 34.9, -115.7),
    ("Central plateau, Iran", "desert", 31.6, 55.4), ("Red Sea hills, Sudan", "desert", 19.6, 36.6),
    ("Red sand near Uluru, Australia", "desert", -25.2, 131.2), ("Thar, India", "desert", 26.9, 70.6),
    ("Gobi, Mongolia", "desert", 43.6, 103.8), ("Negev, Israel", "desert", 30.6, 34.8),
    ("Sahel, Niger", "savanna/shrub", 14.0, 3.0), ("Maasai Mara, Kenya", "savanna/shrub", -1.5, 35.1),
    ("Cerrado, Brazil", "savanna/shrub", -14.5, -46.5), ("Dehesa, Spain", "savanna/shrub", 39.1, -6.2),
    ("Kalahari, Botswana", "savanna/shrub", -22.0, 23.8), ("Top End, Australia", "savanna/shrub", -14.5, 132.3),
    ("Karoo, South Africa", "savanna/shrub", -32.3, 22.5), ("Crete hills, Greece", "savanna/shrub", 35.2, 24.9),
    ("Sardinia hills, Italy", "savanna/shrub", 40.3, 9.3), ("Kufra pivots, Libya", "irrigated", 24.2, 23.3),
    ("Kansas pivots, USA", "irrigated", 37.4, -100.9), ("Nile delta, Egypt", "irrigated", 30.9, 31.0),
    ("Imperial Valley, USA", "irrigated", 32.85, -115.5), ("Central Valley, USA", "irrigated", 36.6, -120.2),
    ("Po Valley, Italy", "farmland", 45.0, 9.9), ("Tavoliere, Italy", "farmland", 41.5, 15.7),
    ("Sicily interior, Italy", "farmland", 37.6, 14.3), ("Beauce, France", "farmland", 48.3, 1.6),
    ("Central Ukraine", "farmland", 49.4, 32.8), ("Iowa, USA", "farmland", 42.0, -93.6),
    ("Pampas, Argentina", "farmland", -34.5, -61.0), ("Olive groves, Jaen, Spain", "farmland", 37.8, -3.9),
    ("Black Forest, Germany", "forest", 48.1, 8.2), ("Amazon, Para, Brazil", "forest", -6.5, -55.5),
    ("Boreal forest, Canada", "forest", 54.5, -100.5), ("Trentino Alps, Italy", "forest", 46.2, 11.3),
    ("Abruzzo Apennines, Italy", "forest", 41.9, 13.9), ("Kalimantan, Indonesia", "forest", -1.5, 113.5),
    ("Rome, Italy", "city", 41.89, 12.49), ("Milan, Italy", "city", 45.47, 9.19), ("Cairo, Egypt", "city", 30.05, 31.24),
    ("Dubai, UAE", "city", 25.2, 55.28), ("Phoenix, USA", "city", 33.45, -112.07), ("Madrid, Spain", "city", 40.42, -3.70),
    ("Johannesburg, South Africa", "city", -26.2, 28.04), ("Tehran, Iran", "city", 35.7, 51.4),
    ("Venice lagoon, Italy", "wetland", 45.43, 12.33), ("Camargue, France", "wetland", 43.55, 4.55),
    ("Flevoland, Netherlands", "wetland", 52.5, 5.5), ("Oristano lagoons, Italy", "wetland", 39.9, 8.55),
]
CHIP_KM, CHIP_WINDOW, CHIP_MAX_TILE_CLOUD, CHIP_TRIES, CHIP_MIN_CLEAR = 10, ("2026-06-01", "2026-09-30"), 30, 3, 0.5


def chips_stage(work, chip_dir, cdse, ocm):
    listing = chip_dir / "chips.csv"
    if listing.exists():
        chips = [c for c in csv.DictReader(listing.open()) if (chip_dir / c["file"]).exists()]
        if chips:
            log(f"{len(chips)} world chips on disk in {chip_dir}")
            return
    chip_dir.mkdir(parents=True, exist_ok=True)
    rows = []
    for name, kind, lat, lon in PLACES:
        file = "".join(ch if ch.isalnum() else "_" for ch in name.lower()) + ".tif"
        path = chip_dir / file
        epsg = (32600 if lat >= 0 else 32700) + int((lon + 180) // 6) + 1
        (x,), (y,) = warp_transform("EPSG:4326", f"EPSG:{epsg}", [lon], [lat])
        left, bottom = math.floor((x - CHIP_KM * 500) / RES) * RES, math.floor((y - CHIP_KM * 500) / RES) * RES
        bounds = (left, bottom, left + CHIP_KM * 1000, bottom + CHIP_KM * 1000)
        _, w, h, tf = grid_of(bounds)
        day = None
        if not path.exists():
            t0 = datetime.fromisoformat(CHIP_WINDOW[0]).replace(tzinfo=timezone.utc)
            t1 = datetime.fromisoformat(CHIP_WINDOW[1]).replace(tzinfo=timezone.utc) + timedelta(hours=23, minutes=59)
            days = [(t, c) for t, c in cdse().days((lon - 0.05, lat - 0.05, lon + 0.05, lat + 0.05), t0, t1)
                    if c is not None and c <= CHIP_MAX_TILE_CLOUD]
            for t, cloud in sorted(days, key=lambda tc: tc[1])[:CHIP_TRIES]:
                d = t.date().isoformat()
                try:
                    a = cdse().fetch("sentinel-2-l1c", EVAL_L1C, bounds, epsg, RES, d, 5, "uint16")
                    l2a = cdse().fetch("sentinel-2-l2a", EVAL_SCL, bounds, epsg, RES, d, 2, "uint8")
                except Exception as e:
                    log(f"   !! {name} {d}: {e}")
                    continue
                data = (a[4] == 1) & (l2a[1] == 1) & (a[:4] > 0).all(0)
                if data.mean() < CHIP_MIN_CLEAR:
                    continue
                cls = ocm().classes(np.where(data, a[[0, 1, 3]] / 10000.0, np.nan).astype("float32"))
                cloudy = ndimage.binary_dilation(np.isin(cls, (1, 2, 3)) | np.isin(l2a[0], CLOUDY_SCL),
                                                 iterations=CLOUD_BUFFER)
                clear = data & ~cloudy
                if clear.mean() >= CHIP_MIN_CLEAR:
                    write_tif(path, np.concatenate([a[:4], clear[None]]).astype("uint16"), epsg, tf)
                    day = d
                    break
            if day is None:
                log(f"   {name}: no clear day found, skipped")
                continue
        rows.append(dict(file=file, place=name, kind=kind, lat=lat, lon=lon, epsg=epsg, day=day or "?"))
        log(f"   chip {name} ({kind}) ready")
    with listing.open("w", newline="") as f:
        wr = csv.DictWriter(f, fieldnames=["file", "place", "kind", "lat", "lon", "epsg", "day"])
        wr.writeheader()
        wr.writerows(rows)
    log(f"{len(rows)} world chips in {chip_dir}")


# ---------------------------------------------------------------------------------------------------------------
# the colour AUX file
# ---------------------------------------------------------------------------------------------------------------

def local(tag):
    return tag.split("}")[-1] if isinstance(tag, str) else ""


def first(root, name):
    return next((el for el in root.iter() if local(el.tag) == name), None)


def decode_table(el):
    raw = base64.b64decode(el.text or "")
    if hashlib.sha256(raw).hexdigest() != (el.get("sha256") or "").strip().lower():
        raise ValueError(f"table {el.get('name')}: checksum mismatch")
    return np.frombuffer(raw, "<f4").reshape([int(s) for s in el.get("shape").split()]).astype("float64")


def read_aux(path):
    """what the LUT step needs from a colour AUX file: axes, vegetation window, balance, shares, tables"""
    root = ET.fromstring(Path(path).read_bytes())
    cc = first(root, "Colour_Correction")
    if cc is None:
        raise SystemExit(f"{path.name}: no <Colour_Correction>")
    axes = {}
    for el in cc.iter():
        if local(el.tag) == "Axis":
            axes[el.get("name")] = (float(el.get("start")), float(el.get("step")), int(el.get("count")))
    around, bal, shares = first(cc, "Vegetation_Around"), first(cc, "Colour_Balance"), first(cc, "NIR_Shares")
    tables = {el.get("name"): decode_table(el) for el in cc.iter() if local(el.tag) == "Table"}
    need = ["veg_share", "q_band1", "q_band2", "q_band3"]
    if sorted(axes) != ["v", "x", "y"] or around is None or bal is None or any(t not in tables for t in need):
        raise SystemExit(f"{path.name}: expected axes x, y, v, Vegetation_Around, Colour_Balance and the tables "
                         f"{', '.join(need)}; found axes {sorted(axes)} and tables {sorted(tables)}")
    method = first(cc, "Method")
    return dict(path=Path(path), axes=axes, cell_m=float(around.get("cell_m")),
                window=int(around.get("window_cells")),
                balance=np.array([float(bal.get(f"band{b}")) for b in (1, 2, 3)]),
                shares=None if shares is None else np.array([float(shares.get(f"band{b}")) for b in (1, 2, 3)]),
                veg_share=tables["veg_share"], q=np.stack([tables[f"q_band{b}"] for b in (1, 2, 3)]),
                method=None if method is None else (method.text or "").strip(),
                version=cc.get("version"))


def newest_aux(folder):
    found = sorted(Path(folder).glob("IME___AXE1D_CLUT_*.xml"), key=lambda p: p.name)
    if not found:
        raise SystemExit(f"no colour AUX file (IME___AXE1D_CLUT_*.xml) in {folder}")
    return found[-1]


# ---------------------------------------------------------------------------------------------------------------
# the table (Cells 11a, 11b, 13, 12 v2)
# ---------------------------------------------------------------------------------------------------------------

def robust_fit(X, y, n_iter=4, k=3.0):
    """least squares that ignores outliers (beyond 3 x the typical residual): coefficients (last = constant), R2"""
    A = np.column_stack([X, np.ones(len(y))])
    keep = np.ones(len(y), bool)
    for _ in range(n_iter):
        coef = np.linalg.lstsq(A[keep], y[keep], rcond=None)[0]
        res = y - A @ coef
        mad = 1.4826 * np.median(np.abs(res[keep] - np.median(res[keep])))
        keep = np.abs(res) < k * max(mad, 1e-9)
    var = np.var(y[keep])
    return coef, float(1 - np.var(y[keep] - A[keep] @ coef) / var) if var > 0 else 0.0


def ndvi(S):
    return (S[:, 3] - S[:, 0]) / (S[:, 3] + S[:, 0] + 1e-9)


def simulate(S, shares, jitter=0.0):
    """what EAGLET would see, from Sentinel-2 rows (R, G, B, NIR): band k = (1 - share_k) x band k + share_k x NIR"""
    s = np.tile(np.asarray(shares, "float64"), (len(S), 1))
    if jitter:
        s = s * (1 + rng.uniform(-jitter, jitter, (len(S), 1)))
    return (1 - s) * S[:, :3] + s * S[:, 3:4]


class Lut:
    """the table layout of the AUX file: x = ln(b1/b2), y = ln(b3/b2), v = vegetation around"""

    def __init__(self, axes, window):
        self.axes, self.window = axes, window
        self.shape = tuple(axes[n][2] for n in ("x", "y", "v"))

    def index(self, E, around=None):
        """fractional table position of each colour (rows of 3 bands), clamped to the edges"""
        E = np.maximum(E, 1e-4)
        vals = [np.log(E[:, 0] / E[:, 1]), np.log(E[:, 2] / E[:, 1])]
        names = ["x", "y"]
        if around is not None:
            vals.append(np.asarray(around, "float64"))
            names.append("v")
        t = [(v - self.axes[n][0]) / self.axes[n][1] for v, n in zip(vals, names)]
        return np.clip(np.stack(t), 0, np.array([self.axes[n][2] for n in names])[:, None] - 1)

    def around_map(self, img, veg_share):
        """vegetation around each 10 m cell from a (3, h, w) image of balanced bands (NaN = no data), as the processor:
        each cell scores veg_share at its colour, mean over the window, cells with no data left out"""
        valid = np.isfinite(img).all(0) & (np.nan_to_num(img) > 0).all(0)
        p = np.zeros(valid.shape)
        if valid.any():
            p[valid] = ndimage.map_coordinates(veg_share, self.index(img[:, valid].T), order=1, mode="nearest")
        num = ndimage.uniform_filter(p, self.window, mode="constant")
        den = ndimage.uniform_filter(valid.astype("float64"), self.window, mode="constant")
        return np.clip(np.where(den > 1e-6, num / np.maximum(den, 1e-6), 0.0), 0, 1)

    def apply(self, q, E, around):
        """each balanced band x its q (linear look-up); no data and values <= 0 pass unchanged"""
        out = E.copy()
        ok = np.isfinite(E).all(1) & (np.nan_to_num(E) > 0).all(1) & np.isfinite(around)
        t = self.index(E[ok], around[ok])
        for b in range(3):
            out[ok, b] = E[ok, b] * ndimage.map_coordinates(q[b], t, order=1, mode="nearest")
        return out

    def sums(self, E, S, veg, around):
        """per table cell: pixel count, sums of Sentinel-2 and of fake EAGLET per band, vegetation count"""
        flat = np.ravel_multi_index(np.rint(self.index(E, around)).astype(int), self.shape)
        size = int(np.prod(self.shape))
        count = lambda wt=None: np.bincount(flat, weights=wt, minlength=size).reshape(self.shape)
        return dict(n=count(), num=np.stack([count(S[:, b]) for b in range(3)]),
                    den=np.stack([count(E[:, b]) for b in range(3)]), veg=count(veg.astype("float64")))

    def veg_counts(self, E, veg):
        flat = np.ravel_multi_index(np.rint(self.index(E)).astype(int), self.shape[:2])
        size = int(np.prod(self.shape[:2]))
        return (np.bincount(flat, minlength=size).reshape(self.shape[:2]).astype("float64"),
                np.bincount(flat, weights=veg.astype("float64"), minlength=size).reshape(self.shape[:2]))

    def finish(self, sums):
        """the tables from the training sums (Cell 13 finish_free): q = Sentinel-2 / fake EAGLET per cell, smoothed;
        cells with little data lean on the colour-only value, cells with none give q = 1 (unchanged)"""
        sm2 = lambda a: ndimage.gaussian_filter(a, SMOOTH, mode="constant")
        sm3 = lambda a: ndimage.gaussian_filter(a, (SMOOTH, SMOOTH, SMOOTH / 2), mode="constant")
        n2 = sm2(sums["n"].sum(2))
        w2 = n2 / (n2 + K_SUPPORT)
        n3 = sm3(sums["n"])
        w3 = n3 / (n3 + K_SUPPORT)
        q = np.ones((3,) + self.shape)
        for b in range(3):
            qc = w2 * sm2(sums["num"][b].sum(2)) / np.maximum(sm2(sums["den"][b].sum(2)), 1e-12) + (1 - w2)
            q[b] = w3 * sm3(sums["num"][b]) / np.maximum(sm3(sums["den"][b]), 1e-12) + (1 - w3) * qc[..., None]
        return np.clip(q, 0.05, 20.0), n3


def load_pairs(work, p):
    path = work / "pairs" / f"{p['stem']}.npz"
    if not path.exists():
        return None
    d = dict(np.load(path))
    return d if len(d["e_dn"]) else None


def fit_stage(prods, work, chip_dir, tpl):
    lut = Lut(tpl["axes"], tpl["window"])
    if abs(tpl["cell_m"] - RES) > 1e-6:
        raise SystemExit(f"the AUX file measures vegetation around in {tpl['cell_m']} m cells; this tool uses {RES} m")
    data = {p["stem"]: d for p in prods if p.get("calibration_ok", True) and (d := load_pairs(work, p)) is not None}
    if len(data) < 3:
        raise SystemExit(f"only {len(data)} products have pairs: at least 3 are needed (see the pairs step)")
    by_stem = {p["stem"]: p for p in prods}

    # near-infrared share of each band: DN = a x its own Sentinel-2 band + n x NIR + c, share = n / (a + n)
    rows, sh_all, good = [], {}, []
    for stem, d in data.items():
        dn, s2 = d["e_dn"].astype("float64"), d["s2"].astype("float64")
        shares, r2 = [], []
        for b in range(3):
            (a_, n_, _), q2 = robust_fit(s2[:, [b, 3]], dn[:, b])
            shares.append(n_ / (a_ + n_) if abs(a_ + n_) > 1e-12 else np.nan)
            r2.append(q2)
        sh_all[stem] = np.array(shares)
        if r2[1] >= MIN_R2_SHARES and np.isfinite(shares).all():
            good.append(stem)
        rows.append(dict(product=by_stem[stem]["label"], pairs=len(dn), share_b1=round(shares[0], 3),
                         share_b2=round(shares[1], 3), share_b3=round(shares[2], 3), R2_green=round(r2[1], 2)))
    if len(good) < 2:
        raise SystemExit("fewer than 2 products fit well enough to measure the near-infrared shares:\n"
                         + table_text(rows))
    shares = np.clip(np.median([sh_all[s] for s in good], axis=0), -0.2, 0.9)
    shares[np.abs(shares) < MIN_SHARE] = 0.0

    # colour balance of the delivered calibration: delivered = scale_k x (fake EAGLET_k + off_k), balance = scale_k /
    # scale_2; off_k is what the delivered band adds on top of Sentinel-2 (haze, or an offset in the calibration)
    bal_all, bal_good = {}, []
    for row, (stem, d) in zip(rows, data.items()):
        X, E = d["e_refl"].astype("float64"), simulate(d["s2"].astype("float64"), shares)
        fits = [robust_fit(E[:, [b]], X[:, b]) for b in range(3)]
        scale = np.array([f[0][0] for f in fits])
        off = [round(f[0][1] / f[0][0], 3) if f[0][0] > 0 else float("nan") for f in fits]
        bal_all[stem] = scale / scale[1]
        worst = min(f[1] for f in fits)
        row.update(brightness_x=round(scale[1], 2), b1_over_g=round(scale[0] / scale[1], 3),
                   b3_over_g=round(scale[2] / scale[1], 3), R2_worst=round(worst, 2),
                   off_b1=off[0], off_b2=off[1], off_b3=off[2])
        plausible = (scale > 0).all() and 0.2 <= scale[1] <= 10
        if not plausible:
            row["R2_worst"] = -1.0                       # never picked for the balance, not even as a fallback
        elif worst >= BAL_MIN_R2:
            bal_good.append(stem)
    if len(bal_good) < 3:
        r2_of = {s: next(r["R2_worst"] for r in rows if r["product"] == by_stem[s]["label"]) for s in data}
        bal_good = [s for s in sorted(data, key=lambda s: -r2_of[s]) if r2_of[s] > -1][:3]
        if not bal_good:
            raise SystemExit("no product with a plausible delivered calibration for the colour balance:\n"
                             + table_text(rows))
        log(f"!! fewer than 3 products fit well enough (R2 >= {BAL_MIN_R2}) for the balance: the best are used")
    balance = np.array([np.median([bal_all[s][0] for s in bal_good]), 1.0, np.median([bal_all[s][2] for s in bal_good])])
    offsets = [round(float(np.nanmedian([r[f"off_b{b}"] for r, s_ in zip(rows, data) if s_ in bal_good])), 3)
               for b in (1, 2, 3)]
    for row, stem in zip(rows, data):
        row["used_for"] = ("shares " if stem in good else "") + ("balance" if stem in bal_good else "")
    # the balance of each satellite (A01 calibrations may differ between satellites): products that fit well
    prod_ok = {stem: row["R2_worst"] >= MIN_R2_SHARES for row, stem in zip(rows, data)}
    sat_rows = []
    for sat in sorted({by_stem[s_]["sat"] for s_ in data}):
        mine = [s_ for s_ in data if by_stem[s_]["sat"] == sat and prod_ok[s_]]
        if mine:
            b = np.median([bal_all[s_] for s_ in mine], axis=0)
            sat_rows.append(dict(satellite=sat, products=len(mine), b1_over_g=round(float(b[0]), 3),
                                 b3_over_g=round(float(b[2]), 3)))

    # training material: Sentinel-2 alone (world chips and the footprints), made into what EAGLET's bands see
    sources = []
    listing = chip_dir / "chips.csv"
    if listing.exists():
        for c in csv.DictReader(listing.open()):
            if (chip_dir / c["file"]).exists():
                sources.append(dict(id=c["place"], kind=c["kind"], path=str(chip_dir / c["file"]), stem=None))
    for p in prods:
        if (work / "s2" / f"{p['stem']}_composite.tif").exists():
            sources.append(dict(id=f"{p['label']} footprint", kind="EAGLET footprint", path=None, stem=p["stem"]))
    if not any(s["kind"] != "EAGLET footprint" for s in sources):
        log("!! no world chips: the table learns from the EAGLET footprints only (see the chips step)")

    def load_source(s):
        if s["path"]:
            with rasterio.open(s["path"]) as src:
                a = src.read()
            return a[:4].astype("float32") / 10000.0, (a[4] == 1) & (a[:4] > 0).all(0)
        S, _ = load_composite(work, by_stem[s["stem"]])
        return S, np.isfinite(S).all(0) & (np.nan_to_num(S) > 0).all(0)

    # 1) how likely each colour is to be vegetation (the processor measures "vegetation around" with it)
    n_xy, veg_xy = np.zeros(lut.shape[:2]), np.zeros(lut.shape[:2])
    for s in sources:
        S, ok = load_source(s)
        Sv = S[:, ok].T.astype("float64")
        Sv = Sv[rng.choice(len(Sv), VEG_SAMPLE, replace=False)] if len(Sv) > VEG_SAMPLE else Sv
        n_, v_ = lut.veg_counts(simulate(Sv, shares, SHARE_JITTER), ndvi(Sv) > 0.4)
        n_xy += n_
        veg_xy += v_
    sm2 = lambda a: ndimage.gaussian_filter(a, SMOOTH, mode="constant")
    veg_share = np.clip(sm2(veg_xy) / np.maximum(sm2(n_xy), 1e-12), 0, 1)

    # 2) the table sums per source, with "vegetation around" measured on the fake EAGLET image
    keys = ("n", "num", "den", "veg")
    per_source = {k: [] for k in keys}
    log("training material (Sentinel-2 only):")
    for s in sources:
        S, ok = load_source(s)
        Sv = S[:, ok].T.astype("float64")
        Ev = simulate(Sv, shares, SHARE_JITTER)
        img = np.full((3,) + ok.shape, np.nan)
        img[:, ok] = Ev.T
        around = lut.around_map(img, veg_share)[ok]
        i = rng.choice(len(Sv), MAX_PER_SOURCE, replace=False) if len(Sv) > MAX_PER_SOURCE else np.arange(len(Sv))
        sm = lut.sums(Ev[i], Sv[i], ndvi(Sv[i]) > 0.4, around[i])
        for k in keys:
            per_source[k].append(sm[k])
        s["pixels"], s["veg_pct"] = int(len(Sv)), round(100 * float((ndvi(Sv) > 0.4).mean()), 1)
    kinds = {}
    for s in sources:
        kinds.setdefault(s["kind"], []).append(s)
    log(table_text([dict(kind=k, sources=len(v), clear_pixels=sum(s["pixels"] for s in v),
                         vegetation_pct=round(float(np.mean([s["veg_pct"] for s in v])), 1)) for k, v in kinds.items()]))
    total = {k: np.sum(per_source[k], axis=0) for k in keys}
    q, support = lut.finish(total)

    np.savez_compressed(work / "lut_fit.npz", prod_stems=np.array(list(data)),
                        prod_sat=np.array([by_stem[s_]["sat"] for s_ in data]),
                        prod_bal=np.array([bal_all[s_] for s_ in data]),
                        prod_bal_ok=np.array([prod_ok[s_] for s_ in data]),
                        q=q.astype("float32"), veg_share=veg_share.astype("float32"),
                        balance=balance, shares=shares, support=support,
                        source_ids=np.array([s["id"] for s in sources]),
                        source_stems=np.array([s["stem"] or "" for s in sources]),
                        **{f"src_{k}": np.array(per_source[k]) for k in keys})
    fit = dict(products=rows, satellites=sat_rows, shares=shares.tolist(), balance=balance.tolist(), offsets=offsets,
               shares_from=len(good),
               balance_from=len(bal_good), sources=len(sources),
               chips=sum(1 for s in sources if s["kind"] != "EAGLET footprint"),
               table_cells_with_data_pct=round(100 * float((support > K_SUPPORT).mean()), 1),
               q_range=[round(float(q.min()), 3), round(float(q.max()), 3)])
    (work / "fit.json").write_text(json.dumps(fit, indent=1))
    log("\nEACH PRODUCT (share = near-infrared share of the band; brightness_x = delivered brightness / what it "
        "should be; b1_over_g, b3_over_g = colour balance of the delivered calibration; off_b1..3 = what each band "
        "adds on top of Sentinel-2, as reflectance: haze, or an offset in the calibration)")
    log(table_text(rows))
    log("\nCOLOUR BALANCE PER SATELLITE (median of its products that fit well)")
    log(table_text(sat_rows))
    old_sh = tpl["shares"]
    log("\nWHAT CHANGED (current AUX file -> fitted now)")
    log("   near-infrared shares: " + (f"{old_sh[0]:.3f} / {old_sh[1]:.3f} / {old_sh[2]:.3f}" if old_sh is not None
                                         else "(not in the current file)")
        + f" -> {shares[0]:.3f} / {shares[1]:.3f} / {shares[2]:.3f}  (from {len(good)} products)")
    log(f"   colour balance:       {tpl['balance'][0]:.4f} / 1 / {tpl['balance'][2]:.4f} -> "
        f"{balance[0]:.4f} / 1 / {balance[2]:.4f}  (from {len(bal_good)} products)")
    log(f"   table: {len(sources)} sources ({fit['chips']} world chips), {fit['table_cells_with_data_pct']}% of the "
        f"cells backed by data, q from {fit['q_range'][0]} to {fit['q_range'][1]}")
    log(f"   added on top of Sentinel-2 (median offset, reflectance): {' / '.join(f'{v:.3f}' for v in offsets)}"
        + ("  !! the LUT only multiplies the bands, so this stays in the L1D as a veil" if max(abs(v) for v in offsets)
           > 0.02 else ""))
    return fit


def load_fit(work):
    path = work / "lut_fit.npz"
    if not path.exists():
        raise SystemExit("run the fit step first")
    return dict(np.load(path))


# ---------------------------------------------------------------------------------------------------------------
# 6 the new AUX file
# ---------------------------------------------------------------------------------------------------------------

def encode(a):
    raw = np.ascontiguousarray(a, dtype="<f4").tobytes()
    return base64.b64encode(raw).decode("ascii"), hashlib.sha256(raw).hexdigest()


def replace_one(text, pattern, repl, what, optional=False):
    fn = repl if callable(repl) else (lambda m: repl)              # taken literally, never as a regex template
    new, n = re.subn(pattern, fn, text, flags=re.S)
    if n == 0 and optional:
        return text
    if n != 1:
        raise SystemExit(f"the current AUX file has {n} {what}, expected exactly 1: it cannot be used as the template")
    return new


def set_attrs(tag, **values):
    """the start tag `tag` with the given attributes set (replaced when present, added before the end otherwise)"""
    for k, v in values.items():
        if re.search(rf'\s{k}="[^"]*"', tag):
            tag = re.sub(rf'(\s{k}=")[^"]*(")', lambda m: f"{m.group(1)}{v}{m.group(2)}", tag)
        else:
            tag = re.sub(r"\s*(/?>)$", lambda m: f' {k}="{v}"{" " if m.group(1) == "/>" else ""}{m.group(1)}', tag)
    return tag


def aux_stage(work, tpl, fit, baseline, n_products):
    now = datetime.now(timezone.utc)
    m = re.match(r"^(IME___AXE1D_CLUT_\d{8}T\d{6}_\d{8}T\d{6})_\d{8}T\d{6}(_+O_\d{3})$", tpl["path"].stem)
    if not m:
        raise SystemExit(f"unexpected name of the current AUX file: {tpl['path'].name}")
    name = f"{m.group(1)}_{now:%Y%m%dT%H%M%S}{m.group(2)}"
    text = tpl["path"].read_text(encoding="utf-8")
    q, veg_share = fit["q"], fit["veg_share"]
    balance, shares = fit["balance"], fit["shares"]

    text = replace_one(text, r"<File_Name>[^<]*</File_Name>", f"<File_Name>{name}</File_Name>", "<File_Name>")
    text = replace_one(text, r"<Creation_Date>[^<]*</Creation_Date>",
                       f"<Creation_Date>UTC={now:%Y-%m-%dT%H:%M:%S}</Creation_Date>", "<Creation_Date>", optional=True)
    text = replace_one(text, r"<Creator>[^<]*</Creator>", "<Creator>tools/build_lut.py</Creator>", "<Creator>",
                       optional=True)
    text = replace_one(text, r"<Creator_Version>[^<]*</Creator_Version>",
                       f"<Creator_Version>LUT v3 fitted on {baseline}</Creator_Version>", "<Creator_Version>",
                       optional=True)
    text = replace_one(text, r"<Colour_Balance\b[^>]*?/?>",
                       lambda mm: set_attrs(mm.group(0), band1=f"{balance[0]:.4f}", band2=f"{balance[1]:.4f}",
                                            band3=f"{balance[2]:.4f}", products=n_products), "<Colour_Balance>")
    text = replace_one(text, r"<NIR_Shares\b[^>]*?/?>",
                       lambda mm: set_attrs(mm.group(0), band1=f"{shares[0]:.3f}", band2=f"{shares[1]:.3f}",
                                            band3=f"{shares[2]:.3f}"), "<NIR_Shares>", optional=True)
    new_tables = {"veg_share": veg_share, **{f"q_band{b + 1}": q[b] for b in range(3)}}
    for tname, arr in new_tables.items():
        b64, sha = encode(arr)
        shape = " ".join(map(str, arr.shape))
        text = replace_one(text, rf'(<Table\b[^>]*\bname="{tname}"[^>]*>)(.*?)(</Table>)',
                           lambda mm: set_attrs(mm.group(1), shape=shape, sha256=sha) + b64 + mm.group(3),
                           f'<Table name="{tname}">')
    training = (f"Baseline {baseline}: {n_products} EAGLET-2 products (near-infrared shares, colour balance, "
                f"footprints) and {int(fit['source_ids'].size)} Sentinel-2 sources in all")
    text = replace_one(text, r"(<Training_Data\b[^>]*>)(.*?)(</Training_Data>)",
                       lambda mm: set_attrs(mm.group(1), sources=int(fit["source_ids"].size)) + training + mm.group(3),
                       "<Training_Data>", optional=True)

    out_dir = work / "aux"
    out_dir.mkdir(exist_ok=True)
    for old in out_dir.glob("IME___AXE1D_CLUT_*.xml"):
        old.unlink()
    path = out_dir / f"{name}.xml"
    path.write_text(text, encoding="utf-8")

    # read it back: the new numbers are there, everything else is as in the current file
    new_root, old_root = ET.fromstring(path.read_bytes()), ET.fromstring(tpl["path"].read_bytes())
    back = read_aux(path)
    checks = {
        "tables decode (SHA-256 ok), same as fitted": all(np.array_equal(back["q"][b].astype("float32"), q[b].astype("float32"))
                                                          for b in range(3))
        and np.array_equal(back["veg_share"].astype("float32"), veg_share.astype("float32")),
        "balance as fitted": np.allclose(back["balance"], balance, atol=1e-4),
        "axes and window unchanged": back["axes"] == tpl["axes"] and back["window"] == tpl["window"],
        "method unchanged": back["method"] == tpl["method"],
    }
    for el_name in ("Class_Model", "Method", "Validity_Period", "Model", "Input"):
        a, b = first(old_root, el_name), first(new_root, el_name)
        if a is not None or b is not None:
            checks[f"<{el_name}> unchanged"] = (a is not None and b is not None and ET.tostring(a) == ET.tostring(b))
    log(f"written {path.name} ({path.stat().st_size / 1e3:.0f} kB), from {tpl['path'].name}")
    for k, ok in checks.items():
        log(f"   {'ok ' if ok else 'BAD'}  {k}")
    if not all(checks.values()):
        raise SystemExit("the new AUX file does not read back as expected")

    # the processor's own code must accept it
    sys.path.insert(0, str(REPO / "src"))
    try:
        from iride_eaglet_processor.colour_correction import ColourCorrection
        cc = ColourCorrection(path)
        desc = cc.describe() if hasattr(cc, "describe") else f"method {getattr(cc, 'method', '?')}"
        log(f"   ok   loaded with the processor's ColourCorrection: {desc}")
    except Exception as e:
        raise SystemExit(f"the processor's ColourCorrection does not accept the new file: {type(e).__name__}: {e}")
    return path


# ---------------------------------------------------------------------------------------------------------------
# 7 scores and pictures
# ---------------------------------------------------------------------------------------------------------------

def srgb_to_lab(rgb):
    """CIELAB (D65) of sRGB values 0..1, rows of 3 (as skimage.color.rgb2lab)"""
    c = np.where(rgb > 0.04045, ((rgb + 0.055) / 1.055) ** 2.4, rgb / 12.92)
    xyz = c @ np.array([[0.412453, 0.357580, 0.180423], [0.212671, 0.715160, 0.072169],
                        [0.019334, 0.119193, 0.950227]]).T
    xyz = xyz / np.array([0.95047, 1.0, 1.08883])
    f = np.where(xyz > 0.008856, np.cbrt(xyz), 7.787 * xyz + 16 / 116)
    return np.column_stack([np.where(xyz[:, 1] > 0.008856, 116 * np.cbrt(xyz[:, 1]) - 16, 903.3 * xyz[:, 1]),
                            500 * (f[:, 0] - f[:, 1]), 200 * (f[:, 1] - f[:, 2])])


def show(rho):
    return np.clip(np.nan_to_num(rho, nan=DISPLAY) / DISPLAY, 0, 1)


def scores(P, T, veg):
    """colour error (CIELAB dE on a 0..0.3 display) and vegetation shown green, after scaling P to T's brightness"""
    ok = np.isfinite(P).all(1) & np.isfinite(T).all(1)
    P, T, veg = P[ok], T[ok], veg[ok]
    k = float(np.median(P.sum(1) / np.maximum(T.sum(1), 1e-6)))
    lp, lt = srgb_to_lab(show(P / k)), srgb_to_lab(show(T))
    de = np.linalg.norm(lp - lt, axis=1)
    return dict(brightness_x=round(k, 2), colour_err=round(float(de.mean()), 2),
                colour_err_veg=round(float(de[veg].mean()), 2) if veg.any() else float("nan"),
                veg_green_pct=round(100 * float((lp[veg, 1] < 0).mean()), 1) if veg.any() else float("nan"))


def score_stage(prods, work, tpl, fit, new_aux):
    lut = Lut(tpl["axes"], tpl["window"])
    keys = ("n", "num", "den", "veg")
    src = {k: fit[f"src_{k}"] for k in keys}
    total = {k: src[k].sum(0) for k in keys}
    stems = list(fit["source_stems"])
    versions = ("as delivered", "current LUT", "new LUT, one balance", "new LUT, balance per satellite")
    prod_bal = {st: (sat, b, ok) for st, sat, b, ok in zip(fit["prod_stems"], fit["prod_sat"], fit["prod_bal"],
                                                            fit["prod_bal_ok"])}

    def satellite_balance(p):
        """the median balance of the satellite's OTHER products that fit well; else the product's own; else one"""
        others = [b for st, (sat, b, ok) in prod_bal.items() if sat == p["sat"] and ok and st != p["stem"]]
        if others:
            return np.median(others, axis=0), f"{p['sat']}, {len(others)} other product(s)"
        own = prod_bal.get(p["stem"])
        if own is not None and own[2]:
            return np.asarray(own[1]), "its own (no other product of this satellite)"
        return fit["balance"], "the one balance"
    rows, crops = [], []
    for p in prods:
        d = load_pairs(work, p)
        eag = work / "eag10" / f"{p['stem']}.npz"
        if not p.get("calibration_ok", True) or d is None or not eag.exists():
            continue
        i = rng.choice(len(d["e_dn"]), SCORE_PAIRS, replace=False) if len(d["e_dn"]) > SCORE_PAIRS else np.arange(len(d["e_dn"]))
        X, S = d["e_refl"][i].astype("float64"), d["s2"][i].astype("float64")
        T, veg = S[:, :3], ndvi(S) > 0.4
        rr, cc = d["row"][i], d["col"][i]
        img = np.load(eag)["refl"].astype("float64")                   # delivered, on the Sentinel-2 grid
        if p["stem"] in stems:                                         # the table without this product's footprint
            k = stems.index(p["stem"])
            q_loo, _ = lut.finish({key: total[key] - src[key][k] for key in keys})
        else:
            q_loo = fit["q"]
        maps, outs, bals = {}, {}, {}
        sat_bal, sat_note = satellite_balance(p)
        for name, q, vs, bal in (("current LUT", tpl["q"], tpl["veg_share"], tpl["balance"]),
                                 ("new LUT, one balance", q_loo, fit["veg_share"], fit["balance"]),
                                 ("new LUT, balance per satellite", q_loo, fit["veg_share"], sat_bal)):
            bal = np.asarray(bal, "float64")
            bals[name] = bal
            maps[name] = lut.around_map(img / bal[:, None, None], vs)
            outs[name] = lut.apply(q, X / bal, maps[name][rr, cc])
        outs["as delivered"] = X
        for name in versions:
            rows.append(dict(product=p["label"], version=name, **scores(outs[name], T, veg),
                             balance=" / ".join(f"{v:.3f}" for v in bals[name]) if name in bals else "",
                             balance_from=sat_note if name == "new LUT, balance per satellite" else ""))

        # picture: the 2 km with the most vegetation among the pairs
        h, w = img.shape[1:]
        busy = np.zeros((h, w))
        np.add.at(busy, (d["row"], d["col"]), np.where(ndvi(d["s2"].astype("float64")) > 0.4, 1.0, 0.05))
        busy = ndimage.uniform_filter(busy, CROP, mode="constant")
        cy, cx = np.unravel_index(np.argmax(busy), busy.shape)
        y0, x0 = int(np.clip(cy - CROP // 2, 0, max(h - CROP, 0))), int(np.clip(cx - CROP // 2, 0, max(w - CROP, 0)))
        win = (slice(y0, y0 + CROP), slice(x0, x0 + CROP))
        s2c, _ = load_composite(work, p)
        panel = {"as delivered": img[:, win[0], win[1]]}
        for name, q in (("current LUT", tpl["q"]), ("new LUT, one balance", q_loo), ("new LUT, balance per satellite", q_loo)):
            e = (img[:, win[0], win[1]] / bals[name][:, None, None]).reshape(3, -1).T
            panel[name] = lut.apply(q, e, maps[name][win].ravel()).T.reshape(3, *maps[name][win].shape)
        bright = {r["version"]: r["brightness_x"] for r in rows if r["product"] == p["label"]}
        crops.append(dict(label=p["label"], panels=[(f"{n} (/ {bright[n]:.2f})", panel[n] / bright[n]) for n in versions]
                          + [("Sentinel-2 L1C", s2c[:3, win[0], win[1]].astype("float64"))]))
    if not rows:
        raise SystemExit("no product to score")
    with (work / "scores.csv").open("w", newline="") as f:
        wr = csv.DictWriter(f, fieldnames=list(rows[0]))
        wr.writeheader()
        wr.writerows(rows)
    avg = []
    for name in versions:
        sel = [r for r in rows if r["version"] == name]
        avg.append(dict(version=name, **{k: round(float(np.nanmean([r[k] for r in sel])), 2)
                                         for k in ("brightness_x", "colour_err", "colour_err_veg", "veg_green_pct")}))
    picture(work / "lut_check.png", crops)

    a = {r["version"]: r for r in avg}

    def beats(v):
        return (a[v]["colour_err"] < a["current LUT"]["colour_err"]
                and a[v]["veg_green_pct"] >= a["current LUT"]["veg_green_pct"] - 5)
    verdict = ("the new LUT (one balance, as in the new AUX file) is better than the current one on these products, "
               "also where it did not learn from them" if beats("new LUT, one balance") else
               "the new LUT with one balance (as in the new AUX file) is NOT clearly better than the current one")
    verdict += ("; with a balance per satellite it IS better" if beats("new LUT, balance per satellite") else
                "; with a balance per satellite it is NOT clearly better either")
    # one LUT for every satellite? First how far each product's own colour balance is from the one balance; then
    # the scores, but only on products whose satellite balance came from OTHER products of that satellite (a product
    # alone on its satellite gets its own balance, which flatters it)
    devs = [(str(sat), float(b[0] / fit["balance"][0] - 1), float(b[2] / fit["balance"][2] - 1))
            for sat, b, ok in zip(fit["prod_sat"], fit["prod_bal"], fit["prod_bal_ok"]) if ok]
    worst = max(devs, key=lambda d: max(abs(d[1]), abs(d[2]))) if devs else None
    spread = (f"every product's own balance is within {100 * max(abs(d[1]) for d in devs):.0f}% (red/green) and "
              f"{100 * max(abs(d[2]) for d in devs):.0f}% (blue/green) of the one balance, furthest {worst[0]} "
              f"({100 * worst[1]:+.0f}% / {100 * worst[2]:+.0f}%)") if devs else "no product balance measured"
    loo = {r["product"] for r in rows if r["version"] == "new LUT, balance per satellite"
           and re.match(r"IME\d+, \d+ other product", r["balance_from"])}
    one = float(np.mean([r["colour_err"] for r in rows if r["version"] == "new LUT, one balance"
                         and r["product"] in loo])) if loo else float("nan")
    per = float(np.mean([r["colour_err"] for r in rows if r["version"] == "new LUT, balance per satellite"
                         and r["product"] in loo])) if loo else float("nan")
    if devs and max(max(abs(d[1]), abs(d[2])) for d in devs) <= 0.05:
        universal = f"YES: {spread}: one LUT for the whole constellation works"
    elif not loo:
        universal = (f"NOT SURE: {spread}, and no satellite has two products to test a balance per satellite "
                     "against (more products of the same satellites would settle it)")
    elif one <= per * 1.10 + 0.5:
        universal = (f"YES: on the {len(loo)} products with another product of their satellite, one balance is about "
                     f"as good as a balance per satellite (colour error {one:.1f} against {per:.1f}); {spread}")
    else:
        universal = (f"NO: on the {len(loo)} products with another product of their satellite, a balance per "
                     f"satellite is clearly better (colour error {per:.1f} against {one:.1f}); {spread}")
    text = "\n".join([
        f"NEW AUX FILE  {new_aux.name}",
        "",
        "SCORES AGAINST SENTINEL-2 (colour_err = CIELAB dE at Sentinel-2 brightness, lower is better; veg_green_pct = "
        "vegetation that looks green; brightness_x = brightness vs Sentinel-2)",
        table_text(rows), "", "AVERAGE", table_text(avg), "", f"VERDICT: {verdict}",
        f"UNIVERSAL (one LUT for every satellite)? {universal}"])
    log("\n" + text)
    return text


def picture(path, crops):
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        log("matplotlib not installed: no picture")
        return
    if not crops:
        return
    ncol = max(len(c["panels"]) for c in crops)
    fig, axes = plt.subplots(len(crops), ncol, figsize=(4 * ncol, 4.2 * len(crops)), squeeze=False)
    for row, c in zip(axes, crops):
        for ax, (title, img) in zip(row, c["panels"]):
            rgb = show(np.moveaxis(img, 0, -1))
            rgb[~np.isfinite(img).all(0)] = 1.0
            ax.imshow(rgb, interpolation="nearest")
            ax.set_title(title, loc="left", fontsize=9)
            ax.set_xticks([])
            ax.set_yticks([])
        row[0].set_ylabel(c["label"], fontsize=10)
    fig.suptitle("2 km crops at 10 m, the same display for all (reflectance 0 to 0.3); EAGLET versions divided by "
                 "their brightness against Sentinel-2", x=0.01, ha="left", fontsize=10)
    fig.tight_layout(rect=(0, 0, 1, 0.98))
    fig.savefig(path, dpi=90)
    plt.close(fig)
    log(f"picture: {path}")


# ---------------------------------------------------------------------------------------------------------------

def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--work", type=Path, default=PROJECT / "lut_a01", help="working folder (default: %(default)s)")
    ap.add_argument("--products", type=Path, default=PROJECT / "OG_data", help="L1C zips (default: %(default)s)")
    ap.add_argument("--baseline", default="A01", help="processing baseline of the products (default: %(default)s)")
    ap.add_argument("--since", default="20261001", help="only products generated on or after this day (YYYYMMDD)")
    ap.add_argument("--template", type=Path, help="the current colour AUX file (default: the newest in aux/)")
    ap.add_argument("--chips", type=Path, help="world chips folder (default: <work>/s2_chips)")
    ap.add_argument("--redo", choices=STAGES, help="do this step and the ones after it again for every product")
    ap.add_argument("--device", default="cpu", help="for OmniCloudMask (default: %(default)s)")
    a = ap.parse_args(argv)
    a.work.mkdir(parents=True, exist_ok=True)
    chip_dir = a.chips or a.work / "s2_chips"
    template = a.template or newest_aux(REPO / "aux")
    tpl = read_aux(template)
    redo = set(STAGES[STAGES.index(a.redo):]) if a.redo else set()
    cdse, ocm = Lazy(CDSE), Lazy(lambda: CloudMask(a.device))
    t0 = time.time()

    log(f"== 1 inventory  (template: {template.name}, method {tpl['method']})")
    prods = inventory_stage(a, a.work)
    log("\n== 2 Sentinel-2")
    s2_stage(prods, a.work, cdse, ocm, "s2" in redo)
    log("\n== 3 EAGLET onto Sentinel-2, pairs")
    pairs_stage(prods, a.work, ocm, "pairs" in redo)
    log("\n== 4 world chips")
    chips_stage(a.work, chip_dir, cdse, ocm)
    log("\n== 5 fit")
    fit_info = fit_stage(prods, a.work, chip_dir, tpl)
    fit = load_fit(a.work)
    log("\n== 6 the new AUX file")
    n_used = sum(1 for r in fit_info["products"] if r["used_for"])
    new_aux = aux_stage(a.work, tpl, fit, a.baseline, n_used)
    log("\n== 7 scores")
    text = score_stage(prods, a.work, tpl, fit, new_aux)
    old_sh = tpl["shares"]
    wrong = [f"   {p['label']}: median reflectance with its XML gains "
             f"{' / '.join(f'{r:.3g}' for r in p['refl_median'])} ({p['stem']})" for p in prods if not p["calibration_ok"]]
    summary = "\n".join([
        f"build_lut.py, {datetime.now(timezone.utc):%Y-%m-%d %H:%M} UTC, baseline {a.baseline}, {len(prods)} products",
        f"current AUX: {template.name}",
        "",
        *(["CALIBRATION LOOKS WRONG (left out of the fit and the scores; footprints still used)"] + wrong + [""]
          if wrong else []),
        "EACH PRODUCT", table_text(fit_info["products"]), "",
        "COLOUR BALANCE PER SATELLITE", table_text(fit_info["satellites"]), "",
        "WHAT CHANGED (current -> new)",
        "   near-infrared shares: " + (f"{old_sh[0]:.3f} / {old_sh[1]:.3f} / {old_sh[2]:.3f}" if old_sh is not None else "?")
        + " -> " + " / ".join(f"{v:.3f}" for v in fit_info["shares"]),
        f"   colour balance:       {tpl['balance'][0]:.4f} / 1 / {tpl['balance'][2]:.4f} -> "
        + f"{fit_info['balance'][0]:.4f} / 1 / {fit_info['balance'][2]:.4f}",
        "   added on top of Sentinel-2 (median offset): " + " / ".join(f"{v:.3f}" for v in fit_info["offsets"]),
        f"   table: {fit_info['sources']} sources ({fit_info['chips']} world chips), "
        f"{fit_info['table_cells_with_data_pct']}% of the cells backed by data", "", text,
        f"\n({(time.time() - t0) / 60:.0f} min)"])
    (a.work / "summary.txt").write_text(summary + "\n")
    log(f"\nsummary: {a.work / 'summary.txt'}")


if __name__ == "__main__":
    main()
