# colour_correction.py
"""
Colour correction of Eaglet-2 L1C RGB3 imagery towards Sentinel-2 L1C colours.

Two methods, plus their mix, all configured by one auxiliary file
(IME___AXE1D_CLUT_..., Earth Explorer XML, Data_Block/Colour_Correction):

  LUT           one colour balance and one look-up table for the whole
                constellation: b = rho / balance; out = b * q(ln(b1/b2),
                ln(b3/b2), vegetation around). The brightness stays as
                delivered. "Vegetation around" is the share of vegetation in
                the ~70 m around each pixel, judged from the image itself at
                ~10 m.
  CLASS_MODEL   Enrico Cadau's model (applica_L1C_RGB.py v2, stored unchanged
                as JSON in the AUX file): each pixel is weighted between
                water / vegetation / built-up by Gaussian mixtures on its RGB
                triplet, one transform per class, negatives set to 0. The
                brightness comes out close to Sentinel-2.
  HYBRID        his colour where his model sees vegetation or water inside
                its training range (scaled to the LUT's brightness), the LUT
                elsewhere.

The AUX file's <Method> element picks one. A version 3 file (no <Method>, no
<Class_Model>) works as LUT only.

The AUX file can be given as the XML itself (the processor's own, in aux/), or
as PDGS hands AUX files over: a zip, <ID>.zip, holding <ID>/<ID>.xml and the
PDGS header <ID>/<ID>.hdr, which the processor reads without unzipping it; a
folder holding the XML (such a zip unzipped) works too. The L1D names the file
by <ID>, its name without .xml or .zip.

Input for every method: top-of-atmosphere reflectance rho = DN * GAIN + OFFSET,
GAIN and OFFSET from the product XML's Rescaling_Factors. DN = 0 in any band is
no data. Output: the L1D image, a Cloud Optimized GeoTIFF (LZW, lossless) with
the Sentinel-2 encoding: uint16 = rho * 10000 + 1000, 0 = no data, so
rho = value * 0.0001 - 0.1.

A pixel saturated in the L1C (DN 65535, the top of the uint16 range, in any
band) has lost its colour: each band's GAIN turns the same clipped DN into a
different reflectance, which shows as purple. The L1D writes it white instead:
its three bands at the highest of its three reflectances.
"""

import base64
import hashlib
import json
import time
import xml.etree.ElementTree as ET
import zipfile
from pathlib import Path

import numpy as np
from scipy import ndimage

METHODS = ("LUT", "CLASS_MODEL", "HYBRID")
ENCODING_SCALE = 10000.0      # value = rho * ENCODING_SCALE + ENCODING_OFFSET
ENCODING_OFFSET = 1000
L1D_GAIN = 1 / ENCODING_SCALE                     # rho = value * L1D_GAIN + L1D_OFFSET (0.0001, -0.1)
L1D_OFFSET = -ENCODING_OFFSET / ENCODING_SCALE
NODATA_OUT = 0
# GDAL COG driver: LZW with the horizontal predictor (lossless), 512 x 512 tiles,
# overviews down to one tile, averaged (no-data pixels left out)
COG_OPTIONS = dict(compress="LZW", predictor="YES", blocksize=512, overviews="AUTO", resampling="AVERAGE",
                   bigtiff="IF_SAFER", num_threads="ALL_CPUS")
RENDER_PERCENTILES = (2, 98)  # display range (STAC renders, quicklooks): dark level of each band, span of the three
SATURATED_DN = 65535          # an L1C band at the top of the uint16 range is clipped: the pixel is written white


class ColourCorrectionError(Exception):
    pass


def aux_id(aux_path):
    """the name the colour AUX file goes by, as the L1C lists its AUX files: the file or folder name without
    .xml or .zip, e.g. IME___AXE1D_CLUT_20240101T000000_20990101T000000_20261019T112255____________O_001"""
    name = Path(aux_path).name
    return name[:-4] if name.lower().endswith((".xml", ".zip")) else name


def read_aux_xml(aux_path):
    """
    The XML of the colour AUX file at aux_path, which is one of:
      - the XML itself (the processor's own, in aux/);
      - a zip, as PDGS hands AUX files over (first PDGS run with an AUX file, 20 Oct 2026): <ID>.zip holding
        <ID>/<ID>.xml and the PDGS header <ID>/<ID>.hdr; read as it is, nothing is unzipped;
      - a folder holding the XML (such a zip, unzipped).
    In a zip or a folder the XML is its one .xml file or, when there are several, the one named <ID>.xml (<ID>:
    the name of the zip or folder). The .hdr and any other file are left alone.
    Returns (the XML as bytes, the .xml file in the zip or folder, or None for the XML itself).
    """
    path = Path(aux_path)
    try:
        if path.is_dir():
            files = {p.relative_to(path).as_posix(): p for p in sorted(path.rglob("*")) if p.is_file()}
            member = _aux_member(list(files), path)
            return files[member].read_bytes(), member
        if zipfile.is_zipfile(path):
            with zipfile.ZipFile(path) as zf:
                member = _aux_member([i.filename for i in zf.infolist() if not i.is_dir()], path)
                return zf.read(member), member
        return path.read_bytes(), None
    except ColourCorrectionError:
        raise
    except Exception as e:          # missing, unreadable, a damaged zip: no L1D image without the AUX file
        raise ColourCorrectionError(f"Cannot read colour AUX file {path}: {e}") from e


def _aux_member(names, package):
    """the AUX XML among the files of a zip or folder (see read_aux_xml)"""
    xmls = [n for n in names if n.lower().endswith(".xml") and not n.startswith("__MACOSX/")
            and not n.rsplit("/", 1)[-1].startswith("._")]
    if len(xmls) > 1:
        named = [n for n in xmls if n.rsplit("/", 1)[-1][:-4] == aux_id(package)]
        xmls = named if len(named) == 1 else xmls
    if len(xmls) != 1:
        listed = ", ".join(names[:10]) + (f" and {len(names) - 10} more" if len(names) > 10 else "")
        raise ColourCorrectionError(f"{package}: expected one .xml file in it, found {len(xmls)} "
                                    f"(it holds: {listed or 'nothing'})")
    return xmls[0]


class ColourCorrection:
    """
    Reads the colour-correction AUX file (the XML, or the zip or folder holding it: see read_aux_xml) and applies
    the selected method to reflectance triplets. method=None uses the file's own <Method>.
    """

    def __init__(self, aux_path, method=None):
        self.path = Path(aux_path)
        self.name = aux_id(self.path)
        data, member = read_aux_xml(self.path)
        self.source = self.path.name if member is None else f"{self.path.name} ({member})"
        try:
            root = ET.fromstring(data)
        except ET.ParseError as e:
            raise ColourCorrectionError(f"Cannot read colour AUX file {self.path}"
                                        f"{'' if member is None else f' ({member})'}: {e}") from e
        cc = root.find("Data_Block/Colour_Correction")
        if cc is None:
            raise ColourCorrectionError(f"{self.path.name}: no Data_Block/Colour_Correction")
        self.version = cc.get("version", "?")
        in_file = cc.find("Method")
        chosen = method or (in_file.text.strip() if in_file is not None and in_file.text else "LUT")
        self.method = chosen.upper()
        if self.method not in METHODS:
            raise ColourCorrectionError(
                f"{self.path.name}: unknown colour correction method {chosen!r}; use one of {', '.join(METHODS)}")

        try:
            tables = {}
            for el in cc.findall("Table"):
                raw = base64.b64decode(el.text)
                if hashlib.sha256(raw).hexdigest() != el.get("sha256"):
                    raise ColourCorrectionError(f"{self.path.name}: checksum mismatch in table {el.get('name')}")
                shape = [int(s) for s in el.get("shape").split()]
                tables[el.get("name")] = np.frombuffer(raw, "<f4").reshape(shape).astype("float64")
            self.q = np.stack([tables[f"q_band{b}"] for b in (1, 2, 3)])
            self.veg_share = tables["veg_share"]
            self.axes = {a.get("name"): (float(a.get("start")), float(a.get("step")), int(a.get("count")))
                         for a in cc.find("Axes")}
            self.balance = np.array([float(cc.find("Colour_Balance").get(f"band{b}")) for b in (1, 2, 3)])
            va = cc.find("Vegetation_Around")
            self.around_cells = int(va.get("window_cells"))
            self.cell_m = float(va.get("cell_m", "10"))
        except ColourCorrectionError:
            raise
        except (KeyError, AttributeError, TypeError, ValueError) as e:
            raise ColourCorrectionError(f"{self.path.name}: incomplete LUT section ({e})") from e

        self.model = None
        cm = cc.find("Class_Model")
        if cm is not None:
            text = (cm.text or "").strip()
            if hashlib.sha256(text.encode("utf-8")).hexdigest() != cm.get("sha256"):
                raise ColourCorrectionError(f"{self.path.name}: checksum mismatch in Class_Model")
            self.model = _prepare_model(json.loads(text))
        if self.method != "LUT" and self.model is None:
            raise ColourCorrectionError(
                f"{self.path.name} has no Class_Model, so method {self.method} is not available")

    @property
    def needs_around(self):
        return self.method in ("LUT", "HYBRID")

    def block_factor(self, pixel_m):
        """pixels per side of the ~10 m blocks used for "vegetation around\""""
        return max(1, int(round(self.cell_m / pixel_m)))

    # ---- LUT ----------------------------------------------------------------
    def _index(self, E, around=None):
        E = np.maximum(E, 1e-4)
        vals = [np.log(E[:, 0] / E[:, 1]), np.log(E[:, 2] / E[:, 1])]
        if around is not None:
            vals.append(np.asarray(around))
        names = "xyv"[:len(vals)]
        t = [(v - self.axes[n][0]) / self.axes[n][1] for v, n in zip(vals, names)]
        return np.clip(np.stack(t), 0, np.array([self.axes[n][2] for n in names])[:, None] - 1)

    def lut(self, rho, around):
        """rho (n, 3) -> (n, 3); a pixel with a band <= 0 gets the balanced value only"""
        E = rho / self.balance
        out = E.copy()
        ok = (E > 0).all(1) & np.isfinite(E).all(1) & np.isfinite(around)
        if ok.any():
            t = self._index(E[ok], around[ok])
            for b in range(3):
                out[ok, b] = E[ok, b] * ndimage.map_coordinates(self.q[b], t, order=1, mode="nearest")
        return out

    def around_map(self, block_means):
        """vegetation around (0 to 1) per ~10 m block, from (nby, nbx, 3) mean rho per block (NaN = no data)"""
        E = np.moveaxis(block_means / self.balance, -1, 0)
        valid = np.isfinite(E).all(0) & (np.nan_to_num(E) > 0).all(0)
        p = np.zeros(valid.shape)
        if valid.any():
            p[valid] = ndimage.map_coordinates(self.veg_share, self._index(E[:, valid].T), order=1, mode="nearest")
        num = ndimage.uniform_filter(p, self.around_cells, mode="constant")
        den = ndimage.uniform_filter(valid.astype("float64"), self.around_cells, mode="constant")
        return np.clip(np.where(den > 1e-6, num / np.maximum(den, 1e-6), 0.0), 0, 1)

    @staticmethod
    def around_at(amap, rows, cols, f):
        """vegetation around at pixel positions (full-image rows, cols), interpolated between block centres"""
        return ndimage.map_coordinates(
            amap, [(np.asarray(rows) + 0.5) / f - 0.5, (np.asarray(cols) + 0.5) / f - 0.5],
            order=1, mode="nearest")

    # ---- Enrico's class model ---------------------------------------------------
    def class_model(self, rho):
        """rho (n, 3) -> (output with negatives set to 0, class weights (n, 3): water, vegetation, built-up)"""
        out, w = _class_model_apply(np.asarray(rho, dtype="float64"), self.model)
        return np.maximum(out, 0), w

    # ---- the switch -------------------------------------------------------------
    def apply(self, rho, around=None):
        """the selected method on rho (n, 3) of valid pixels; around (n,) is needed for LUT and HYBRID"""
        if len(rho) == 0:
            return np.zeros((0, 3))
        if self.method == "CLASS_MODEL":
            return self.class_model(rho)[0]
        ours = self.lut(rho, around)
        if self.method == "LUT":
            return ours
        his, w = self.class_model(rho)
        m = self.model
        inside = ~((rho < m["low"]) | (rho > m["high"])).any(1)
        alpha = np.where(inside, w[:, 0] + w[:, 1], 0.0)[:, None]
        s_ours, s_his = ours.sum(1, keepdims=True), his.sum(1, keepdims=True)
        his_at_ours = np.where(s_his > 1e-4, his * s_ours / np.maximum(s_his, 1e-4), ours)
        return alpha * his_at_ours + (1 - alpha) * ours

    def describe(self):
        return f"method {self.method}, AUX {self.name} (version {self.version})"


def _prepare_model(M):
    """Enrico's MODEL dict with its lists as arrays, plus his training range (LOW / HIGH in his script)"""
    P = dict(center=np.asarray(M["center"], "float64"), scale=np.asarray(M["scale"], "float64"),
             components=[dict(mean=np.asarray(g["mean"], "float64"),
                              precision=np.asarray(g["precision"], "float64"),
                              log_weight=np.asarray(g["log_weight"], "float64")) for g in M["components"]],
             experts={})
    for key, e in M["experts"].items():
        P["experts"][key] = {k: (np.asarray(v, "float64") if isinstance(v, list) else v) for k, v in e.items()}
    P["low"] = np.min([np.asarray(e["lo"], "float64") for e in M["experts"].values()], axis=0)
    P["high"] = np.max([np.asarray(e["hi"], "float64") for e in M["experts"].values()], axis=0)
    return P


def _class_model_apply(x, m):
    """the same computation as apply() in Enrico's applica_L1C_RGB.py v2"""
    z = (x - m["center"]) / m["scale"]
    logs = []
    for g in m["components"]:
        v = []
        for mu, p, lw in zip(g["mean"], g["precision"], g["log_weight"]):
            d = z - mu
            v.append(lw - .5 * np.einsum("ni,ij,nj->n", d, p, d))
        v = np.stack(v, 1)
        mx = v.max(1)
        logs.append(mx + np.log(np.exp(v - mx[:, None]).sum(1)))
    logs = np.stack(logs, 1)
    w = np.exp(logs - logs.max(1)[:, None])
    w /= w.sum(1)[:, None]
    out = np.zeros_like(x)
    for k, c in enumerate((1, 2, 3)):
        e = m["experts"][str(c)]
        if c == 1:
            y = np.column_stack([np.interp(x.mean(1), e["xq"], e["yq"][j]) for j in range(3)])
        else:
            y = np.c_[(x - e["mean"]) / e["scale"], np.ones(len(x))] @ np.asarray(e["coef"])
        out += y * w[:, k, None]
    return out, w


def encode(rho):
    """reflectance (n, 3) -> uint16, Sentinel-2 convention (+1000 offset), never 0 for valid pixels"""
    return np.clip(np.rint(rho * ENCODING_SCALE) + ENCODING_OFFSET, 1, 65535).astype("uint16")


def percentiles_from_counts(counts, percentiles=RENDER_PERCENTILES):
    """[[low, high], ...] per band from (3, 65536) counts of uint16 values"""
    out = []
    for c in counts:
        total = c.sum()
        cum = np.cumsum(c)
        out.append([int(np.searchsorted(cum, total * p / 100.0)) if total else 0 for p in percentiles])
    return out


def display_range(counts, percentiles=RENDER_PERCENTILES):
    """
    [[low, high]] per band, in L1D values, from (3, 65536) counts of the
    values of the pixels not saturated. Each band starts at its own dark
    level, its 2nd percentile: the haze, which top-of-atmosphere colour has
    most of in blue. The three bands share one span, the 98th percentile of
    the three after their dark levels. So a display with this range takes
    the haze (the blue cast) away and keeps the colour balance of the
    correction; a stretch of each band to its own 98th percentile would
    re-balance the colours of every scene (a large thin cloud turns green).
    """
    lows = [low for low, _ in percentiles_from_counts(counts, percentiles)]
    pooled = np.zeros(65536, np.int64)               # the three bands, each minus its dark level
    for c, low in zip(counts, lows):
        pooled[0] += c[:low].sum()
        pooled[:65536 - low] += c[low:]
    total = pooled.sum()
    span = int(np.searchsorted(np.cumsum(pooled), total * percentiles[1] / 100.0)) if total else 1
    return [[low, min(low + max(span, 1), 65535)] for low in lows]


def write_colour_product(tif_path, gains, offsets, out_path, cc, logger, tile=1024, strip=512):
    """
    Colour-corrects the L1C image tif_path with cc's method and writes the
    L1D image to out_path: a Cloud Optimized GeoTIFF on the same grid, 3 bands
    uint16 (encode() convention, 0 = no data), LZW with the horizontal
    predictor (lossless), 512 x 512 tiles and internal overviews.
    The image is read and corrected in tiles, so memory stays small whatever
    the product size; the tiles go to a plain tiled GeoTIFF first, which
    GDAL's COG driver then rewrites as the COG. A pixel saturated in the L1C
    (DN SATURATED_DN in any band) is written white: its three bands at the
    highest of its three reflectances.
    Returns a dict of facts for the logs, the XML and the STAC entry: the
    display range (display_range() of the pixels not saturated), the number of
    saturated pixels and the seconds spent correcting and writing the COG.
    """
    import rasterio
    from rasterio.shutil import copy as copy_dataset
    from rasterio.windows import Window

    if tuple(int(v) for v in rasterio.__gdal_version__.split(".")[:2]) < (3, 1):
        raise ColourCorrectionError(f"GDAL {rasterio.__gdal_version__} has no COG driver (3.1 or newer needed)")

    gains = np.asarray(gains, dtype="float64")
    offsets = np.asarray(offsets, dtype="float64")
    out_path = Path(out_path)
    tiles_path = out_path.with_name(out_path.name + ".tiles.tif")    # plain tiled GeoTIFF, removed at the end
    part_path = out_path.with_name(out_path.name + ".part")          # the COG while it is being written
    counts = np.zeros((3, 65536), np.int64)          # values of the valid pixels not saturated, per band

    def valid_px(dn, nodata):
        v = (dn > 0).all(0)
        if nodata is not None and np.isfinite(nodata):
            v &= (dn != nodata).all(0)
        return v

    t_start = time.perf_counter()
    try:
        with rasterio.open(tif_path) as src:
            H, W = src.height, src.width
            amap, f = None, 1
            if cc.needs_around:
                f = cc.block_factor(abs(src.transform.a))
                nby, nbx = -(-H // f), -(-W // f)
                tot, cnt = np.zeros((nby, nbx, 3)), np.zeros((nby, nbx))
                rows_per = f * max(1, strip // f)
                for r0 in range(0, H, rows_per):
                    h = min(rows_per, H - r0)
                    dn = src.read([1, 2, 3], window=Window(0, r0, W, h)).astype("float32")
                    valid = valid_px(dn, src.nodata)
                    X = np.where(valid, dn * gains[:, None, None].astype("float32")
                                 + offsets[:, None, None].astype("float32"), 0.0)
                    hb = -(-h // f)
                    Xp = np.pad(X, ((0, 0), (0, hb * f - h), (0, nbx * f - W)))
                    vp = np.pad(valid, ((0, hb * f - h), (0, nbx * f - W)))
                    tot[r0 // f:r0 // f + hb] += np.moveaxis(
                        Xp.reshape(3, hb, f, nbx, f).sum((2, 4), dtype="float64"), 0, -1)
                    cnt[r0 // f:r0 // f + hb] += vp.reshape(hb, f, nbx, f).sum((1, 3))
                amap = cc.around_map(np.where((cnt > 0.5 * f * f)[..., None],
                                              tot / np.maximum(cnt, 1)[..., None], np.nan))
                logger.info(f"Colour: vegetation around on {amap.shape[1]} x {amap.shape[0]} blocks of {f} x {f} pixels")

            profile = dict(driver="GTiff", width=W, height=H, count=3, dtype="uint16", nodata=NODATA_OUT,
                           crs=src.crs, transform=src.transform, tiled=True, blockxsize=512, blockysize=512,
                           photometric="RGB", BIGTIFF="IF_SAFER")
            n_tiles = -(-H // tile) * -(-W // tile)
            done, n_valid, n_saturated = 0, 0, 0
            with rasterio.open(tiles_path, "w", **profile) as dst:
                dst.update_tags(COLOUR_METHOD=cc.method, COLOUR_AUX_FILE=cc.name,
                                ENCODING="uint16 = reflectance * 10000 + 1000, 0 = no data",
                                CALIBRATION="reflectance = DN * GAIN + OFFSET, product XML Rescaling_Factors",
                                BRIGHTNESS="as delivered" if cc.method != "CLASS_MODEL"
                                else "close to Sentinel-2 (class model)",
                                SATURATED=f"L1C DN {SATURATED_DN} in any band: written white, the three bands at "
                                          "the highest of the three reflectances")
                for b, name in enumerate(("red", "green", "blue"), 1):
                    dst.set_band_description(b, name)
                for r0 in range(0, H, tile):
                    for c0 in range(0, W, tile):
                        h, w = min(tile, H - r0), min(tile, W - c0)
                        win = Window(c0, r0, w, h)
                        dn = src.read([1, 2, 3], window=win).astype("float64")
                        valid = valid_px(dn, src.nodata)
                        rr, cc_ = np.nonzero(valid)
                        arr = np.full((3, h, w), NODATA_OUT, "uint16")
                        if len(rr):
                            rho = dn[:, rr, cc_].T * gains + offsets
                            around = cc.around_at(amap, r0 + rr, c0 + cc_, f) if cc.needs_around else None
                            out = cc.apply(rho, around)
                            sat = (dn[:, rr, cc_] >= SATURATED_DN).any(0)
                            if sat.any():       # white, not the purple the band gains give a clipped DN
                                out[sat] = rho[sat].max(1, keepdims=True)
                                n_saturated += int(sat.sum())
                            values = encode(out)
                            arr[:, rr, cc_] = values.T
                            for b in range(3):
                                counts[b] += np.bincount(values[~sat, b], minlength=65536)
                            n_valid += len(rr)
                        dst.write(arr, window=win)
                        done += 1
                        if done % 25 == 0 or done == n_tiles:
                            logger.progress(f"Colour correction: {done}/{n_tiles} tiles")
        t_corrected = time.perf_counter()
        logger.progress("Writing the L1D image as a COG (LZW, overviews)")
        copy_dataset(str(tiles_path), str(part_path), driver="COG", **COG_OPTIONS)
        part_path.replace(out_path)
        t_done = time.perf_counter()
    finally:
        for p in (tiles_path, part_path):
            if p.exists():
                p.unlink()
    return dict(method=cc.method, aux_file=cc.name, aux_version=cc.version, valid_pixels=n_valid,
                saturated_pixels=n_saturated,
                width=W, height=H, rescale=display_range(counts),
                seconds_correction=t_corrected - t_start, seconds_cog=t_done - t_corrected)