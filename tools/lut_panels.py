#!/usr/bin/env python3
"""
lut_panels.py: every product of the LUT build side by side, the same display for all four images:
    L1C as delivered | L1D, current LUT | L1D, new LUT | Sentinel-2 L1C, the pass closest in time
the whole scene on top and 2 km at full resolution below. Run after tools/build_lut.py and the processor runs:

    python tools/lut_panels.py --work /lut/work --runs /runs [--new-tag LUT_NEW] [--zoom 0.5 0.5]

The L1Ds come from the processor's working folders in --runs (made by run_l1d.sh): the current LUT from the runs
without a tag (the processor's own AUX file), the new LUT from the runs tagged --new-tag; the newest run of each.
Sentinel-2 is the pass closest in time that build_lut.py downloaded (<work>/s2), moved by the map correction
build_lut.py found for EAGLET (<work>/eag10) so that both show the same ground. Every image: reflectance 0 to 0.3 in
every band. The colour scores against Sentinel-2 come from build_lut.py (<work>/scores.csv). Writes, in
<work>/final: NN_<satellite>_<sensing>.png for each product, final_sheet.png (every product, small), final_map.html
(where they are; open it in a browser) and final.csv.
"""
import argparse
import csv
import json
import re
import sys
import zipfile
from pathlib import Path

import numpy as np
import rasterio
from PIL import Image, ImageDraw, ImageFont
from rasterio.enums import Resampling
from rasterio.transform import Affine
from rasterio.warp import reproject
from rasterio.windows import Window

sys.path.insert(0, str(Path(__file__).resolve().parent))
import build_lut as bl  # noqa: E402

TOP = 0.3                  # reflectance shown as white, in every band
PANEL = 640                # width of each image, pixels
ZOOM = 1000                # zoom window, full-resolution pixels (2 km at 2 m)
THUMB = 300                # width of each image on the sheet
PICTURE_RES = 20.0         # metres: a Sentinel-2 day fetched for the picture only (when build_lut.py kept none)
L1D_KEY = re.compile(r"^(IME\d{2})_1DSP__RGB3_(\d{8}T\d{6})_")
RUN = re.compile(r"^WD_(IME\d{2})_(?:(.+)_)?(\d{8}T\d{6})$")
COLOURS = ("#2a78d6", "#e34948", "#008300", "#eda100", "#7a3fbf", "#00a3a3", "#c2185b", "#5d4037", "#455a64")


def font(size):
    try:
        return ImageFont.load_default(size=size)
    except TypeError:                                   # Pillow before 10.1
        return ImageFont.load_default()


def to_rgb(rho, valid):
    """reflectance (3, h, w) as an 8-bit picture: 0 black, TOP white, the same for the three bands; no data white"""
    x = np.clip(np.nan_to_num(rho / TOP), 0, 1)
    rgb = np.moveaxis(np.rint(255 * x).astype("uint8"), 0, -1)
    rgb[~valid] = 255
    return Image.fromarray(rgb)


def blank(size, text):
    img = Image.new("RGB", size, "#eeeeee")
    ImageDraw.Draw(img).text((16, 16), text, fill="#555555", font=font(20))
    return img


def l1d_index(runs):
    """{(satellite, sensing start): {tag: L1D zip}}, the newest run of each tag ('' = no tag)"""
    found = {}
    for wd in sorted(Path(runs).glob("WD_*"), key=lambda d: (RUN.match(d.name) or [None] * 4)[3] or ""):
        m = RUN.match(wd.name)
        if not m or not wd.is_dir():
            continue
        for z in wd.glob("IME*_1DSP__RGB3_*.zip"):
            k = L1D_KEY.match(z.name)
            if k:
                found.setdefault(k.groups(), {})[m.group(2) or ""] = z    # later (newer) runs replace older ones
    return found


def image_member(zpath):
    with zipfile.ZipFile(zpath) as z:
        tifs = sorted(n for n in z.namelist() if re.search(r"/IMG_DATA/[^/]+\.tif$", n))
    if not tifs:
        raise ValueError(f"no IMG_DATA image in {zpath.name}")
    return f"/vsizip/{zpath}/{tifs[0]}"


def read_rho(path, kind, p, window=None, out_shape=None):
    """(3, h, w) reflectance and the valid mask: the L1C with its XML gains, the L1D as value x 0.0001 - 0.1"""
    with rasterio.open(path) as src:
        dn = src.read([1, 2, 3], window=window, out_shape=out_shape,
                      resampling=Resampling.average if out_shape else Resampling.nearest).astype("float64")
    valid = (dn > 0).all(0)
    if kind == "L1D":
        return dn * 0.0001 - 0.1, valid
    return np.stack([dn[b] * p["gains"][b] + p["offsets"][b] for b in range(3)]), valid


def s2_day(p, work, out, cdse):
    """the Sentinel-2 day: of the passes build_lut.py downloaded, the closest in time; else the closest day with
    Sentinel-2 over at least half the footprint (fetched for the picture only); else None"""
    info_path = work / "s2" / f"{p['stem']}.json"
    info = json.loads(info_path.read_text()) if info_path.exists() else {}
    for c in sorted(info.get("passes", []), key=lambda c: abs(c["dt_days"])):
        path = work / "s2" / p["stem"] / f"{c['day']}.tif"
        if path.exists():
            return path, c
    for c in sorted(info.get("candidates", []), key=lambda c: (abs(c["dt_days"]), -c.get("clear", 0))):
        if c.get("cover", 0) < 0.5:
            continue
        path = out / "s2_days" / f"{p['stem']}_{c['day']}.tif"
        if not path.exists():
            bounds, _, _, tf = bl.grid_of(p["bounds"], PICTURE_RES)
            a = cdse().fetch("sentinel-2-l1c", bl.EVAL_L1C, bounds, p["epsg"], PICTURE_RES, c["day"], 5, "uint16")
            path.parent.mkdir(exist_ok=True)
            bl.write_tif(path, a[:3], p["epsg"], tf)
        return path, c
    return None, None


def s2_on(path, fix, dst_tf, shape, how):
    """Sentinel-2 bands 1 to 3 (reflectance x 10000) where EAGLET's pixels of the grid dst_tf really are: EAGLET's
    map position X is on the ground at fix(X)"""
    with rasterio.open(path) as src:
        out = np.zeros((3, *shape), "float32")
        for b in range(3):
            reproject(rasterio.band(src, b + 1), out[b], src_nodata=0, dst_transform=fix * dst_tf, dst_crs=src.crs,
                      dst_nodata=0, resampling=how)
    return to_rgb(out / 10000.0, (out > 0).all(0))


def footprint(p):
    with zipfile.ZipFile(p["zip"]) as zf:
        geom = json.loads(zf.read(f"{p['stem']}/{p['stem']}.json")).get("geometry")
    if not geom:
        w, s, e, n = p["bbox_ll"]
        geom = {"type": "Polygon", "coordinates": [[[w, s], [e, s], [e, n], [w, n], [w, s]]]}
    return geom


def scores_by_product(work):
    path = work / "scores.csv"
    if not path.exists():
        return {}
    out = {}
    for r in csv.DictReader(path.open()):
        out.setdefault(r["product"], {})[r["version"]] = r
    return out


def score_text(sc):
    def v(version, key, fmt):
        r = sc.get(version)
        try:
            return fmt.format(float(r[key]))
        except (TypeError, KeyError, ValueError):
            return "?"
    if not sc:
        return "colour scores: none (build_lut.py had no pairs for this product)"
    names = (("as delivered", "delivered"), ("current LUT", "current LUT"), ("new LUT, one balance", "new LUT"))
    return ("against Sentinel-2 (build_lut.py; the new LUT scored without this product's footprint): colour error "
            "(dE, lower is better) "
            + ", ".join(f"{n} {v(k, 'colour_err', '{:.2f}')}" for k, n in names)
            + "; vegetation shown green " + ", ".join(f"{n} {v(k, 'veg_green_pct', '{:.0f}')}%" for k, n in names))


def one(k, p, work, out, index, new_tag, zoom_at, scores, cdse):
    with rasterio.open(p["tif"]) as src:
        H, W, tf = src.height, src.width, src.transform
    oh = max(1, int(round(PANEL * H / W)))
    disp_tf = tf * Affine.scale(W / PANEL, H / oh)
    col0 = int(np.clip(zoom_at[0] * W - ZOOM / 2, 0, max(W - ZOOM, 0)))
    row0 = int(np.clip(zoom_at[1] * H - ZOOM / 2, 0, max(H - ZOOM, 0)))
    zw, zh = min(ZOOM, W - col0), min(ZOOM, H - row0)
    win = Window(col0, row0, zw, zh)
    win_tf = tf * Affine.translation(col0, row0)

    eag = work / "eag10" / f"{p['stem']}.npz"
    fix = Affine(*np.load(eag)["fix"]) if eag.exists() else Affine.identity()
    rep_path = work / "align" / f"{p['stem']}.json"
    rep = json.loads(rep_path.read_text()) if rep_path.exists() else {}
    runs = index.get((p["sat"], p["sensing"]), {})

    columns = [("L1C, as delivered", p["tif"], "L1C"),
               ("L1D, current LUT", runs.get(""), "L1D"),
               (f"L1D, new LUT ({new_tag})", runs.get(new_tag), "L1D")]
    tops, bottoms, titles = [], [], []
    for title, src, kind in columns:
        titles.append(title)
        if src is None:
            tops.append(blank((PANEL, oh), "no L1D in the runs folder"))
            bottoms.append(blank((PANEL, PANEL), "no L1D in the runs folder"))
            continue
        path = src if kind == "L1C" else image_member(src)
        rho, valid = read_rho(path, kind, p, out_shape=(3, oh, PANEL))
        tops.append(to_rgb(rho, valid))
        rho, valid = read_rho(path, kind, p, window=win)
        bottoms.append(to_rgb(rho, valid).resize((PANEL, PANEL), Image.LANCZOS))   # 2 km: 1000 pixels to PANEL
    s2_path, day = s2_day(p, work, out, cdse)
    if s2_path is None:
        titles.append("Sentinel-2 L1C: none within 6 days")
        tops.append(blank((PANEL, oh), "no Sentinel-2 within 6 days"))
        bottoms.append(blank((PANEL, PANEL), "no Sentinel-2 within 6 days"))
    else:
        titles.append(f"Sentinel-2 L1C, {day['day']} ({day['dt_days']:+.1f} d)")
        tops.append(s2_on(s2_path, fix, disp_tf, (oh, PANEL), Resampling.average))
        bottoms.append(s2_on(s2_path, fix, win_tf, (zh, zw), Resampling.nearest).resize((PANEL, PANEL),
                                                                                       Image.NEAREST))
    w, s, e, n = p["bbox_ll"]
    lon, lat = (w + e) / 2, (s + n) / 2
    t = bl.utc(p["time"])
    moved = (f"Sentinel-2 moved onto EAGLET's ground: EAGLET was {abs(rep['east_m'])} m "
             f"{'east' if rep['east_m'] >= 0 else 'west'} and {abs(rep['north_m'])} m "
             f"{'north' if rep['north_m'] >= 0 else 'south'} of it (white along an edge: outside the Sentinel-2 "
             "download)" if eag.exists() and "east_m" in rep else
             "Sentinel-2 NOT moved onto EAGLET (build_lut.py found no trusted fit): the two may show different ground")
    notes = [p["stem"], score_text(scores.get(p["label"], {})), moved,
             f"Every image: reflectance 0 to {TOP} in every band. Bottom row: 2 km at full resolution (red box)."]

    pad, gap, head, cap, line = 20, 16, 60, 34, 26
    sheet = Image.new("RGB", (2 * pad + 4 * PANEL + 3 * gap,
                              head + cap + oh + gap + PANEL + 14 + line * len(notes) + pad), "white")
    d = ImageDraw.Draw(sheet)
    d.text((pad, 16), f"{p['label']}   {t:%Y-%m-%d %H:%M} UTC   centre {lat:.3f}, {lon:.3f}", fill="black",
           font=font(26))
    for i, (title, top, bottom) in enumerate(zip(titles, tops, bottoms)):
        x = pad + i * (PANEL + gap)
        d.text((x, head), title, fill="#333333", font=font(20))
        sheet.paste(top, (x, head + cap))
        sheet.paste(bottom, (x, head + cap + oh + gap))
    s = PANEL / W
    for i in range(4):                                         # where the 2 km are, on every whole-scene image
        x = pad + i * (PANEL + gap)
        d.rectangle([x + col0 * s, head + cap + row0 * s, x + (col0 + zw) * s, head + cap + (row0 + zh) * s],
                    outline="red", width=2)
    y = head + cap + oh + gap + PANEL + 14
    for text in notes:
        d.text((pad, y), text, fill="#555555", font=font(17))
        y += line
    name = f"{k:02d}_{p['sat']}_{p['sensing']}.png"
    sheet.save(out / name)

    thumbs = []
    for top in tops:
        th = top.copy()
        th.thumbnail((THUMB, THUMB))
        thumbs.append(th)
    sc = scores.get(p["label"], {})
    row = dict(n=k, label=p["label"], sensing_utc=f"{t:%Y-%m-%d %H:%M}", lat=round(lat, 4), lon=round(lon, 4),
               s2_day="" if day is None else day["day"], s2_dt_days="" if day is None else day["dt_days"],
               s2_moved_onto_eaglet=bool(eag.exists()), east_m=rep.get("east_m", ""), north_m=rep.get("north_m", ""),
               l1d_current_lut=bool(runs.get("")), l1d_new_lut=bool(runs.get(new_tag)),
               colour_err_delivered=sc.get("as delivered", {}).get("colour_err", ""),
               colour_err_current=sc.get("current LUT", {}).get("colour_err", ""),
               colour_err_new=sc.get("new LUT, one balance", {}).get("colour_err", ""),
               veg_green_current_pct=sc.get("current LUT", {}).get("veg_green_pct", ""),
               veg_green_new_pct=sc.get("new LUT, one balance", {}).get("veg_green_pct", ""),
               panel=name, product=p["stem"])
    scene = dict(label=p["label"], sat=p["sat"], time=f"{t:%Y-%m-%d %H:%M}", lat=lat, lon=lon,
                 s2="none" if day is None else f"{day['day']} ({day['dt_days']:+.1f} d)", panel=name,
                 geometry=footprint(p))
    return row, scene, (f"{p['label']}  {t:%H:%M} UTC  ({lat:.2f}, {lon:.2f})", thumbs)


MAP = """<!doctype html>
<html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width, initial-scale=1">
<title>EAGLET-2 LUT products</title>
<link rel="stylesheet" href="https://unpkg.com/leaflet@1.9.4/dist/leaflet.css">
<script src="https://unpkg.com/leaflet@1.9.4/dist/leaflet.js"></script>
<style>html, body {margin: 0; height: 100%; font: 14px sans-serif} #map {height: 100%}
.legend {background: #fff; padding: 6px 10px; border-radius: 4px; line-height: 1.7}
.legend i {display: inline-block; width: 12px; height: 12px; margin-right: 6px; border-radius: 50%}</style>
</head><body><div id="map"></div><script>
const scenes = __SCENES__;
const colours = __COLOURS__;
const map = L.map("map", {worldCopyJump: true});
L.tileLayer("https://tile.openstreetmap.org/{z}/{x}/{y}.png",
  {maxZoom: 18, attribution: "&copy; OpenStreetMap contributors"}).addTo(map);
const all = L.featureGroup().addTo(map);
for (const s of scenes) {
  const c = colours[s.sat] || "#444";
  const pop = `<b>${s.label}</b><br>${s.time} UTC<br>centre ${s.lat.toFixed(3)}, ${s.lon.toFixed(3)}<br>` +
              `Sentinel-2: ${s.s2}<br><a href="${s.panel}" target="_blank">open the panel</a>`;
  L.geoJSON(s.geometry, {style: {color: c, weight: 2, fillOpacity: 0.2}}).bindPopup(pop).addTo(all);
  L.circleMarker([s.lat, s.lon], {radius: 7, color: "#fff", weight: 2, fillColor: c, fillOpacity: 1})
    .bindTooltip(s.label).bindPopup(pop).addTo(all);
}
map.fitBounds(all.getBounds(), {padding: [40, 40], maxZoom: 10});
const legend = L.control({position: "bottomleft"});
legend.onAdd = () => {
  const d = L.DomUtil.create("div", "legend");
  d.innerHTML = Object.entries(colours).map(([k, v]) => `<i style="background:${v}"></i>${k}`).join("<br>");
  return d;
};
legend.addTo(map);
</script></body></html>
"""


def write_overview(rows, scenes, items, new_tag, out):
    """the sheet (one product per row, the four images small), the map and the list, of the products done so far"""
    tw = max(t.size[0] for _, thumbs in items for t in thumbs)
    heights = [max(t.size[1] for t in thumbs) for _, thumbs in items]
    pad, inner, gap, lab = 20, 6, 22, 28
    sheet = Image.new("RGB", (2 * pad + 4 * tw + 3 * inner, 90 + sum(lab + h + gap for h in heights) + pad), "white")
    d = ImageDraw.Draw(sheet)
    d.text((pad, 16), f"L1C as delivered | L1D, current LUT | L1D, new LUT ({new_tag}) | Sentinel-2 L1C",
           fill="black", font=font(20))
    d.text((pad, 46), f"reflectance 0 to {TOP} in every band, the same display for all", fill="#555555", font=font(17))
    y = 90
    for (label, thumbs), h in zip(items, heights):
        d.text((pad, y), label, fill="#333333", font=font(17))
        for j, t in enumerate(thumbs):
            sheet.paste(t, (pad + j * (tw + inner), y + lab))
        y += lab + h + gap
    sheet.save(out / "final_sheet.png")
    sats = sorted({s["sat"] for s in scenes})
    colours = {s: COLOURS[i % len(COLOURS)] for i, s in enumerate(sats)}
    (out / "final_map.html").write_text(MAP.replace("__SCENES__", json.dumps(scenes))
                                        .replace("__COLOURS__", json.dumps(colours)))
    with open(out / "final.csv", "w", newline="") as f:
        wr = csv.DictWriter(f, fieldnames=list(rows[0]))
        wr.writeheader()
        wr.writerows(rows)


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--work", type=Path, default=bl.PROJECT / "lut_a01", help="build_lut.py's folder")
    ap.add_argument("--runs", type=Path, required=True, help="the processor's working folders (run_l1d.sh's runs/)")
    ap.add_argument("--new-tag", default="LUT_NEW", help="tag of the runs with the new LUT (default: %(default)s)")
    ap.add_argument("--zoom", type=float, nargs=2, default=(0.5, 0.5), metavar=("X", "Y"),
                    help="centre of the 2 km, as fractions of the scene (default: the middle)")
    a = ap.parse_args(argv)
    prods = bl.load_inventory(a.work)
    out = a.work / "final"
    out.mkdir(exist_ok=True)
    index = l1d_index(a.runs)
    scores = scores_by_product(a.work)
    cdse = bl.Lazy(bl.CDSE)
    bl.log(f"{len(prods)} products in {a.work / 'inventory.json'}; L1Ds of {len(index)} images in {a.runs}")
    rows, scenes, items = [], [], []
    for k, p in enumerate(prods):
        try:
            row, scene, item = one(k, p, a.work, out, index, a.new_tag, a.zoom, scores, cdse)
        except Exception as e:
            bl.log(f"!! {p['label']}: no panel: {type(e).__name__}: {e}")
            continue
        rows.append(row)
        scenes.append(scene)
        items.append(item)
        write_overview(rows, scenes, items, a.new_tag, out)
        missing = [n for n, ok in (("current LUT", row["l1d_current_lut"]), ("new LUT", row["l1d_new_lut"])) if not ok]
        bl.log(f"{p['label']}: {out / row['panel']}" + (f"   (no L1D with the {' or '.join(missing)})" if missing else ""))
    if not rows:
        raise SystemExit("no panel made (see the !! lines above)")
    bl.log("\n" + bl.table_text([{c: r[c] for c in ("label", "sensing_utc", "lat", "lon", "s2_day", "east_m", "north_m",
                                                    "colour_err_current", "colour_err_new", "veg_green_current_pct",
                                                    "veg_green_new_pct")} for r in rows]))
    bl.log(f"\n{len(rows)} panels in {out}, and:\n   {out / 'final_sheet.png'}\n   {out / 'final_map.html'}")


if __name__ == "__main__":
    main()
