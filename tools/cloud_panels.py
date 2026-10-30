#!/usr/bin/env python3
"""
cloud_panels.py: for the scenes given, side by side, the whole scene on top and 2 km at full resolution below:
    L1C as delivered | L1D tagged --a | L1D tagged --b | what --b changed against --a | Sentinel-2 L1C
The 2 km are where the --a L1D shows the most cyan cloud; where the two L1Ds differ the most when it shows none; the
middle of the scene when they do not differ. Cyan: red below 0.3, green and blue above (as on the 0 to 0.3 display).
The changes: black where both L1Ds are the same, brighter the bigger the change (full at 0.1 reflectance in a band);
magenta where the cloud confidence (the COM in the --b L1D) is 0.3 or more, orange where it is less (bright pixels
the cloud mask does not call cloud). Every image: reflectance 0 to --top in every band (0.3 by default; a higher
--top shows the colours of bright cloud, snow and desert). Writes <work>/cloud_panels/<scene>.png, or
<scene>_top<top>.png with another --top.
    python tools/cloud_panels.py --work /lut/work_v4 --runs /runs --a LUT_V4 --b V4_CLOUD 12 14 18 5
    python tools/cloud_panels.py --work /lut/work_v4 --runs /runs --a LUT_V4 --b V4_CLOUD --top 0.8 20 14 18 12 5
"""
import argparse
import re
import sys
import zipfile
from pathlib import Path

import numpy as np
import rasterio
from PIL import Image, ImageDraw
from rasterio.enums import Resampling
from rasterio.transform import Affine
from rasterio.windows import Window
from scipy import ndimage

sys.path.insert(0, str(Path(__file__).resolve().parent))
import build_lut as bl  # noqa: E402
import lut_panels as lp  # noqa: E402

CYAN_AT = 0.3  # cyan: red below, green and blue above (as on the 0 to 0.3 display)
CHANGED = 0.02  # a pixel counts as changed when a band differs by more than this
CHANGE_FULL = 0.1  # a change this big or bigger is shown at full colour
CONF_CLOUD = 0.3  # cloud confidence from which the experiment pulls a pixel towards grey
MAGENTA, ORANGE = np.array([255, 0, 255]), np.array([255, 150, 0])


def cyan(rho, valid):
    return valid & (rho[0] < CYAN_AT) & (rho[1] >= CYAN_AT) & (rho[2] >= CYAN_AT)


def com_member(zpath):
    """the cloud confidence map (COM) in the L1D zip, as a GDAL path; None when there is none"""
    with zipfile.ZipFile(zpath) as z:
        names = sorted(n for n in z.namelist() if re.search(r"/AUX_DATA/[^/]+\.COM\.tif$", n))
    return f"/vsizip/{zpath}/{names[0]}" if names else None


def read_conf(path, window=None, out_shape=None):
    how = Resampling.average if out_shape else Resampling.nearest
    with rasterio.open(path) as src:
        c = src.read(1, window=window, out_shape=out_shape, resampling=how)
    return c.astype("float64") / 10000.0


def changes(a, b, conf):
    """(changed, cloud, valid, picture): the pixels that differ by more than CHANGED in a band, those with a cloud
    confidence of CONF_CLOUD or more, the pixels valid in both, and the picture of the changes"""
    (ra, va), (rb, vb) = a, b
    valid = va & vb
    d = np.where(valid, np.abs(rb - ra).max(0), 0.0)
    cloud = conf >= CONF_CLOUD if conf is not None else np.zeros(valid.shape, bool)
    x = np.clip(d / CHANGE_FULL, 0, 1)[..., None]
    rgb = np.rint(x * np.where(cloud[..., None], MAGENTA, ORANGE)).astype("uint8")
    rgb[~valid] = 255
    return valid & (d > CHANGED), cloud, valid, Image.fromarray(rgb)


def one(p, work, out, runs, tag_a, tag_b, cdse, suffix):
    with rasterio.open(p["tif"]) as src:
        H, W, tf = src.height, src.width, src.transform
    oh = max(1, int(round(lp.PANEL * H / W)))
    disp_tf = tf * Affine.scale(W / lp.PANEL, H / oh)
    name_a, name_b = tag_a or "current LUT", tag_b or "current LUT"
    za, zb = runs.get(tag_a), runs.get(tag_b)
    paths = {"L1C": p["tif"], "a": lp.image_member(za) if za else None, "b": lp.image_member(zb) if zb else None}
    kind = {"L1C": "L1C", "a": "L1D", "b": "L1D"}
    com = com_member(zb) if zb else None
    small = {k: lp.read_rho(v, kind[k], p, out_shape=(3, oh, lp.PANEL)) for k, v in paths.items() if v}
    conf = read_conf(com, out_shape=(oh, lp.PANEL)) if com else None
    ch = changes(small["a"], small["b"], conf) if "a" in small and "b" in small else None

    zoom_on, fx, fy = "the middle of the scene", 0.5, 0.5
    size = max(3, int(round(lp.ZOOM * lp.PANEL / W)))  # 2 km in pixels of the whole-scene picture
    for what, m in ((f"where {name_a} shows the most cyan", cyan(*small["a"]) if "a" in small else None),
                    ("where the two L1Ds differ the most", ch[0] if ch else None)):
        if m is not None and m.sum() >= 20:
            dens = ndimage.uniform_filter(m.astype("float64"), size, mode="constant")
            r, c = np.unravel_index(np.argmax(dens), dens.shape)
            zoom_on, fx, fy = what, (c + 0.5) / lp.PANEL, (r + 0.5) / oh
            break
    col0 = int(np.clip(fx * W - lp.ZOOM / 2, 0, max(W - lp.ZOOM, 0)))
    row0 = int(np.clip(fy * H - lp.ZOOM / 2, 0, max(H - lp.ZOOM, 0)))
    zw, zh = min(lp.ZOOM, W - col0), min(lp.ZOOM, H - row0)
    win = Window(col0, row0, zw, zh)
    win_tf = tf * Affine.translation(col0, row0)
    big = {k: lp.read_rho(v, kind[k], p, window=win) for k, v in paths.items() if v}
    eag = work / "eag10" / f"{p['stem']}.npz"
    fix = Affine(*np.load(eag)["fix"]) if eag.exists() else Affine.identity()

    def zoomed(img):
        return img.resize((lp.PANEL, lp.PANEL), Image.LANCZOS)

    gone = "no L1D in the runs folder"
    missing = (lp.blank((lp.PANEL, oh), gone), lp.blank((lp.PANEL, lp.PANEL), gone))
    titles, tops, bottoms = ["L1C, as delivered"], [lp.to_rgb(*small["L1C"])], [zoomed(lp.to_rgb(*big["L1C"]))]
    shares, stats = {}, None
    for k, name in (("a", name_a), ("b", name_b)):
        if k not in small:
            titles.append(f"L1D, {name}")
            tops.append(missing[0])
            bottoms.append(missing[1])
            continue
        valid = small[k][1]
        shares[name] = 100 * float(cyan(*small[k])[valid].mean()) if valid.any() else 0.0
        titles.append(f"L1D, {name} (cyan {shares[name]:.1f}%)")
        tops.append(lp.to_rgb(*small[k]))
        bottoms.append(zoomed(lp.to_rgb(*big[k])))
    if ch:
        changed, cloud, valid, picture = ch
        n = max(int(valid.sum()), 1)
        stats = (100 * float((changed & cloud).sum()) / n, 100 * float((changed & ~cloud).sum()) / n)
        titles.append(f"changed: cloud {stats[0]:.1f}%, bright {stats[1]:.1f}%")
        tops.append(picture)
        bottoms.append(zoomed(changes(big["a"], big["b"], read_conf(com, window=win) if com else None)[3]))
    else:
        titles.append("changed")
        tops.append(missing[0])
        bottoms.append(missing[1])
    final = work / "final"  # lut_panels.py's folder: the Sentinel-2 days it fetched for pictures
    s2_path, day = lp.s2_day(p, work, final if final.is_dir() else out, cdse)
    if s2_path is None:
        titles.append("Sentinel-2 L1C: none within 6 days")
        tops.append(lp.blank((lp.PANEL, oh), "no Sentinel-2 within 6 days"))
        bottoms.append(lp.blank((lp.PANEL, lp.PANEL), "no Sentinel-2 within 6 days"))
    else:
        titles.append(f"Sentinel-2 L1C, {day['day']} ({day['dt_days']:+.1f} d)")
        tops.append(lp.s2_on(s2_path, fix, disp_tf, (oh, lp.PANEL), Resampling.average))
        s2_zoom = lp.s2_on(s2_path, fix, win_tf, (zh, zw), Resampling.nearest)
        bottoms.append(s2_zoom.resize((lp.PANEL, lp.PANEL), Image.NEAREST))
    notes = [
        p["stem"],
        f"Cyan (share of the scene in the titles): red below {CYAN_AT}, green and blue above. Bottom row: 2 km at "
        f"full resolution (red box), {zoom_on}.",
        f"Changed: black = the same in both L1Ds, brighter = a bigger change (full colour at {CHANGE_FULL} in a "
        f"band); magenta where the cloud confidence is {CONF_CLOUD} or more, orange where it is less (bright "
        f"pixels). Percentages: pixels changed by more than {CHANGED} in a band.",
        f"Every image: reflectance 0 to {lp.TOP:g} in every band.",
    ]
    pad, gap, head, cap, line = 20, 16, 60, 34, 26
    n_col = len(tops)
    width = 2 * pad + n_col * lp.PANEL + (n_col - 1) * gap
    height = head + cap + oh + gap + lp.PANEL + 14 + line * len(notes) + pad
    sheet = Image.new("RGB", (width, height), "white")
    d = ImageDraw.Draw(sheet)
    d.text((pad, 16), f"{p['label']} | clouds: {name_a} against {name_b}", fill="black", font=lp.font(26))
    s = lp.PANEL / W
    for i, (title, top, bottom) in enumerate(zip(titles, tops, bottoms)):
        x = pad + i * (lp.PANEL + gap)
        d.text((x, head), title, fill="#333333", font=lp.font(20))
        sheet.paste(top, (x, head + cap))
        sheet.paste(bottom, (x, head + cap + oh + gap))
        d.rectangle([x + col0 * s, head + cap + row0 * s, x + (col0 + zw) * s, head + cap + (row0 + zh) * s],
                    outline="red", width=2)
    y = head + cap + oh + gap + lp.PANEL + 14
    for text in notes:
        d.text((pad, y), text, fill="#555555", font=lp.font(17))
        y += line
    name = f"{p['label'].split()[0][1:].zfill(2)}_{p['sat']}_{p['sensing']}{suffix}.png"
    sheet.save(out / name)
    return name, shares, stats


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--work", type=Path, required=True, help="build_lut's folder (inventory, eag10, s2)")
    ap.add_argument("--runs", type=Path, required=True, help="the processor's working folders (runs/)")
    ap.add_argument("--a", default="LUT_V4", help="tag of the first L1D column ('' = no tag, the current LUT)")
    ap.add_argument("--b", default="V4_CLOUD", help="tag of the second L1D column")
    ap.add_argument("--top", type=float, default=0.3, help="reflectance shown as white (default: %(default)s)")
    ap.add_argument("scenes", nargs="+", type=int, help="scene numbers, as #12 in the panels")
    a = ap.parse_args(argv)
    lp.TOP = a.top
    suffix = "" if a.top == 0.3 else f"_top{a.top:g}"
    prods = {int(p["label"].split()[0][1:]): p for p in bl.load_inventory(a.work)}
    out = a.work / "cloud_panels"
    out.mkdir(exist_ok=True)
    index = lp.l1d_index(a.runs)
    cdse = bl.Lazy(bl.CDSE)
    for n in a.scenes:
        p = prods.get(n)
        if p is None:
            bl.log(f"#{n}: not in the inventory")
            continue
        try:
            runs = index.get((p["sat"], p["sensing"]), {})
            name, shares, stats = one(p, a.work, out, runs, a.a, a.b, cdse, suffix)
        except Exception as e:
            bl.log(f"!! {p['label']}: no panel: {type(e).__name__}: {e}")
            continue
        text = ", ".join(f"{t} {s:.1f}%" for t, s in shares.items())
        if stats is not None:
            text += f" | changed: cloud {stats[0]:.1f}%, bright {stats[1]:.1f}%"
        bl.log(f"{p['label']}: {out / name} | cyan: {text}")


if __name__ == "__main__":
    main()
