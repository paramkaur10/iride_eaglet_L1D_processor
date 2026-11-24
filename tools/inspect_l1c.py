#!/usr/bin/env python3
"""
inspect_l1c.py: what is in the L1C folder, batch by batch, and whether OHB's radiometric coefficients hang together.

For every EAGLET-2 L1C zip it reads the product XML (processing baseline, processor version, the AXE1A calibration
file, GAIN / OFFSET, solar irradiance, Earth-Sun factor U, sun elevation, reflectance outliers) and a small copy of
the image (median DN, saturation, median reflectance with the XML gains). From the XML it works out each band's
radiance gain, GAIN x E_sun x U x sin(sun elevation) / pi: a property of each camera, so it should be the same for
every product of a satellite once the coefficients are right. U is checked against the date (U = 1 / d^2).

Products are grouped by configuration (baseline, processor version, AXE1A file). The report ends with the
build_lut.py command for the newest configuration.

    python tools/inspect_l1c.py                 # every zip in ../OG_data; results in ../l1c_inspect
"""

import argparse
import csv
import json
import math
import re
import shutil
import sys
import xml.etree.ElementTree as ET
import zipfile
from datetime import datetime
from pathlib import Path

import numpy as np
import rasterio

sys.path.insert(0, str(Path(__file__).resolve().parent))
import build_lut as bl                                         # noqa: E402

BANDS = (1, 2, 3)
STAMP = re.compile(r"_(\d{8}T\d{6})_+O_\d{3}")                   # the creation time in an AUX file name


def local(tag):
    return tag.split("}")[-1] if isinstance(tag, str) else ""


def xml_facts(xml_bytes):
    """the radiometric facts of the product XML (None where an element is missing)"""
    root = ET.fromstring(xml_bytes)
    first = lambda name: next(((el.text or "").strip() for el in root.iter() if local(el.tag) == name), None)
    per_band = lambda name: {int(el.get("Band_ID")): float(el.text) for el in root.iter()
                             if local(el.tag) == name and el.get("Band_ID") and (el.text or "").strip()}
    out = dict(baseline=first("PROCESSING_BASELINE"), processor=first("Processor_Version"),
               spacecraft=first("SPACECRAFT_NAME"), generated=first("GENERATION_TIME"),
               aux=[(el.text or "").strip() for el in root.iter() if local(el.tag) == "AUXILIARY_FILE"],
               U=float(first("U")) if first("U") else None,
               esun=per_band("SOLAR_IRRADIANCE"),
               sun_elev=float(first("SOLAR_ELEVATION_ANGLE")) if first("SOLAR_ELEVATION_ANGLE") else None,
               below0={}, above1={})
    for el in root.iter():                                     # NUMBER_OF_OUTLIERS Band_ID=n: LOWER_THAN / GREATER_THAN
        if local(el.tag) != "NUMBER_OF_OUTLIERS" or not el.get("Band_ID"):
            continue
        for part in el:
            pct = next((float(x.text) for x in part.iter() if local(x.tag) == "PERCENTAGE" and x.text), None)
            if pct is not None and local(part.tag) == "LOWER_THAN":
                out["below0"][int(el.get("Band_ID"))] = pct
            elif pct is not None and local(part.tag) == "GREATER_THAN":
                out["above1"][int(el.get("Band_ID"))] = pct
    return out


def expected_u(when):
    """Earth-Sun distance factor 1/d^2 for a date, as in Sentinel-2 L1C metadata"""
    doy = when.timetuple().tm_yday
    return 1.0 / (1.0 - 0.01672 * math.cos(math.radians(0.9856 * (doy - 4)))) ** 2


def dn_stats(tif, cache):
    """median, 99th percentile and maximum DN per band, and the share of pixels at the maximum, on a small copy"""
    if cache.exists():
        return json.loads(cache.read_text())
    with rasterio.open(tif) as src:
        f = max(1, max(src.width, src.height) // 1500)
        a = src.read(out_shape=(3, max(1, src.height // f), max(1, src.width // f)))
    valid = (a > 0).all(0)
    out = {}
    for b in range(3):
        v = a[b][valid]
        if v.size:
            top = int(v.max())
            out[f"b{b + 1}"] = dict(p50=float(np.median(v)), p99=float(np.percentile(v, 99)), max=top,
                                    at_max_pct=round(100.0 * float((v >= top).mean()), 3))
    cache.write_text(json.dumps(out))
    return out


def inspect_one(z, out_dir, reader):
    """everything about one zip, or a note on why it cannot be read"""
    m = bl.NAME.match(z.stem)
    row = dict(product=z.stem, sat=m.group(1), sensing=m.group(2), generated=m.group(4), baseline=m.group(8),
               note="")
    try:
        p = bl.read_product(z, out_dir / "xml", reader)
    except zipfile.BadZipFile:
        row["note"] = f"not a complete zip ({z.stat().st_size / 1e6:.0f} MB): an unfinished upload or download?"
        return row
    except Exception as e:
        row["note"] = f"cannot be read: {type(e).__name__}: {e}"
        return row
    x = xml_facts((out_dir / "xml" / f"{z.stem}.xml").read_bytes())
    with zipfile.ZipFile(z) as zf:
        axe = sorted(n for n in zf.namelist() if "AXE1A" in Path(n).name)
        for n in axe:                                          # a copy of every calibration file, once
            dest = out_dir / "axe1a" / Path(n).name
            if not dest.exists():
                dest.parent.mkdir(exist_ok=True)
                with zf.open(n) as src, dest.open("wb") as dst:
                    shutil.copyfileobj(src, dst)
    axe_names = sorted({Path(n).stem for n in axe} | {a for a in x["aux"] if "AXE1A" in a})
    stamps = sorted({s for a in axe_names for s in STAMP.findall(a)})
    when = datetime.strptime(row["sensing"], "%Y%m%dT%H%M%S")
    u_exp = expected_u(when)
    row.update(processor=x["processor"] or "?", axe1a=" ".join(stamps) or "-", axe1a_files=" ".join(axe_names),
               xml_baseline=x["baseline"] or "?", spacecraft=x["spacecraft"] or "?",
               sun_elev=x["sun_elev"], U=x["U"], U_expected=round(u_exp, 6),
               U_off_pct=round(100 * (x["U"] / u_exp - 1), 2) if x["U"] else None,
               calibration="ok" if p["calibration_ok"] else "LOOKS WRONG", gsd_m=round(p["res"], 2),
               size_px=f"{p['width']} x {p['height']}")
    stats = dn_stats(p["tif"], out_dir / "xml" / f"{z.stem}.dn.json")
    sin_e = math.sin(math.radians(x["sun_elev"])) if x["sun_elev"] is not None else None
    for b in BANDS:
        g, o = p["gains"][b - 1], p["offsets"][b - 1]
        e = x["esun"].get(b)
        s = stats.get(f"b{b}", {})
        row.update({f"gain_b{b}": g, f"offset_b{b}": o, f"esun_b{b}": e,
                    f"radiance_gain_b{b}": (g * e * x["U"] * sin_e / math.pi
                                            if None not in (e, x["U"], sin_e) else None),
                    f"dn_median_b{b}": s.get("p50"), f"dn_p99_b{b}": s.get("p99"), f"dn_max_b{b}": s.get("max"),
                    f"at_max_pct_b{b}": s.get("at_max_pct"), f"refl_median_b{b}": p["refl_median"][b - 1],
                    f"below0_pct_b{b}": x["below0"].get(b), f"above1_pct_b{b}": x["above1"].get(b)})
    return row


def f(v, nd=3):
    if v is None or (isinstance(v, float) and not math.isfinite(v)):
        return "-"
    if isinstance(v, float):
        if v != 0 and (abs(v) < 1e-2 or abs(v) >= 1e4):
            return f"{v:.{nd}g}"
        t = f"{v:.{nd}f}".rstrip("0").rstrip(".")
        return "0" if t in ("-0", "") else t
    return str(v)


def table(rows, cols, heads=None, digits=None):
    heads, digits = heads or cols, digits or {}
    cells = [list(heads)] + [[f(r.get(c), digits.get(c, 3)) for c in cols] for r in rows]
    w = [max(len(c[i]) for c in cells) for i in range(len(cols))]
    return "\n".join("  ".join(v.ljust(w[i]) if i == 0 else v.rjust(w[i]) for i, v in enumerate(c)).rstrip()
                     for c in cells)


def spread(values):
    v = [x for x in values if x]
    return max(v) / min(v) if len(v) > 1 and min(v) > 0 else None


def report(rows, products_dir):
    good = [r for r in rows if not r["note"]]
    bad = [r for r in rows if r["note"]]
    configs = {}
    for r in good:
        configs.setdefault((r["baseline"], r["processor"], r["axe1a"]), []).append(r)
    order = sorted(configs, key=lambda k: max(r["generated"] for r in configs[k]))
    lines = [f"L1C PRODUCTS IN {products_dir}: {len(rows)} zips, {len(order)} configuration(s)", ""]
    for i, key in enumerate(order, 1):
        rs = sorted(configs[key], key=lambda r: (r["sat"], r["sensing"]))
        gen = sorted(r["generated"][:8] for r in rs)
        lines += [f"CONFIGURATION {i}: baseline {key[0]}, processor {key[1]}, AXE1A {key[2]} "
                  f"({len(rs)} products, generated {gen[0]} to {gen[-1]})"]
        for r in rs:
            r["label"] = f"{r['sat']} {r['sensing'][:13]}"
        lines.append(table(rs, ["label", "generated", "sun_elev", "U", "U_off_pct", "gain_b1", "gain_b2", "gain_b3",
                                "radiance_gain_b1", "radiance_gain_b2", "radiance_gain_b3"],
                           ["product", "generated", "sun_elev", "U", "U_off_%", "GAIN_1", "GAIN_2", "GAIN_3",
                            "rad_gain_1", "rad_gain_2", "rad_gain_3"], dict(U=4, sun_elev=1, U_off_pct=1)))
        lines.append("")
        lines.append(table(rs, ["label", "dn_median_b1", "dn_median_b2", "dn_median_b3", "refl_median_b1",
                                "refl_median_b2", "refl_median_b3", "calibration", "above1_pct_b1", "above1_pct_b2",
                                "above1_pct_b3", "at_max_pct_b1", "at_max_pct_b2", "at_max_pct_b3"],
                           ["product", "DN_1", "DN_2", "DN_3", "refl_1", "refl_2", "refl_3", "calibration",
                            "refl>1_%_1", "refl>1_%_2", "refl>1_%_3", "DN_at_max_%_1", "_2", "_3"]))
        lines.append("")

    lines += ["RADIANCE GAIN PER SATELLITE (GAIN x E_sun x U x sin(sun elevation) / pi, W/m2/sr/um per DN: a property "
              "of the camera, the same for every product of a satellite when the coefficients hang together; spread "
              "= highest / lowest product)"]
    srows = []
    for i, key in enumerate(order, 1):
        for sat in sorted({r["sat"] for r in configs[key]}):
            mine = [r for r in configs[key] if r["sat"] == sat]
            row = dict(config=str(i), satellite=sat, products=str(len(mine)))
            for b in BANDS:
                v = [r[f"radiance_gain_b{b}"] for r in mine if r.get(f"radiance_gain_b{b}")]
                row[f"b{b}"] = float(np.median(v)) if v else None
                row[f"spread_b{b}"] = spread(v)
            srows.append(row)
    lines.append(table(srows, ["config", "satellite", "products", "b1", "spread_b1", "b2", "spread_b2", "b3",
                               "spread_b3"]))
    sat_cfgs = {}
    for row in srows:
        sat_cfgs.setdefault(row["satellite"], []).append(row)
    changes = []
    for sat, rs in sorted(sat_cfgs.items()):
        for a, b in zip(rs, rs[1:]):
            changes.append(dict(satellite=sat, configs=f"{a['config']} -> {b['config']}",
                                **{f"b{k}": (b[f"b{k}"] / a[f"b{k}"] if a[f"b{k}"] and b[f"b{k}"] else None)
                                   for k in BANDS}))
    if changes:
        lines += ["", "RADIANCE GAIN CHANGE BETWEEN CONFIGURATIONS (new / old, per satellite)",
                  table(changes, ["satellite", "configs", "b1", "b2", "b3"])]
    else:
        lines += ["", "(no satellite in more than one configuration: the radiance gains cannot be compared over time)"]

    same = {}
    for r in good:
        same.setdefault((r["sat"], r["sensing"]), []).append(r)
    pairs = []
    for (sat, sensing), rs in sorted(same.items()):
        rs = sorted(rs, key=lambda r: r["generated"])
        for a, b in zip(rs, rs[1:]):
            if (a["baseline"], a["processor"], a["axe1a"]) == (b["baseline"], b["processor"], b["axe1a"]):
                continue
            pairs.append(dict(image=f"{sat} {sensing[:13]}", processed=f"{a['generated'][:8]} -> {b['generated'][:8]}",
                              **{f"refl_b{k}": (b[f"refl_median_b{k}"] / a[f"refl_median_b{k}"]
                                                if a[f"refl_median_b{k}"] else None) for k in BANDS},
                              **{f"rad_gain_b{k}": (b[f"radiance_gain_b{k}"] / a[f"radiance_gain_b{k}"]
                                                    if a.get(f"radiance_gain_b{k}") and b.get(f"radiance_gain_b{k}")
                                                    else None) for k in BANDS}))
    if pairs:
        lines += ["", "SAME IMAGE PROCESSED AGAIN (new / old: median reflectance and radiance gain per band)",
                  table(pairs, ["image", "processed", "refl_b1", "refl_b2", "refl_b3", "rad_gain_b1", "rad_gain_b2",
                                "rad_gain_b3"])]

    flags = []
    for r in good:
        if r["calibration"] != "ok":
            flags.append(f"{r['product']}: median reflectance {f(r['refl_median_b1'])} / {f(r['refl_median_b2'])} / "
                         f"{f(r['refl_median_b3'])} with the XML gains: the calibration looks wrong")
        if r.get("U_off_pct") is not None and abs(r["U_off_pct"]) > 1:
            flags.append(f"{r['product']}: U = {r['U']} in the XML, {r['U_expected']} for the date "
                         f"({r['U_off_pct']:+.1f}%): every reflectance is off by that much")
        if r["xml_baseline"] not in ("?", r["baseline"]):
            flags.append(f"{r['product']}: the XML says baseline {r['xml_baseline']}, the name {r['baseline']}")
        top = max([r.get(f"at_max_pct_b{b}") or 0 for b in BANDS])
        if top > 1:
            flags.append(f"{r['product']}: {top:.1f}% of the pixels at the highest DN of a band: saturated?")
    lines += ["", "TO LOOK AT"] + ([f"  {x}" for x in flags] or ["  nothing"])
    if bad:
        lines += ["", "NOT READ"] + [f"  {r['product']}: {r['note']}" for r in bad]

    if order:
        key = order[-1]
        since = min(r["generated"][:8] for r in configs[key])
        later_same = [r for k in order[:-1] if k[0] == key[0] for r in configs[k] if r["generated"][:8] >= since]
        chips = next((c for c in ("s2_chips", "lut_a01/s2_chips") if (bl.PROJECT / c / "chips.csv").exists()), None)
        cmd = (f"python tools/build_lut.py --baseline {key[0]} --since {since} --work ../lut_{key[0].lower()}_{since}"
               + (f" --chips ../{chips}" if chips else ""))
        lines += ["", f"NEWEST CONFIGURATION: {len(order)} (baseline {key[0]}, processor {key[1]}, AXE1A {key[2]}). "
                      "To build the LUT on it:", f"  {cmd}"]
        if later_same:
            lines.append(f"  (careful: {len(later_same)} product(s) of another configuration with baseline {key[0]} "
                         f"were also generated from {since} on, and build_lut.py would take them too)")
    return "\n".join(lines) + "\n"


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--products", type=Path, default=bl.PROJECT / "OG_data", help="L1C zips (default: %(default)s)")
    ap.add_argument("--out", type=Path, default=bl.PROJECT / "l1c_inspect", help="results (default: %(default)s)")
    a = ap.parse_args(argv)
    (a.out / "xml").mkdir(parents=True, exist_ok=True)
    reader = bl.processor_xml_reader()
    zips = sorted(z for z in a.products.rglob("IME*_1CSP__RGB3_*.zip") if bl.NAME.match(z.stem))
    if not zips:
        raise SystemExit(f"no EAGLET-2 L1C zip (IMEnn_1CSP__RGB3_*.zip) in {a.products}")
    rows = []
    for z in zips:
        print(f"reading {z.name}", flush=True)
        rows.append(inspect_one(z, a.out, reader))
    cols = list(dict.fromkeys(k for r in rows for k in r))
    with (a.out / "inspect.csv").open("w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=cols)
        w.writeheader()
        w.writerows(rows)
    text = report(rows, a.products)
    (a.out / "inspect.txt").write_text(text)
    print("\n" + text)
    print(f"results: {a.out / 'inspect.txt'}, inspect.csv, the AXE1A files in {a.out / 'axe1a'}")


if __name__ == "__main__":
    main()
