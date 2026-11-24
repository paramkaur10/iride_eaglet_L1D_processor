#!/usr/bin/env python3
"""
compare_l1d.py: two L1D products of the same L1C, made by two processor versions (e.g. 01.01 and 01.02),
compared file by file.

What may differ:
  - always: the product name, the generation time, the processor version, checksums and file sizes, and the
    name of the colour AUX file (from 01.02 the built-in one is valid from 17 Oct 2026; same tables);
  - the cloud mask (CLM), the cloud confidence (COM) and the cloud numbers of the STAC: compared at the end
    (format, class shares, how much the two agree);
  - between 01.01 and a later version only: the L1D image where the later one writes saturated L1C pixels
    white (every pixel that differs must have its three bands equal there), the quicklooks (the new stretch)
    and the STAC display range (renders rescale).
Anything else that differs is reported as unexpected.

    python tools/compare_l1d.py <L1D A> <L1D B>

each an L1D zip, or a folder holding one (e.g. a working directory of docker/test_image.sh). Exits with
0 when nothing unexpected differs. Needs numpy and rasterio, as in the processor's image:
    docker run --rm -i --network none -v <folder A>:/a:ro -v <folder B>:/b:ro \
        ipf-ime-msi-l1d:01.02 python - /a /b < tools/compare_l1d.py
"""
import json
import re
import sys
import warnings
import zipfile
from pathlib import Path

import numpy as np
import rasterio
from rasterio.errors import NotGeoreferencedWarning

warnings.filterwarnings("ignore", category=NotGeoreferencedWarning)     # the quicklooks have no georeferencing

NAMES = {0: "no data", 1: "clear", 2: "thick cloud", 3: "thin cloud", 4: "cloud shadow"}
AUX_NAME = re.compile(rb"IME___AXE1D_CLUT_\d{8}T\d{6}_\d{8}T\d{6}_\d{8}T\d{6}_+O_\d{3}")
# STAC fields that differ between two runs by design (name, times, version, checksums, cloud numbers)
STAC_ALWAYS = re.compile(r"(^id$|properties\.(created|updated|eo:cloud_cover)$|processing:software|"
                         r"file:checksum$|file:size$|classification:classes\.\d+\.percentage$)")
STAC_STRETCH = re.compile(r"renders\..*rescale")     # the display range, from 01.02 on
# XML values that differ between two runs by design: the generation time, the processor version and the image
# checksum (the image itself is compared pixel by pixel), whether its <Checksum> is on one line or several
XML_VOLATILE = [(re.compile(rb"(<GENERATION_TIME>)[^<]*"), rb"\1T"),
                (re.compile(rb"(<Processor_Version>)[^<]*"), rb"\1V"),
                (re.compile(rb"(<Checksum[^>]*>\s*<Image>)\s*[0-9a-fA-F]+\s*(</Image>)"), rb"\1SHA\2")]


def product(path):
    """the L1D zip at path, or the one L1D zip in the folder path: (zip, product ID, {file in it: name in the zip})"""
    path = Path(path)
    if path.is_dir():
        found = sorted(path.glob("IME*_1DSP__RGB3_*.zip"))
        if len(found) != 1:
            sys.exit(f"{path}: {len(found)} L1D zips in it, not one")
        path = found[0]
    z = zipfile.ZipFile(path)
    pid = path.name[:-4]
    files = {n[len(pid) + 1:].replace(pid, "<ID>"): n for n in z.namelist() if not n.endswith("/")}
    return z, pid, files


def version(z, pid):
    """the processor version in the STAC (processing:software)"""
    try:
        software = json.loads(z.read(f"{pid}/{pid}.json"))["properties"]["processing:software"]
        return ", ".join(software.values())
    except (KeyError, ValueError, AttributeError):
        return "?"


def aux_file(z, pid):
    """the colour AUX file the product names in its XML (AUXILIARY_FILE)"""
    try:
        found = re.search(rb"<AUXILIARY_FILE>([^<]*)</AUXILIARY_FILE>", z.read(f"{pid}/{pid}.xml"))
        return found.group(1).decode() if found else "?"
    except KeyError:
        return "?"


def flat(x, prefix=""):
    """{dotted path: value} of a JSON document"""
    out = {}
    if isinstance(x, dict):
        for k, v in x.items():
            out.update(flat(v, f"{prefix}.{k}" if prefix else k))
    elif isinstance(x, list):
        for i, v in enumerate(x):
            out.update(flat(v, f"{prefix}.{i}"))
    else:
        out[prefix] = x
    return out


def raster(z, name):
    with rasterio.open(f"/vsizip/{z.filename}/{name}") as r:
        return r.read(), r.profile


def pixels(za, name_a, zb, name_b):
    """the pixels of two rasters compared band by band: (same size and bands, pixels that differ,
    those pixels have their bands all equal in A, the same in B)"""
    with rasterio.open(f"/vsizip/{za.filename}/{name_a}") as ra, \
            rasterio.open(f"/vsizip/{zb.filename}/{name_b}") as rb:
        if ra.count != rb.count or ra.shape != rb.shape:
            return False, None, False, False
        differ = np.zeros(ra.shape, bool)
        for i in range(1, ra.count + 1):
            differ |= ra.read(i) != rb.read(i)
        n = int(differ.sum())
        if n == 0 or ra.count < 3:
            return True, n, False, False

        def grey(r):         # red = green = blue at every pixel that differs
            red, green, blue = (r.read(i)[differ] for i in (1, 2, 3))
            return bool(np.all((red == green) & (green == blue)))
        return True, n, grey(ra), grey(rb)


def main(a_path, b_path):
    za, ida, fa = product(a_path)
    zb, idb, fb = product(b_path)
    va, vb = version(za, ida), version(zb, idb)
    print(f"A: {ida} (processor {va})\nB: {idb} (processor {vb})")
    print(f"colour AUX file: A {aux_file(za, ida)}\n                 B {aux_file(zb, idb)}\n")
    later = min(va, vb) <= "01.01" < max(va, vb)        # 01.01 against 01.02 or later: white pixels, new stretch
    problems, expected = 0, set()

    only = sorted(set(fa) ^ set(fb))
    print("files: " + ("the same in both" if not only else f"DIFFERENT, only in one: {only}"))
    problems += bool(only)

    for rel in sorted(set(fa) & set(fb)):
        a, b = za.read(fa[rel]), zb.read(fb[rel])
        a_txt, b_txt = (AUX_NAME.sub(b"<AUX>", x.replace(ida.encode(), b"<ID>").replace(idb.encode(), b"<ID>"))
                        for x in (a, b))
        if a_txt == b_txt:
            print(f"  same      {rel}")
        elif rel.endswith(".CLM.tif") or rel.endswith(".COM.tif"):
            print(f"  cloud     {rel} (compared below)")
        elif rel.endswith((".tif", ".png", ".jpg")):                 # the L1D image, the quicklooks: their pixels
            size, n, grey_a, grey_b = pixels(za, fa[rel], zb, fb[rel])
            if not size:
                print(f"  DIFFERENT {rel}: another size or number of bands")
                problems += 1
            elif n == 0:
                print(f"  same px   {rel} (file bytes differ, pixels identical)")
            elif rel.startswith("IMG_DATA/") and later and (grey_a or grey_b):
                print(f"  expected  {rel}: {n} pixels differ, all of them white in {'B' if grey_b else 'A'} "
                      f"(saturated in the L1C)")
                expected.add("the white pixels")
            elif rel.startswith("PREVIEW/") and later:
                print(f"  expected  {rel}: the quicklook stretch of 01.02")
                expected.add("the quicklooks")
            else:
                print(f"  DIFFERENT {rel}: {n} pixels differ")
                problems += 1
        elif rel.endswith(".json") and rel.count("/") == 0:
            sa, sb = flat(json.loads(a_txt)), flat(json.loads(b_txt))
            keys = [k for k in sorted(set(sa) | set(sb)) if sa.get(k) != sb.get(k)]
            stretch = [k for k in keys if later and STAC_STRETCH.search(k)]
            unexpected = [k for k in keys if not STAC_ALWAYS.search(k) and k not in stretch]
            print(f"  {'stac ok' if not unexpected else 'DIFFERENT'} {rel}: {len(keys) - len(unexpected)} fields "
                  f"differ as expected (times, version, checksums, cloud numbers"
                  + (", display range" if stretch else "") + ")"
                  + (f"; UNEXPECTED: {unexpected}" if unexpected else ""))
            problems += bool(unexpected)
            print(f"            eo:cloud_cover: A {sa.get('properties.eo:cloud_cover')}, "
                  f"B {sb.get('properties.eo:cloud_cover')}")
        elif rel.endswith(".xml") and rel.count("/") == 0:
            la, lb = a_txt.decode().splitlines(), b_txt.decode().splitlines()
            diff = sum(x != y for x, y in zip(la, lb)) + abs(len(la) - len(lb))
            na, nb = a_txt, b_txt
            for pattern, value in XML_VOLATILE:
                na, nb = pattern.sub(value, na), pattern.sub(value, nb)
            la, lb = na.decode().splitlines(), nb.decode().splitlines()
            unexpected = [(x.strip(), y.strip()) for x, y in zip(la, lb) if x != y]
            unexpected += [("(number of lines)", len(la) - len(lb))] * (len(la) != len(lb))
            print(f"  {'xml ok' if not unexpected else 'DIFFERENT'}    {rel}: {diff} lines differ"
                  + (f"; UNEXPECTED: {unexpected[:4]}" if unexpected
                     else " (generation time, processor version, image checksum)"))
            problems += bool(unexpected)
        else:
            print(f"  DIFFERENT {rel}")
            problems += 1

    clm = [k for k in fa if k.endswith(".CLM.tif")]
    com = [k for k in fa if k.endswith(".COM.tif")]
    if clm and com and clm[0] in fb and com[0] in fb:
        (ca, pa), (cb, pb) = raster(za, fa[clm[0]]), raster(zb, fb[clm[0]])
        (ma, qa), (mb, qb) = raster(za, fa[com[0]]), raster(zb, fb[com[0]])
        ca, cb, ma, mb = ca[0], cb[0], ma[0], mb[0]
        fmt = all((pa[k] == pb[k]) for k in ("dtype", "width", "height", "crs", "transform", "count")) and \
            all((qa[k] == qb[k]) for k in ("dtype", "width", "height", "crs", "transform", "count"))
        print(f"\ncloud mask: format {'the same' if fmt else 'DIFFERENT'} (CLM {pa['dtype']}, COM {qa['dtype']}, "
              f"{pa['width']} x {pa['height']})")
        problems += not fmt
        valid = ca > 0
        for name, c, m in (("A", ca, ma), ("B", cb, mb)):
            v = c > 0
            rule = np.array_equal(np.isin(c[v], (2, 3)), m[v] >= 5000)
            shares = " · ".join(f"{NAMES[k]} {np.mean(c[v] == k):.1%}" for k in (1, 2, 3, 4))
            print(f"  {name}: {shares}; cloud = COM >= 5000: {'yes' if rule else 'NO'}; values "
                  f"{sorted(int(x) for x in np.unique(c))}")
            problems += not rule
        same_nodata = np.array_equal(ca == 0, cb == 0)
        print(f"  no data in the same place: {'yes' if same_nodata else 'NO'}")
        problems += not same_nodata
        agree = np.mean(ca[valid] == cb[valid])
        cloud_a, cloud_b = np.isin(ca[valid], (2, 3)), np.isin(cb[valid], (2, 3))
        both = np.sum(cloud_a & cloud_b)
        print(f"  same class: {agree:.1%} of the valid pixels; cloud or not: "
              f"{np.mean(cloud_a == cloud_b):.1%} the same (cloud in both: {both / max(cloud_a.sum(), 1):.1%} of "
              f"A's cloud, {both / max(cloud_b.sum(), 1):.1%} of B's)")
        print("  classes, A (rows) against B (columns), % of the valid pixels:")
        print("                 " + "".join(f"{NAMES[k]:>14}" for k in (1, 2, 3, 4)))
        for i in (1, 2, 3, 4):
            row = [np.mean((ca[valid] == i) & (cb[valid] == j)) for j in (1, 2, 3, 4)]
            print(f"  {NAMES[i]:>14} " + "".join(f"{r:>14.1%}" for r in row))
        fa_, fb_ = ma[valid].astype("float64") / 1e4, mb[valid].astype("float64") / 1e4
        corr = np.corrcoef(fa_, fb_)[0, 1] if fa_.std() > 0 and fb_.std() > 0 else float("nan")
        print(f"  COM: mean A {fa_.mean():.3f}, B {fb_.mean():.3f}; mean difference {np.abs(fa_ - fb_).mean():.3f}; "
              f"correlation {corr:.3f}")

    print()
    if problems:
        print(f"{problems} UNEXPECTED DIFFERENCE(S)")
    elif expected:
        print("AS EXPECTED: BESIDES THE CLOUD MASK, ONLY " + " AND ".join(sorted(expected)).upper() + " DIFFER")
    else:
        print("EVERYTHING BUT THE CLOUD MASK IS THE SAME")
    return 0 if problems == 0 else 1


if __name__ == "__main__":
    if len(sys.argv) != 3:
        sys.exit(__doc__)
    sys.exit(main(sys.argv[1], sys.argv[2]))
