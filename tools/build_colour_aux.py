#!/usr/bin/env python3
"""
build_colour_aux.py: builds the colour-correction AUX file the processor reads
(IME___AXE1D_CLUT_20240101T000000_20990101T000000_<creation>____________O_001.xml,
Colour_Correction version 4) from:
  --lut-aux       the version 3 AUX file written by the lut_build notebook
                  (LUT tables, colour balance, vegetation-around settings)
  --class-model   Enrico's applica_L1C_RGB.py: its MODEL is copied unchanged,
                  as JSON with a SHA-256
  --method        LUT (default), CLASS_MODEL or HYBRID: what the processor
                  applies. It can be changed later by editing <Method> in the
                  file (the tables and the model are checksummed, Method is not)

The new file is read back with the processor's own module and checked: LUT
tables identical to the v3 file, model identical to Enrico's, and his model
giving the same numbers through the processor as through his script (when
his script can be imported here). Nothing is overwritten.

    python tools/build_colour_aux.py --lut-aux <v3.xml> --class-model <applica_L1C_RGB.py> \
        [--method LUT] [--out-dir <folder>]
"""

import argparse
import hashlib
import importlib.util
import json
import re
import sys
import xml.etree.ElementTree as ET
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from iride_eaglet_processor.colour_correction import ColourCorrection, METHODS  # noqa: E402

MODEL_PATTERN = re.compile(r"MODEL\s*=\s*json\.loads\(r'''(.*?)'''\)", re.S)


def read_class_model(script_path):
    """Enrico's MODEL, read from the script's text (no need to import it, nor rasterio)"""
    match = MODEL_PATTERN.search(Path(script_path).read_text())
    if not match:
        raise SystemExit(f"MODEL = json.loads(r'''...''') not found in {script_path}")
    return json.loads(match.group(1))


def set_field(parent, path, text):
    el = parent
    for part in path.split("/"):
        nxt = el.find(part)
        el = nxt if nxt is not None else ET.SubElement(el, part)
    el.text = text


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--lut-aux", type=Path, required=True)
    ap.add_argument("--class-model", type=Path, required=True)
    ap.add_argument("--method", default="LUT", choices=METHODS)
    ap.add_argument("--out-dir", type=Path)
    a = ap.parse_args(argv)

    tree = ET.parse(a.lut_aux)
    root = tree.getroot()
    cc = root.find("Data_Block/Colour_Correction")
    if cc is None or cc.get("version") != "3":
        raise SystemExit(f"{a.lut_aux.name}: expected a version 3 colour AUX file (LUT only)")
    model = read_class_model(a.class_model)

    now = datetime.now(timezone.utc)
    name = f"IME___AXE1D_CLUT_20240101T000000_20990101T000000_{now:%Y%m%dT%H%M%S}____________O_001"
    eeh = root.find("Earth_Explorer_Header")
    if eeh is None:
        eeh = ET.SubElement(root, "Earth_Explorer_Header")
    fh = eeh.find("Fixed_Header")
    if fh is None:
        fh = ET.SubElement(eeh, "Fixed_Header")
    set_field(fh, "File_Name", name)
    set_field(fh, "File_Description", "EAGLET-2 RGB3 colour correction: LUT towards Sentinel-2 L1C and Enrico's "
                                      "class model; Method selects which one the processor applies")
    set_field(fh, "Notes", "One file for the whole constellation. Input: DN x GAIN + OFFSET from the product XML "
                           "(correct from 2026-09-24). LUT and HYBRID keep the brightness as delivered; "
                           "CLASS_MODEL brings it close to Sentinel-2")
    set_field(fh, "Source/Creator", "iride_eaglet_processor tools/build_colour_aux.py")
    set_field(fh, "Source/Creator_Version", f"LUT from {a.lut_aux.name} + class model from {a.class_model.name}")
    set_field(fh, "Source/Creation_Date", f"UTC={now:%Y-%m-%dT%H:%M:%S}")

    cc.set("version", "4")
    method_el = ET.Element("Method", options=" ".join(METHODS))
    method_el.text = a.method
    cc.insert(0, method_el)
    text = json.dumps(model, separators=(",", ":"))
    ET.SubElement(cc, "Class_Model", encoding="json", sha256=hashlib.sha256(text.encode("utf-8")).hexdigest(),
                  source=f"{a.class_model.name} (v2), unchanged",
                  training="one scene: Cape Town, Sentinel-2 S2B 2026-09-26, tile 34HBH",
                  description="water / vegetation / built-up picked from the RGB triplet by Gaussian mixtures "
                              "(8 per class), one transform per class; negatives set to 0").text = text
    ET.indent(tree)
    out_dir = a.out_dir or a.lut_aux.parent
    out_dir.mkdir(parents=True, exist_ok=True)
    out = out_dir / f"{name}.xml"
    if out.exists():
        raise SystemExit(f"{out} exists")
    tree.write(out, encoding="UTF-8", xml_declaration=True)

    # ---- read back with the processor's module and check ----
    v3 = ColourCorrection(a.lut_aux)
    checks = {"Method in the file": ColourCorrection(out).method == a.method}
    lut = ColourCorrection(out, method="LUT")
    checks["LUT identical to the v3 file"] = (np.array_equal(lut.q, v3.q) and np.array_equal(lut.veg_share, v3.veg_share)
                                              and np.array_equal(lut.balance, v3.balance))
    stored = ET.parse(out).getroot().find("Data_Block/Colour_Correction/Class_Model").text
    checks["class model identical to the script's MODEL"] = json.loads(stored) == model
    rs = np.random.default_rng(0)
    rho = rs.uniform(0.01, 1.0, (20000, 3))
    try:
        spec = importlib.util.spec_from_file_location("enrico_rgb", a.class_model)
        enrico = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(enrico)
        his_out, _ = enrico.apply(rho, enrico.MODEL)
        checks["his model: same numbers as his script"] = np.array_equal(
            ColourCorrection(out, method="CLASS_MODEL").apply(rho), np.maximum(his_out, 0))
    except Exception as e:                                   # e.g. rasterio missing where this runs
        print(f"(could not import {a.class_model.name} to compare numbers: {e})")
    print(f"written {out}  ({out.stat().st_size / 1e3:.0f} kB), Method = {a.method}")
    for check, ok in checks.items():
        print(f"   {'ok      ' if ok else 'PROBLEM '} {check}")
    if not all(checks.values()):
        out.unlink()
        raise SystemExit("checks failed, file removed")


if __name__ == "__main__":
    main()