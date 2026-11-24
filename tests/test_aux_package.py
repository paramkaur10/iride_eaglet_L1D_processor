#!/usr/bin/env python3
"""
The colour AUX file in every form the processor may get it (local test, not part of the repository):
the bare XML (the processor's own in aux/), the zip PDGS hands AUX files over in (<ID>.zip holding
<ID>/<ID>.xml and <ID>/<ID>.hdr, first PDGS run with an AUX file, 20 Oct 2026), the same zip unzipped
(a folder), and the ways each can be broken. Every form must give exactly the same colour correction.

    python tests/test_aux_package.py          (or: python -m pytest tests/test_aux_package.py)
"""

import shutil
import sys
import tempfile
import xml.etree.ElementTree as ET
import zipfile
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))

from iride_eaglet_processor.colour_correction import (  # noqa: E402
    ColourCorrection, ColourCorrectionError, aux_id, read_aux_xml)

AUX_XML = sorted((REPO / "aux").glob("IME___AXE1D_CLUT_*.xml"))[-1]
ID = AUX_XML.stem
HDR = (f'<?xml version="1.0" encoding="UTF-8"?>\n<Earth_Explorer_Header><Fixed_Header><File_Name>{ID}'
       f'</File_Name></Fixed_Header></Earth_Explorer_Header>\n').encode()


def pdgs_zip(folder, members=None, folder_entry=True, name=f"{ID}.zip"):
    """a zip as PDGS packs an AUX file: the folder entry <ID>/, then <ID>/<ID>.xml and <ID>/<ID>.hdr"""
    folder = Path(folder)
    folder.mkdir(parents=True, exist_ok=True)
    if members is None:
        members = {f"{ID}/{ID}.xml": AUX_XML.read_bytes(), f"{ID}/{ID}.hdr": HDR}
    path = folder / name
    with zipfile.ZipFile(path, "w", zipfile.ZIP_DEFLATED) as zf:
        if folder_entry:
            zf.writestr(zipfile.ZipInfo(f"{ID}/"), b"")
        for member, data in members.items():
            zf.writestr(member, data)
    return path


def same(a, b):
    """two ColourCorrections that correct colours exactly alike"""
    assert (a.method, a.version, a.around_cells, a.cell_m, a.axes) == \
           (b.method, b.version, b.around_cells, b.cell_m, b.axes)
    assert np.array_equal(a.q, b.q) and np.array_equal(a.veg_share, b.veg_share)
    assert np.array_equal(a.balance, b.balance)
    rho = np.random.default_rng(0).uniform(0.02, 0.4, (500, 3))
    around = np.linspace(0, 1, 500)
    assert np.array_equal(a.apply(rho, around), b.apply(rho, around))


def expect_error(path, *words):
    """ColourCorrection(path) must stop with a ColourCorrectionError that says all of words"""
    try:
        ColourCorrection(path)
    except ColourCorrectionError as e:
        for w in words:
            assert w in str(e), f"{w!r} not in the error: {e}"
        return str(e)
    raise AssertionError(f"no error for {path}")


def test_names():
    assert aux_id(AUX_XML) == ID
    assert aux_id(Path("/wd") / f"{ID}.zip") == ID
    assert aux_id(Path("/wd") / f"{ID}.ZIP") == ID
    assert aux_id(Path("/wd") / ID) == ID


def test_every_form_gives_the_same_correction():
    ref = ColourCorrection(AUX_XML)
    assert ref.name == ID and ref.source == AUX_XML.name
    with tempfile.TemporaryDirectory() as tmp:
        t = Path(tmp)
        z = pdgs_zip(t / "pdgs")                                 # as in the PDGS run of 20 Oct 2026
        cz = ColourCorrection(z)
        same(ref, cz)
        assert cz.name == ID and cz.source == f"{ID}.zip ({ID}/{ID}.xml)"
        assert ID in cz.describe() and ".zip" not in cz.describe()

        with zipfile.ZipFile(z) as zf:                          # the same zip, unzipped: the folder <ID>/
            zf.extractall(t / "unzipped")
        cd = ColourCorrection(t / "unzipped" / ID)
        same(ref, cd)
        assert cd.name == ID and cd.source == f"{ID} ({ID}.xml)"

        flat = pdgs_zip(t / "flat", {f"{ID}.xml": AUX_XML.read_bytes(), f"{ID}.hdr": HDR}, folder_entry=False)
        same(ref, ColourCorrection(flat))                        # a zip without the folder level

        renamed = t / "renamed.zip"                              # any zip name: read by its content,
        shutil.copy(z, renamed)                                  # named after the zip
        assert ColourCorrection(renamed).name == "renamed"

        no_suffix = t / ID                                       # a zip without .zip in its name
        shutil.copy(z, no_suffix)
        same(ref, ColourCorrection(no_suffix))


def test_several_xml_files():
    with tempfile.TemporaryDirectory() as tmp:
        t = Path(tmp)
        two = pdgs_zip(t / "named", {f"{ID}/{ID}.xml": AUX_XML.read_bytes(), f"{ID}/notes.xml": b"<notes/>",
                                     f"{ID}/{ID}.hdr": HDR, f"__MACOSX/{ID}/._{ID}.xml": b"junk"})
        same(ColourCorrection(AUX_XML), ColourCorrection(two))  # the one named <ID>.xml
        bad = pdgs_zip(t / "unnamed", {f"{ID}/a.xml": b"<a/>", f"{ID}/b.xml": b"<b/>"})
        expect_error(bad, "expected one .xml file in it, found 2", "a.xml", "b.xml")


def test_broken_packages():
    with tempfile.TemporaryDirectory() as tmp:
        t = Path(tmp)
        expect_error(pdgs_zip(t / "hdr_only", {f"{ID}/{ID}.hdr": HDR}),
                     "expected one .xml file in it, found 0", f"{ID}/{ID}.hdr")
        cut = AUX_XML.read_bytes()[:5000]                       # an XML cut short, inside the zip
        expect_error(pdgs_zip(t / "cut", {f"{ID}/{ID}.xml": cut}), "Cannot read colour AUX file",
                     f"({ID}/{ID}.xml)")
        whole = pdgs_zip(t / "whole").read_bytes()
        damaged = t / "damaged" / f"{ID}.zip"                   # an unfinished copy: not a zip any more
        damaged.parent.mkdir()
        damaged.write_bytes(whole[: len(whole) // 2])
        expect_error(damaged, "Cannot read colour AUX file")
        expect_error(t / "missing.zip", "Cannot read colour AUX file", "missing.zip")
        expect_error(t / "whole", "expected one .xml file in it, found 0")   # a folder holding only the zip
        no_data = t / "no_data.xml"
        no_data.write_text("<Earth_Explorer_File><Data_Block/></Earth_Explorer_File>")
        expect_error(no_data, "no Data_Block/Colour_Correction")


def test_what_01_00_did_with_the_pdgs_zip():
    """01.00 parsed the path given as XML: the PDGS zip gives exactly the error of the PDGS log of 20 Oct 2026"""
    with tempfile.TemporaryDirectory() as tmp:
        z = pdgs_zip(tmp)
        try:
            ET.parse(z)
        except ET.ParseError as e:
            assert "not well-formed (invalid token): line 1, column 2" in str(e), e
        else:
            raise AssertionError("a zip parsed as XML")
        data, member = read_aux_xml(z)
        assert member == f"{ID}/{ID}.xml" and data == AUX_XML.read_bytes()


if __name__ == "__main__":
    tests = [(n, f) for n, f in sorted(globals().items()) if n.startswith("test_") and callable(f)]
    for n, f in tests:
        f()
        print(f"ok  {n}")
    print(f"all {len(tests)} tests passed")
