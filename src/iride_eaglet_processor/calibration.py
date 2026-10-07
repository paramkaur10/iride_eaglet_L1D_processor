# calibration.py
"""
Per-band radiometric calibration of Eaglet-2 L1C products:
reflectance = DN * GAIN + OFFSET.

GAIN and OFFSET are read from the product XML's Rescaling_Factors (Band_ID
1, 2, 3 = red, green, blue): the exact values. The STAC sidecar's
raster:scale / raster:offset are a rounded copy of the same numbers (up to a
few percent off), so they are only a fallback when the XML is missing, and
the two are compared in the logs when both are present.

Read fresh from each product, never cached: they change between acquisitions
(and are taken as correct for products from 24 September 2026 on).

In the L1D product the XML describes the L1D image instead: its
Rescaling_Factors and image checksum are rewritten by set_l1d_image_in_xml().
"""

import hashlib
import json
import re
import xml.etree.ElementTree as ET
from pathlib import Path

BAND_IDS = ("1", "2", "3")          # red, green, blue


class CalibrationError(Exception):
    pass


def _local(tag):
    return tag.rsplit("}", 1)[-1]


def find_product_xml(product_root, stac=None):
    """
    The product's own metadata XML: where the STAC "metadata" asset points,
    otherwise <product_id>.xml in the product root, otherwise the only XML
    there. Returns None if there is none.
    """
    product_root = Path(product_root)
    if stac is not None:
        href = stac.get("assets", {}).get("metadata", {}).get("href")
        if href and (product_root / href).is_file():
            return (product_root / href).resolve()
    same_name = product_root / f"{product_root.name}.xml"
    if same_name.is_file():
        return same_name
    xmls = sorted(p for p in product_root.glob("*.xml") if p.is_file())
    return xmls[0] if len(xmls) == 1 else None


def read_xml_calibration(xml_path):
    """{band_index (1, 2, 3): (gain, offset)} from Rescaling_Factors; namespaces ignored"""
    xml_path = Path(xml_path)
    try:
        root = ET.parse(xml_path).getroot()
    except (OSError, ET.ParseError) as e:
        raise CalibrationError(f"Cannot read product XML {xml_path}: {e}") from e
    found = {}
    for el in root.iter():
        if _local(el.tag) != "Rescaling_Factors":
            continue
        band = {_local(k): v for k, v in el.attrib.items()}.get("Band_ID", "").strip()
        if band not in BAND_IDS:
            continue
        vals = {}
        for child in el:
            key = _local(child.tag)
            if key in ("GAIN", "OFFSET"):
                try:
                    vals[key] = float((child.text or "").strip())
                except ValueError as e:
                    raise CalibrationError(f"{xml_path.name}: {key} for Band_ID {band} is not a number") from e
        if set(vals) != {"GAIN", "OFFSET"}:
            raise CalibrationError(f"{xml_path.name}: GAIN or OFFSET missing for Band_ID {band}")
        pair = (vals["GAIN"], vals["OFFSET"])
        if band in found and found[band] != pair:
            raise CalibrationError(f"{xml_path.name}: two different GAIN/OFFSET pairs for Band_ID {band}")
        found[band] = pair
    missing = [b for b in BAND_IDS if b not in found]
    if missing:
        raise CalibrationError(f"{xml_path.name}: no Rescaling_Factors for Band_ID {', '.join(missing)}")
    return {int(b): found[b] for b in BAND_IDS}


def read_stac_calibration(stac):
    """{band_index: (scale, offset)} from assets.rgb.bands raster:scale / raster:offset"""
    try:
        bands_meta = stac["assets"]["rgb"]["bands"]
        return {i + 1: (float(bands_meta[i]["raster:scale"]), float(bands_meta[i]["raster:offset"]))
                for i in range(3)}
    except (KeyError, IndexError, TypeError, ValueError) as e:
        raise CalibrationError(f"No usable assets.rgb.bands raster:scale/offset in the STAC sidecar ({e})") from e


def load_calibration(product_root, json_path, logger):
    """
    Returns ({band_index: (gain, offset)}, source string). Product XML first,
    STAC sidecar as a fallback; logs the values and how far the STAC's
    rounded scale is from the XML gain.
    """
    with open(json_path, "r") as f:
        stac = json.load(f)
    xml_path = find_product_xml(product_root, stac)
    stac_cal = None
    try:
        stac_cal = read_stac_calibration(stac)
    except CalibrationError:
        pass

    if xml_path is not None:
        cal = read_xml_calibration(xml_path)
        source = f"product XML {xml_path.name} (Rescaling_Factors)"
        for b, (gain, offset) in cal.items():
            note = ""
            if stac_cal is not None and gain != 0:
                note = f" | STAC raster:scale differs by {100 * (stac_cal[b][0] / gain - 1):+.2f}%"
            logger.info(f"Calibration band {b}: gain={gain!r}, offset={offset!r}{note}")
        return cal, source

    if stac_cal is None:
        raise CalibrationError(f"No product XML and no STAC calibration for {Path(product_root).name}")
    logger.warning("Product XML not found: calibration taken from the STAC sidecar's rounded raster:scale/offset")
    for b, (scale, offset) in stac_cal.items():
        logger.info(f"Calibration band {b} (STAC): scale={scale!r}, offset={offset!r}")
    return stac_cal, "STAC sidecar (raster:scale/offset)"


# ---- the L1D product's XML ----------------------------------------------------
_NS = r"(?:[\w.-]+:)?"                          # optional namespace prefix
_ATTRS = r"""(?:[^>"']|"[^"]*"|'[^']*')*"""      # attributes, quoted values may hold / or >


def _element(tag):
    """regex for <tag ...>body</tag>: groups 1 opening tag, 2 attributes, 3 body, 4 closing tag"""
    return re.compile(rf"(<{_NS}{tag}\b({_ATTRS})(?<!/)>)(.*?)(</{_NS}{tag}\s*>)", re.S)


_RESCALING, _CHECKSUM, _IMAGE = _element("Rescaling_Factors"), _element("Checksum"), _element("Image")


def _attribute(attrs, name):
    m = re.search(rf"""\b{name}\s*=\s*(?:"([^"]*)"|'([^']*)')""", attrs)
    return None if m is None else (m.group(1) if m.group(1) is not None else m.group(2)).strip()


def _set_leaf(block, tag, value, what):
    new, n = _element(tag).subn(lambda m: f"{m.group(1)}{value}{m.group(4)}", block)
    if n != 1:
        raise CalibrationError(f"{what}: expected one <{tag}>, found {n}")
    return new


def _sha256_like(old, path):
    """SHA-256 of the file, written like the old value (plain hex or multihash "1220...", same case)"""
    digest = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            digest.update(chunk)
    new = digest.hexdigest()
    old = old.strip()
    if len(old) == 68 and old.startswith("1220"):
        new = "1220" + new
    return new.upper() if old.isupper() else new


def set_l1d_image_in_xml(xml_path, image_path, gain, offset, logger):
    """
    Makes the product XML describe the L1D image now in IMG_DATA: GAIN and
    OFFSET of every Rescaling_Factors (Band_ID 1, 2, 3) and the image's
    SHA-256 (Checksum algorithm="SHA256" / Image). Text edits only, so the
    rest of the file stays byte for byte as it was; checked by reading the
    file back.
    """
    xml_path = Path(xml_path)
    raw = xml_path.read_bytes()
    declared = re.match(rb"""<\?xml[^>]*encoding\s*=\s*["']([\w.-]+)["']""", raw)
    codec = declared.group(1).decode("ascii") if declared else "utf-8"
    text = raw.decode(codec)

    bands = []

    def rescaling(m):
        band = _attribute(m.group(2), "Band_ID")
        if band not in BAND_IDS:
            return m.group(0)
        what = f"{xml_path.name} Rescaling_Factors Band_ID {band}"
        body = _set_leaf(_set_leaf(m.group(3), "GAIN", repr(gain), what), "OFFSET", repr(offset), what)
        bands.append(band)
        return m.group(1) + body + m.group(4)

    text = _RESCALING.sub(rescaling, text)
    if sorted(set(bands)) != list(BAND_IDS):
        raise CalibrationError(f"{xml_path.name}: Rescaling_Factors found for Band_ID {sorted(set(bands))} only")

    checksum = None

    def image_checksum(m):
        nonlocal checksum
        algorithm = (_attribute(m.group(2), "algorithm") or "").upper().replace("-", "")
        image = _IMAGE.search(m.group(3))
        if checksum is not None or algorithm != "SHA256" or image is None:
            return m.group(0)
        checksum = _sha256_like(image.group(3), image_path)
        return m.group(1) + _set_leaf(m.group(3), "Image", checksum, f"{xml_path.name} Checksum") + m.group(4)

    text = _CHECKSUM.sub(image_checksum, text)

    tmp = xml_path.with_name(xml_path.name + ".part")
    tmp.write_bytes(text.encode(codec))
    try:
        ET.parse(tmp)                                            # still well-formed
        if read_xml_calibration(tmp) != {int(b): (gain, offset) for b in BAND_IDS}:
            raise CalibrationError(f"{xml_path.name}: Rescaling_Factors did not read back as written")
        tmp.replace(xml_path)
    finally:
        if tmp.exists():
            tmp.unlink()

    logger.info(f"Product XML {xml_path.name}: Rescaling_Factors set to GAIN {gain!r}, OFFSET {offset!r} "
                f"(bands {', '.join(BAND_IDS)}) for the L1D image")
    if checksum is None:
        logger.warning(f"Product XML {xml_path.name}: no Checksum algorithm=SHA256 / Image found, not updated")
    else:
        logger.info(f"Product XML {xml_path.name}: image SHA-256 set to {checksum}")