# product_xml.py
"""
The header of the L1D product XML, as in Enrico's L1D template (12 Oct 2026).

The L1D product keeps the L1C XML, element names included (Level-1C_Main_Product_Header,
L1C_Product_Image_Characteristics, ...). The template marks what the L1D processor updates:
    PRODUCT_ID           the new L1D product ID
    PROCESSING_LEVEL     L1D
    GENERATION_TIME      the L1D generation time, the same as in the product name and the STAC
                         (Enrico, 13 Oct 2026)
    AUXILIARY_FILE       only the colour AUX file used (its name without .xml or .zip), in place of the L1C list
    INPUT_PRODUCTS       one PRODUCT, elem="1": the input L1C product zip
    Processor_Version    this processor's version
Rescaling_Factors and the image checksum are set by calibration.set_l1d_image_in_xml. Everything else
stays as in the L1C, as the template asks (outliers, RPC, ...).

The edits are made on the text, so the rest of the file stays byte for byte as it was, and every value
is read back from the written file. An XML without exactly the expected L1C elements stops the
processor rather than being guessed around.
"""

import re
import xml.etree.ElementTree as ET
from datetime import timedelta
from pathlib import Path

from iride_eaglet_processor.colour_correction import aux_id

AUX_FILE = re.compile(r"(?P<lead>\r?\n[ \t]*)?<AUXILIARY_FILE>[^<]*</AUXILIARY_FILE>")
INPUT_PRODUCT = re.compile(r"(?P<lead>\r?\n[ \t]*)?<PRODUCT\b[^>]*>[^<]*</PRODUCT>")
INPUT_PRODUCTS = re.compile(r"(<INPUT_PRODUCTS>)(.*?)(</INPUT_PRODUCTS>)", re.S)
PROCESSING_INFO = re.compile(r"<Processing_Info>.*?</Processing_Info>", re.S)


def _local(tag):
    return tag.split("}")[-1] if isinstance(tag, str) else ""


def _set_value(text, element, new, expected=None):
    """text with the value of its one <element> set to new; an error when there is not exactly one, or
    when its value is not the expected one"""
    pattern = re.compile(rf"(<{element}>)([^<]*)(</{element}>)")
    found = pattern.findall(text)
    if len(found) != 1:
        raise ValueError(f"product XML: expected one <{element}>, found {len(found)}")
    if expected is not None and found[0][1].strip() != expected:
        raise ValueError(f"product XML: <{element}> is {found[0][1].strip()!r}, expected {expected!r}")
    return pattern.sub(lambda m: m.group(1) + new + m.group(3), text, count=1)


def _keep_one(text, pattern, new_element, what):
    """text with the first match of pattern replaced by new_element and the other matches removed, each
    with the line break and indentation before it. Returns (text, number of matches)."""
    matches = list(pattern.finditer(text))
    if not matches:
        raise ValueError(f"product XML: no {what}")
    parts, last = [], 0
    for i, m in enumerate(matches):
        parts.append(text[last:m.start()])
        if i == 0:
            parts.append((m.group("lead") or "") + new_element)
        last = m.end()
    parts.append(text[last:])
    return "".join(parts), len(matches)


def set_l1d_header_in_xml(xml_path, input_product_id, new_product_id, generation_time, input_zip_name, aux_path,
                          processor_version, logger):
    """
    Writes the L1D header into the product XML (still named after the L1C at this point), as in Enrico's
    template. input_product_id must be the PRODUCT_ID of the XML (the L1C), whose PROCESSING_LEVEL must
    be L1C. generation_time (UTC) is the one in the new product name; it is written as in the STAC
    (created / updated), e.g. 2026-10-13T09:12:45.123456Z. aux_path is the colour AUX file used (the XML, or
    the zip PDGS gives it in); it is listed by its name without .xml or .zip, as the L1C lists its AUX files.
    """
    if generation_time.utcoffset() != timedelta(0):
        raise ValueError(f"generation time must be in UTC, got {generation_time!r}")
    gen_time = generation_time.strftime("%Y-%m-%dT%H:%M:%S.%f") + "Z"
    xml_path, aux_path = Path(xml_path), Path(aux_path)
    aux_name = aux_id(aux_path)
    text = xml_path.read_bytes().decode("utf-8")

    text = _set_value(text, "PRODUCT_ID", new_product_id, expected=input_product_id)
    text = _set_value(text, "PROCESSING_LEVEL", "L1D", expected="L1C")

    infos = PROCESSING_INFO.findall(text)
    if len(infos) != 1:
        raise ValueError(f"product XML: expected one <Processing_Info>, found {len(infos)}")
    info = infos[0]
    outside = text.replace(info, "")
    for element in ("GENERATION_TIME", "AUXILIARY_FILE", "INPUT_PRODUCTS", "Processor_Version"):
        if f"<{element}" in outside:
            raise ValueError(f"product XML: <{element}> outside <Processing_Info>")

    new_info = _set_value(info, "GENERATION_TIME", gen_time)
    new_info, n_aux = _keep_one(new_info, AUX_FILE, f"<AUXILIARY_FILE>{aux_name}</AUXILIARY_FILE>",
                                "<AUXILIARY_FILE> in <Processing_Info>")
    lists = INPUT_PRODUCTS.findall(new_info)
    if len(lists) != 1:
        raise ValueError(f"product XML: expected one <INPUT_PRODUCTS>, found {len(lists)}")
    body, n_inputs = _keep_one(lists[0][1], INPUT_PRODUCT, f'<PRODUCT elem="1">{input_zip_name}</PRODUCT>',
                               "<PRODUCT> in <INPUT_PRODUCTS>")
    if re.sub(r"<PRODUCT\b[^>]*>[^<]*</PRODUCT>", "", body).strip():
        raise ValueError("product XML: <INPUT_PRODUCTS> holds something other than <PRODUCT> elements")
    new_info = INPUT_PRODUCTS.sub(lambda m: m.group(1) + body + m.group(3), new_info, count=1)
    new_info = _set_value(new_info, "Processor_Version", processor_version)
    text = text.replace(info, new_info, 1)

    data = text.encode("utf-8")
    root = ET.fromstring(data)                      # read back what is about to be written

    def values(name):
        return [(e.text or "").strip() for e in root.iter() if _local(e.tag) == name]

    inputs = [(p.get("elem"), (p.text or "").strip())
              for e in root.iter() if _local(e.tag) == "INPUT_PRODUCTS" for p in e if _local(p.tag) == "PRODUCT"]
    read_back = {"PRODUCT_ID": values("PRODUCT_ID"), "PROCESSING_LEVEL": values("PROCESSING_LEVEL"),
                 "GENERATION_TIME": values("GENERATION_TIME"), "AUXILIARY_FILE": values("AUXILIARY_FILE"),
                 "INPUT_PRODUCTS": inputs, "Processor_Version": values("Processor_Version")}
    expected = {"PRODUCT_ID": [new_product_id], "PROCESSING_LEVEL": ["L1D"], "GENERATION_TIME": [gen_time],
                "AUXILIARY_FILE": [aux_name], "INPUT_PRODUCTS": [("1", input_zip_name)],
                "Processor_Version": [processor_version]}
    if read_back != expected:
        raise ValueError(f"product XML: L1D header read back as {read_back}, expected {expected}")

    tmp = xml_path.with_name(xml_path.name + ".part")
    tmp.write_bytes(data)
    tmp.replace(xml_path)
    logger.info(f"Product XML, L1D header: PRODUCT_ID {new_product_id}, PROCESSING_LEVEL L1D, GENERATION_TIME {gen_time}, "
                f"AUXILIARY_FILE {aux_name} (in place of {n_aux} L1C AUX file(s)), "
                f"INPUT_PRODUCTS {input_zip_name} (in place of {n_inputs}), Processor_Version {processor_version}")
