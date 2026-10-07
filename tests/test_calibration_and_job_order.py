# test_calibration_and_job_order.py: calibration from the product XML, and the Job Order's typed inputs.
import json
import re

import pytest

from conftest import ROOT
from iride_eaglet_processor.calibration import (
    CalibrationError, find_product_xml, load_calibration, read_xml_calibration,
)
from job_order import parse_job_order

PRODUCT_ID = "IME03_1CSP__RGB3_20260929T085012_20260929T085014_20260929T130000_04100_R020_O_A00"
XML_GAINS = {1: (1.62019221805984e-05, -0.0006557707036607915),
             2: (1.3992538505613578e-05, 0.001990793157538951),
             3: (1.8717640881898817e-05, 0.0023442647290774555)}


def write_product_xml(path, namespace=True, drop_band=None):
    ns = ' xmlns="urn:test"' if namespace else ""
    blocks = "".join(
        f'<Rescaling_Factors Band_ID="{b}"><GAIN>{g!r}</GAIN><OFFSET>{o!r}</OFFSET></Rescaling_Factors>'
        for b, (g, o) in XML_GAINS.items() if b != drop_band)
    path.write_text(f'<?xml version="1.0"?><Level-1C_User_Product{ns}><General_Info/>'
                    f'<L1C_Product_Image_Characteristics>{blocks}</L1C_Product_Image_Characteristics>'
                    f'</Level-1C_User_Product>')
    return path


def write_stac(root, with_metadata_href=True, scale_factor=1.03):
    stac = {"id": root.name, "properties": {}, "assets": {
        "rgb": {"href": f"./IMG_DATA/{root.name}.tif",
                "bands": [{"eo:common_name": n, "raster:scale": round(XML_GAINS[b][0] * scale_factor, 8),
                           "raster:offset": XML_GAINS[b][1]} for b, n in ((1, "red"), (2, "green"), (3, "blue"))]}}}
    if with_metadata_href:
        stac["assets"]["metadata"] = {"href": f"./{root.name}.xml"}
    path = root / f"{root.name}.json"
    path.write_text(json.dumps(stac))
    return path


def test_xml_calibration_is_exact_and_ignores_namespaces(tmp_path):
    for namespace in (True, False):
        cal = read_xml_calibration(write_product_xml(tmp_path / f"p{namespace}.xml", namespace=namespace))
        assert cal == XML_GAINS


def test_missing_band_is_an_error(tmp_path):
    with pytest.raises(CalibrationError, match="Band_ID 2"):
        read_xml_calibration(write_product_xml(tmp_path / "p.xml", drop_band=2))


def test_xml_preferred_over_stac_and_difference_logged(tmp_path, logger):
    root = tmp_path / PRODUCT_ID
    root.mkdir()
    write_product_xml(root / f"{PRODUCT_ID}.xml")
    json_path = write_stac(root)
    assert find_product_xml(root, json.loads(json_path.read_text())).name == f"{PRODUCT_ID}.xml"
    cal, source = load_calibration(root, json_path, logger)
    assert cal == XML_GAINS and "product XML" in source
    assert any("STAC raster:scale differs by +3.00%" in m for m in logger.text("info"))


def test_stac_fallback_with_a_warning(tmp_path, logger):
    root = tmp_path / PRODUCT_ID
    root.mkdir()
    json_path = write_stac(root, with_metadata_href=False, scale_factor=1.0)
    cal, source = load_calibration(root, json_path, logger)
    assert "STAC" in source and logger.text("warning")
    bands = json.loads(json_path.read_text())["assets"]["rgb"]["bands"]
    assert cal == {b + 1: (bands[b]["raster:scale"], bands[b]["raster:offset"]) for b in range(3)}


def write_job_order(path, files, task_version="02.00"):
    inputs = "".join(
        f"<Input><File_Type>{t}</File_Type><File_Name_Type>Physical</File_Name_Type>"
        f"<List_of_File_Names count=\"1\"><File_Name>{f}</File_Name></List_of_File_Names></Input>"
        for t, f in files)
    path.write_text(f"""<Ipf_Job_Order><Ipf_Conf><Logging_Level>DEBUG</Logging_Level></Ipf_Conf>
      <List_of_Ipf_Procs count="1"><Ipf_Proc><Task_Name>EAGLET2_L1D_CLM</Task_Name>
      <Task_Version>{task_version}</Task_Version><List_of_Inputs count="{len(files)}">{inputs}</List_of_Inputs>
      </Ipf_Proc></List_of_Ipf_Procs></Ipf_Job_Order>""")
    return path


def test_job_order_groups_inputs_by_type(tmp_path):
    jo = write_job_order(tmp_path / "JobOrder.xml", [("1CSP__RGB3", tmp_path / "p.zip"),
                                                     ("AXE1D_CLUT", tmp_path / "aux.xml")])
    parsed = parse_job_order(jo, "EAGLET2_L1D_CLM", "02.00")
    assert parsed["inputs_by_type"] == {"1CSP__RGB3": [tmp_path / "p.zip"], "AXE1D_CLUT": [tmp_path / "aux.xml"]}
    assert parsed["input_files"] == [tmp_path / "p.zip", tmp_path / "aux.xml"]
    assert parsed["logging_level"] == "DEBUG"


def test_job_order_for_another_task_version_is_refused(tmp_path):
    from job_order import JobOrderError
    jo = write_job_order(tmp_path / "JobOrder.xml", [("1CSP__RGB3", tmp_path / "p.zip")], task_version="01.00")
    with pytest.raises(JobOrderError):
        parse_job_order(jo, "EAGLET2_L1D_CLM", "02.00")


def test_make_job_order_writes_what_the_processor_reads(tmp_path):
    from make_job_order import main as make_job_order
    # the task the processor answers to, read from process_l1c.py's text (importing it needs torch)
    text = (ROOT / "process_l1c.py").read_text()
    TASK_NAME, TASK_VERSION = (re.search(rf'^{k} = "(.*?)"', text, re.M)[1] for k in ("TASK_NAME", "TASK_VERSION"))
    tmp = tmp_path.resolve()
    product = tmp / "OG_data" / f"{PRODUCT_ID}.zip"
    product.parent.mkdir()
    product.write_bytes(b"zip")
    aux = tmp / "IME___AXE1D_CLUT_20240101T000000_20990101T000000_20261005T000000____________O_001.xml"
    aux.write_text('<a><Data_Block><Colour_Correction version="4"><Method options="LUT CLASS_MODEL HYBRID">LUT'
                   '</Method><Table sha256="x">AAAA</Table></Colour_Correction></Data_Block></a>')

    make_job_order([str(product), "--work-dir", str(tmp / "run"), "--aux", str(aux), "--method", "HYBRID"])
    parsed = parse_job_order(tmp / "run" / "JobOrder.xml", TASK_NAME, TASK_VERSION)
    link, aux_copy = tmp / "run" / product.name, tmp / "run" / aux.name
    assert parsed["inputs_by_type"] == {"1CSP__RGB3": [link], "AXE1D_CLUT": [aux_copy]}
    assert link.is_symlink() and link.resolve() == product
    assert aux_copy.read_text() == aux.read_text().replace(">LUT</Method>", ">HYBRID</Method>")