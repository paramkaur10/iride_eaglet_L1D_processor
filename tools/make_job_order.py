#!/usr/bin/env python3
"""
make_job_order.py: prepares a local test run of the processor, so no XML has
to be edited by hand. (In operations PDGS writes the Job Order.)

    python tools/make_job_order.py <product.zip> --work-dir <folder> \
        --aux <IME___AXE1D_CLUT_...xml> [--method LUT|CLASS_MODEL|HYBRID] [--copy]

  - creates <folder> and puts the product zip in it: a symbolic link (or a
    copy with --copy); the original zip is never touched
  - --method: copies the AUX file into <folder> with <Method> set to that
    value (the only change; the tables and the model keep their checksums)
  - writes <folder>/JobOrder.xml from sample_job_order.xml with these paths
  - prints the command that runs the processor
"""

import argparse
import re
import shutil
import xml.etree.ElementTree as ET
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SAMPLE = ROOT / "sample_job_order.xml"
PRODUCT_TYPE, CLUT_TYPE = "1CSP__RGB3", "AXE1D_CLUT"
METHODS = ("LUT", "CLASS_MODEL", "HYBRID")
METHOD_ELEMENT = re.compile(r"(<Method\b[^>]*>)\s*\w*\s*(</Method>)")


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("product", type=Path, help="L1C RGB3 product zip (IMEnn_1CSP__RGB3_....zip)")
    ap.add_argument("--work-dir", type=Path, required=True, help="folder for this run (created if missing)")
    ap.add_argument("--aux", type=Path, required=True, help="colour AUX file (IME___AXE1D_CLUT_...xml)")
    ap.add_argument("--method", choices=METHODS, help="use this method instead of the AUX file's")
    ap.add_argument("--copy", action="store_true", help="copy the product zip instead of linking it")
    ap.add_argument("--log-level", default="INFO", choices=("DEBUG", "INFO", "PROGRESS", "WARNING", "ERROR"))
    a = ap.parse_args(argv)

    product = a.product.resolve()
    if not (product.is_file() and product.suffix == ".zip"):
        raise SystemExit(f"not a product zip: {product}")
    aux = a.aux.resolve()
    if not aux.is_file():
        raise SystemExit(f"AUX file not found: {aux}")

    work = a.work_dir.resolve()
    work.mkdir(parents=True, exist_ok=True)

    # the product zip has to be in the working folder: the output is written next to it
    target = work / product.name
    if target.is_symlink() and not target.exists():
        target.unlink()                                          # broken link from an earlier run
    if not target.exists():
        if a.copy:
            shutil.copy2(product, target)
        else:
            target.symlink_to(product)
    elif not (target.samefile(product) or target.stat().st_size == product.stat().st_size):
        raise SystemExit(f"{target} exists and is a different file: remove it or use another --work-dir")

    if a.method:
        if aux.parent == work:
            raise SystemExit("with --method, use a --work-dir other than the AUX file's folder (it gets a copy)")
        text, n = METHOD_ELEMENT.subn(rf"\g<1>{a.method}\g<2>", aux.read_text(encoding="utf-8"), count=1)
        if n != 1:
            raise SystemExit(f"{aux.name} has no <Method> (a version 3 file is LUT only)")
        aux = work / aux.name
        aux.write_text(text, encoding="utf-8")

    tree = ET.parse(SAMPLE)
    root = tree.getroot()
    root.find("Ipf_Conf/Logging_Level").text = a.log_level
    times = re.search(r"_(\d{8})T(\d{6})_(\d{8})T(\d{6})_", product.name)
    if times:                                                    # sensing start / stop from the product name
        root.find("Ipf_Conf/Sensing_Time/Start").text = f"{times[1]}_{times[2]}000000"
        root.find("Ipf_Conf/Sensing_Time/Stop").text = f"{times[3]}_{times[4]}000000"
    inputs = root.find("List_of_Ipf_Procs/Ipf_Proc/List_of_Inputs")
    for inp in inputs.findall("Input"):
        file_type = inp.findtext("File_Type")
        if file_type == PRODUCT_TYPE:
            inp.find("List_of_File_Names/File_Name").text = str(target)
        elif file_type == CLUT_TYPE:
            inp.find("List_of_File_Names/File_Name").text = str(aux)
        else:
            inputs.remove(inp)
    inputs.set("count", str(len(inputs.findall("Input"))))
    ET.indent(tree)
    job_order = work / "JobOrder.xml"
    tree.write(job_order, encoding="UTF-8", xml_declaration=True)

    print(f"written {job_order}")
    print(f"  product:    {target}{' (link)' if target.is_symlink() else ''}")
    print(f"  colour AUX: {aux}{f' (Method set to {a.method})' if a.method else ''}")
    print(f"run:\n  python {ROOT / 'process_l1c.py'} {job_order}")


if __name__ == "__main__":
    main()