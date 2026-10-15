# job_order.py
"""
Parses the Job Order XML file passed to this processor as its single
command-line argument, per PDGS-ID-XPR-IRIDE-0209 Section 3.1.2.

The Job Order tells us, per the ICD:
  - the logging level to use (Ipf_Conf/Logging_Level)
  - the input file(s) to process (List_of_Ipf_Procs/Ipf_Proc/List_of_Inputs)
  - where output is expected (List_of_Outputs)

We deliberately do NOT scan the working directory for input files: the
ICD is explicit that a processor "must rely on the paths specified in the
JobOrder, trusting that they exist as volumes mounted at container
runtime," not make assumptions about paths.
"""

import xml.etree.ElementTree as ET
from pathlib import Path


class JobOrderError(Exception):
    pass


def parse_job_order(job_order_path, task_name, task_version, logger=None):
    """
    Returns a dict:
      logging_level    str, one of DEBUG/INFO/PROGRESS/WARNING/ERROR
      working_dir       Path, the directory containing the Job Order file
      input_files       list[Path], absolute paths of every input file
                        found for this task, in Job Order order
      inputs_by_type    dict File_Type -> list[Path], the same files grouped
                        by their <Input>'s File_Type (e.g. "1CSP__RGB3" for
                        the product, "AXE1D_CLUT" for the colour LUT);
                        files of an <Input> without File_Type are under ""
      dyn_params        dict Name -> Value of Ipf_Conf/List_of_Dyn_Processing_Parameters
                        (e.g. "Environment" -> "O"); an empty value is ""
      output_spec       dict describing the single declared Output
                        (destination, file_type, file_name, reg_exp), for
                        reference / future use
    """
    job_order_path = Path(job_order_path).resolve()
    if not job_order_path.exists():
        raise JobOrderError(f"Job Order file not found: {job_order_path}")

    working_dir = job_order_path.parent

    try:
        tree = ET.parse(job_order_path)
    except ET.ParseError as e:
        raise JobOrderError(f"Failed to parse Job Order XML: {job_order_path} ({e})") from e

    root = tree.getroot()

    ipf_conf = root.find("Ipf_Conf")
    if ipf_conf is None:
        raise JobOrderError(f"Missing <Ipf_Conf> in Job Order: {job_order_path}")

    logging_level_el = ipf_conf.find("Logging_Level")
    logging_level = logging_level_el.text.strip() if logging_level_el is not None and logging_level_el.text else "INFO"

    dyn_params = {}
    for param_el in ipf_conf.findall("./List_of_Dyn_Processing_Parameters/Dyn_Processing_Parameter"):
        name = _text(param_el, "Name")
        if name:
            dyn_params[name] = _text(param_el, "Value") or ""

    # Find our task's section within the Job Order. A Job Order can contain
    # directives for multiple tasks; we must identify our own by
    # {Task_Name, Task_Version}, per the ICD.
    matching_proc = None
    for ipf_proc in root.findall("./List_of_Ipf_Procs/Ipf_Proc"):
        name_el = ipf_proc.find("Task_Name")
        version_el = ipf_proc.find("Task_Version")
        name = name_el.text.strip() if name_el is not None and name_el.text else None
        version = version_el.text.strip() if version_el is not None and version_el.text else None
        if name == task_name and version == task_version:
            matching_proc = ipf_proc
            break

    if matching_proc is None:
        raise JobOrderError(
            f"No matching Ipf_Proc for Task_Name='{task_name}' "
            f"Task_Version='{task_version}' in Job Order: {job_order_path}"
        )

    input_files = []
    inputs_by_type = {}
    for input_el in matching_proc.findall("./List_of_Inputs/Input"):
        file_type = _text(input_el, "File_Type") or ""
        for file_name_el in input_el.findall("./List_of_File_Names/File_Name"):
            if file_name_el.text and file_name_el.text.strip():
                path = Path(file_name_el.text.strip())
                input_files.append(path)
                inputs_by_type.setdefault(file_type, []).append(path)

    if len(input_files) == 0:
        raise JobOrderError(
            f"No input files found in Job Order for task "
            f"'{task_name}' {task_version}: {job_order_path}"
        )

    output_spec = None
    output_el = matching_proc.find("./List_of_Outputs/Output")
    if output_el is not None:
        output_spec = {
            "destination": _text(output_el, "Destination"),
            "mandatory": _text(output_el, "Mandatory"),
            "file_type": _text(output_el, "File_Type"),
            "file_name": _text(output_el, "File_Name"),
            "reg_exp": _text(output_el, "Reg_Exp"),
        }

    result = {
        "logging_level": logging_level,
        "working_dir": working_dir,
        "input_files": input_files,
        "inputs_by_type": inputs_by_type,
        "dyn_params": dyn_params,
        "output_spec": output_spec,
    }

    if logger:
        logger.info(f"Job Order parsed: {job_order_path}")
        logger.info(f"Logging level: {logging_level}")
        logger.info(f"Working directory: {working_dir}")
        for name, value in dyn_params.items():
            logger.info(f"Dynamic parameter {name}: '{value}'")
        for file_type, paths in inputs_by_type.items():
            logger.info(f"Input file(s) of type {file_type or '(none)'}: {[str(p) for p in paths]}")

    return result


def _text(element, tag):
    el = element.find(tag)
    return el.text.strip() if el is not None and el.text else None