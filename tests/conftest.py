# conftest.py: make the processor's modules and tools importable and share test fixtures.
import base64
import hashlib
import json
import sys
import xml.etree.ElementTree as ET
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "tools"))


class ListLogger:
    """stands in for PdgsLogger: keeps (level, message) pairs"""
    def __init__(self):
        self.messages = []

    def __getattr__(self, level):
        if level in ("debug", "info", "progress", "warning", "error"):
            return lambda message: self.messages.append((level, message))
        raise AttributeError(level)

    def text(self, level):
        return [m for lv, m in self.messages if lv == level]


@pytest.fixture
def logger():
    return ListLogger()


# ---- a small but complete colour AUX file -------------------------------------
X_START, Y_START, STEP, NX, NY, NV = -1.2, -1.0, 0.05, 49, 41, 5


def make_tables(seed=0):
    """smooth q tables (3, NX, NY, NV) around 1 and a vegetation-likelihood table (NX, NY)"""
    rs = np.random.default_rng(seed)
    x = X_START + STEP * np.arange(NX)
    y = Y_START + STEP * np.arange(NY)
    v = np.linspace(0, 1, NV)
    X, Y, Vg = np.meshgrid(x, y, v, indexing="ij")
    q = np.stack([0.6 + 0.3 * np.tanh(X) - 0.2 * Vg, 0.9 + 0.05 * Y + 0.0 * Vg, 0.8 - 0.1 * np.tanh(Y) + 0.05 * Vg])
    q += rs.normal(0, 0.01, q.shape)
    veg = 1 / (1 + np.exp(-(X[..., 0] * 6 - 1)))          # likely vegetation when band 1 / band 2 is high
    return q.astype("float32"), veg.astype("float32")


def make_model(seed=0):
    """a small model in the exact format of Enrico's MODEL (3 classes, 2 Gaussians each)"""
    rs = np.random.default_rng(seed)
    comps = []
    for c in range(3):
        means, precs, lws = [], [], []
        for j in range(2):
            A = rs.normal(0, 1, (3, 3))
            cov = A @ A.T * 0.05 + np.eye(3) * 0.02
            P = np.linalg.inv(cov)
            means.append((rs.normal(0, 0.5, 3) + (c - 1)).tolist())
            precs.append(P.tolist())
            lws.append(float(np.log(0.5) + 0.5 * np.linalg.slogdet(P)[1]))
        comps.append({"mean": means, "precision": precs, "log_weight": lws})
    xq = np.linspace(0.05, 0.4, 15)
    experts = {
        "1": {"kind": "water_joint_curve", "xq": xq.tolist(),
              "yq": [(0.02 + 0.1 * xq).tolist(), (0.04 + 0.1 * xq).tolist(), (0.06 + 0.1 * xq).tolist()],
              "lo": [0.01, 0.01, 0.02], "hi": [0.4, 0.4, 0.5], "tails": "clamped"},
        "2": {"mean": [0.2, 0.15, 0.2], "scale": [0.05, 0.04, 0.05],
              "coef": rs.normal(0, 0.02, (4, 3)).tolist(), "alpha": 0.001,
              "lo": [0.05, 0.05, 0.1], "hi": [0.5, 0.4, 0.6]},
        "3": {"mean": [0.3, 0.3, 0.35], "scale": [0.1, 0.1, 0.1],
              "coef": rs.normal(0, 0.03, (4, 3)).tolist(), "alpha": 0.001,
              "lo": [0.08, 0.1, 0.12], "hi": [0.8, 0.7, 0.9]},
    }
    return {"center": [0.25, 0.22, 0.3], "scale": [0.12, 0.1, 0.11], "components": comps, "experts": experts}


def write_aux(path, method="LUT", with_model=True, balance=(0.92, 1.0, 1.29), corrupt=None):
    """writes a colour AUX file like the notebook does (version 4, or 3 without the model and Method)"""
    q, veg = make_tables()
    root = ET.Element("Earth_Explorer_File")
    ET.SubElement(ET.SubElement(root, "Earth_Explorer_Header"), "Fixed_Header")
    cc = ET.SubElement(ET.SubElement(root, "Data_Block", type="xml"), "Colour_Correction",
                       version="4" if with_model else "3")
    if with_model and method is not None:
        ET.SubElement(cc, "Method").text = method
    ET.SubElement(cc, "Colour_Balance", band1=str(balance[0]), band2=str(balance[1]), band3=str(balance[2]))
    axes = ET.SubElement(cc, "Axes")
    ET.SubElement(axes, "Axis", name="x", start=str(X_START), step=str(STEP), count=str(NX))
    ET.SubElement(axes, "Axis", name="y", start=str(Y_START), step=str(STEP), count=str(NY))
    ET.SubElement(axes, "Axis", name="v", start="0", step="0.25", count=str(NV))
    ET.SubElement(cc, "Vegetation_Around", cell_m="10", window_cells="7", table="veg_share")
    for name, arr in [("veg_share", veg)] + [(f"q_band{b + 1}", q[b]) for b in range(3)]:
        raw = np.ascontiguousarray(arr, dtype="<f4").tobytes()
        sha = hashlib.sha256(raw).hexdigest()
        if corrupt == name:
            sha = "0" * 64
        ET.SubElement(cc, "Table", name=name, shape=" ".join(map(str, arr.shape)), dtype="float32",
                      byte_order="little", encoding="base64", sha256=sha).text = base64.b64encode(raw).decode()
    if with_model:
        text = json.dumps(make_model(), separators=(",", ":"))
        sha = hashlib.sha256(text.encode()).hexdigest() if corrupt != "Class_Model" else "0" * 64
        ET.SubElement(cc, "Class_Model", encoding="json", sha256=sha).text = text
    ET.ElementTree(root).write(path, encoding="UTF-8", xml_declaration=True)
    return path


@pytest.fixture
def aux_v4(tmp_path):
    return write_aux(tmp_path / "IME___AXE1D_CLUT_20240101T000000_20990101T000000_20261005T000000____________O_001.xml")