# inference.py
"""
The cloud model, Clarity for IRIDE Eaglet II (float32), on the whole scene.

Clarity is delivered as a package, a zip (clarity_eaglet2_fp32_<date>.zip) unzipped unchanged into MODEL_DIR,
models/clarity_eaglet2_fp32 in the processor folder:
    clarity_eaglet2.pt        the network for the 5 m grid (TorchScript, float32)
    clarity_eaglet2_15m.pt    the same network told a 15 m grid
    sensor.json               its bands (red, green, blue), its pixel (5 m) and its file names
    run_clarity.py            how a scene is read, as the model was tested
    MODEL.md, README.md, requirements.txt
    SHA256SUMS                the SHA-256 of every file above
Every file is checked against SHA256SUMS before the model is used (ModelError otherwise), and the scene is read
with run_clarity.py's own functions, imported from the package, so that the processor reads a scene exactly as the
package does: tiles of 240 x 240 pixels every 120, blended, on the 5 m grid, then again on a 15 m grid (the scene
averaged by area), the two readings averaged. The package travels with the processor (in the Docker image):
nothing is downloaded, at run time or before.

Clarity's four classes are, in this order, clear, thin cloud, thick cloud and cloud shadow. The rest of the
processor keeps its own order, clear, thick cloud, thin cloud, cloud shadow (masks.py, stats.py and the CLM
values), so the probabilities are put in that order here (TO_PROCESSOR_ORDER).
"""

import hashlib
import importlib.util
import json
import re
import warnings
from pathlib import Path

import numpy as np
import torch

MODEL_DIR = Path(__file__).resolve().parents[2] / "models" / "clarity_eaglet2_fp32"
MODEL_NAME = "Clarity"
BANDS = ["red", "green", "blue"]       # the bands of the L1C image, in its order, as the model reads them
TO_PROCESSOR_ORDER = [0, 2, 1, 3]      # Clarity's clear, thin, thick, shadow -> clear, thick, thin, shadow
BATCH = 16                             # tiles through the network at a time (run_clarity.py's default)


class ModelError(Exception):
    """the cloud model package is missing, incomplete or not the delivered one"""


def sha256_of(path):
    digest = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def check_package(model_dir=MODEL_DIR):
    """
    {file name: SHA-256} of the model package in model_dir, every file checked against the package's own
    SHA256SUMS. ModelError when SHA256SUMS is missing, or a file it lists is missing or differs from it.
    """
    model_dir = Path(model_dir)
    sums = model_dir / "SHA256SUMS"
    if not sums.is_file():
        raise ModelError(f"no cloud model package in {model_dir} (no SHA256SUMS there): unzip the delivered "
                         f"package into models/ (models/README.md)")
    listed = {}
    for line in sums.read_text().splitlines():
        if not line.strip():
            continue
        digest, name = line.split(maxsplit=1)
        name = name.lstrip("*")
        if Path(name).name != name:
            raise ModelError(f"cloud model package in {model_dir}: SHA256SUMS lists {name}, not a file of the "
                             f"package folder")
        listed[name] = digest.lower()
    for name, digest in listed.items():
        path = model_dir / name
        if not path.is_file():
            raise ModelError(f"cloud model package in {model_dir}: {name} is missing")
        if sha256_of(path) != digest:
            raise ModelError(f"cloud model package in {model_dir}: {name} is not the delivered file (its SHA-256 "
                             f"is not the one in SHA256SUMS)")
    return listed


def limit_threads(n):
    """PyTorch uses at most n CPU threads. Returns (intra-op, inter-op) thread counts."""
    torch.set_num_threads(n)
    try:
        torch.set_num_interop_threads(n)
    except RuntimeError:            # can be set only once, before any inter-op work
        pass
    return torch.get_num_threads(), torch.get_num_interop_threads()


class CloudModel:
    """
    Clarity, from its package in model_dir (checked against its SHA256SUMS), on the CPU: probabilities() reads a
    scene. ModelError when the package is missing, incomplete or not the delivered one.
    """

    def __init__(self, model_dir=MODEL_DIR, device="cpu", batch=BATCH):
        self.dir = Path(model_dir)
        self.checked = check_package(self.dir)
        try:
            sensor = json.loads((self.dir / "sensor.json").read_text())
            self.files = (sensor["model"], sensor["model_15m"])
            self.pixel_m = float(sensor["pixel_m"])
            bands = list(sensor["bands"])
        except (OSError, ValueError, KeyError, TypeError) as e:
            raise ModelError(f"cloud model package in {self.dir}: sensor.json is not usable ({e})") from None
        if bands != BANDS:
            raise ModelError(f"the cloud model reads the bands {bands}; the L1C image has {BANDS}")
        for name in ("run_clarity.py", "sensor.json", *self.files):
            if name not in self.checked:
                raise ModelError(f"cloud model package in {self.dir}: {name} is not in SHA256SUMS")
        spec = importlib.util.spec_from_file_location("run_clarity", self.dir / "run_clarity.py")
        reader = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(reader)
        self.reader = reader
        with warnings.catch_warnings():
            # PyTorch 2.14 warns that torch.jit.load is deprecated (Python 3.13) or not supported on
            # Python 3.14+ and may break (the image's Python 3.14), in favour of torch.export. With the
            # image's pinned versions these files load and run as they should (checked when the image is
            # built, and by docker/test_image.sh); the move to torch.export is for a later package (README,
            # open items). Left out of the log, where it would be a raw line on stderr, not in the PDGS
            # format, and nothing an operator can act on.
            warnings.filterwarnings("ignore", message=r".*torch\.jit\.load.* is (deprecated|not supported)",
                                    category=FutureWarning)
            self.own = reader.network(self.dir / self.files[0], device)      # the 5 m grid
            self.rough = reader.network(self.dir / self.files[1], device)    # the 15 m grid
        self.coarse_m = float(reader.COARSE)
        self.factor = self.coarse_m / self.pixel_m
        self.batch = batch
        notes = self.dir / "MODEL.md"
        packed = re.search(r"packed (\d{4}-\d{2}-\d{2})", notes.read_text()) if notes.is_file() else None
        self.packed = packed.group(1) if packed else "date unknown"

    def describe(self):
        """one line for the log: the package, its networks (and the start of their SHA-256) and how it reads"""
        networks = ", ".join(f"{name} (SHA-256 {self.checked[name][:12]})" for name in self.files)
        return (f"{MODEL_NAME} ({self.dir.name}, packed {self.packed}): {networks}; {len(self.checked)} files "
                f"checked against its SHA256SUMS; tiles of {self.reader.TILE} x {self.reader.TILE} pixels every "
                f"{self.reader.STEP}, on the {self.pixel_m:g} m grid and on a {self.coarse_m:g} m grid, averaged")

    def probabilities(self, image):
        """
        image: (3, H, W), top-of-atmosphere reflectance (0 to about 1), red, green, blue, on the model's grid
        (pixel_m metres); no data: every band 0.
        Returns (probs, nothing): probs (4, H, W) float32, the probabilities of clear, thick cloud, thin cloud and
        cloud shadow (the processor's order); nothing (H, W) bool, where the model read no data.
        """
        image = np.ascontiguousarray(image, dtype=np.float32)
        if image.ndim != 3 or image.shape[0] != len(BANDS):
            raise ValueError(f"expected an image of {len(BANDS)} bands (bands, H, W), got {image.shape}")
        classes, probs = self.reader.classify(image, self.own, self.rough, self.factor, self.batch)
        nothing = classes == self.reader.NO_DATA
        return np.ascontiguousarray(probs[TO_PROCESSOR_ORDER], dtype=np.float32), nothing
