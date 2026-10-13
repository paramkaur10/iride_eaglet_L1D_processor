#!/usr/bin/env python3
"""
download_models.py

ONE-TIME setup step, on a machine WITH internet access: downloads the
OmniCloudMask model weights from Hugging Face into hf_cache/omnicloudmask/
next to this script, the folder the processor reads them from (MODEL_DIR in
src/iride_eaglet_processor/inference.py).

process_l1c.py runs offline and never downloads anything. Without the weights
in hf_cache/omnicloudmask/ it stops with exit code 255. The Docker image copies
the folder (see Dockerfile), so it must be filled before the image is built.

Usage (once, with internet access):
    python download_models.py

hf_cache/ is git-ignored: it holds large binary files that don't belong in
version control.
"""

import os
import sys
from pathlib import Path

_PROCESSOR_ROOT = Path(__file__).resolve().parent

# Hugging Face's own cache and metadata go to hf_cache/ too, not to the home
# folder. Deliberately NOT offline: this script's purpose is to download.
os.environ["HF_HOME"] = str(_PROCESSOR_ROOT / "hf_cache")

sys.path.insert(0, str(_PROCESSOR_ROOT / "src"))

import numpy as np
from omnicloudmask import predict_from_array

from iride_eaglet_processor.inference import MODEL_DIR, model_weights


def main():
    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    print(f"Downloading the OmniCloudMask model weights into: {MODEL_DIR}")
    print("(requires internet access)")

    # One small inference downloads every model of OmniCloudMask's default
    # ensemble into MODEL_DIR and checks that they load.
    dummy = np.zeros((3, 64, 64), dtype="float32")
    predict_from_array(
        dummy,
        export_confidence=True,
        softmax_output=True,
        inference_device="cpu",
        inference_dtype="fp32",
        destination_model_dir=MODEL_DIR,
    )

    for path in model_weights():
        print(f"  {path.name}  {path.stat().st_size / 1e6:.1f} MB")
    print(f"Done. Ship the '{MODEL_DIR.relative_to(_PROCESSOR_ROOT)}/' folder with the processor "
          f"(the Docker image copies it).")


if __name__ == "__main__":
    main()
