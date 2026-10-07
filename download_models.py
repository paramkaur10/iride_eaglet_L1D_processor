#!/usr/bin/env python3
"""
download_models.py

ONE-TIME, BUILD-TIME setup step. Run this on a machine WITH internet
access (e.g. during container build, or on any dev machine before
packaging for deployment) to pre-download OmniCloudMask's neural network
weights from Hugging Face into a local cache bundled alongside this
processor.

process_l1c.py runs with HF_HUB_OFFLINE=1 (no network calls at runtime,
required for air-gapped environments such as IRIDE PDGS). This cache
MUST be populated before deploying the processor to any network-isolated
target — process_l1c.py will fail at runtime otherwise, since it has no
way to reach Hugging Face to fetch the weights on demand.

Usage (run once, with internet access, before packaging for deployment):
    python download_models.py

After running, the "hf_cache/" directory next to this script will
contain the model weights. Include that directory when packaging this
processor for delivery to an air-gapped target (e.g. when zipping the
repo for PDGS deployment) — it is git-ignored by default since it
contains large binary files that don't belong in version control.
"""

import os
import sys
from pathlib import Path

_PROCESSOR_ROOT = Path(__file__).resolve().parent
HF_CACHE_DIR = _PROCESSOR_ROOT / "hf_cache"

# Deliberately NOT setting HF_HUB_OFFLINE here — this script's entire
# purpose is to reach the network and populate the cache. Only
# process_l1c.py itself runs offline.
os.environ["HF_HOME"] = str(HF_CACHE_DIR)

sys.path.insert(0, str(_PROCESSOR_ROOT / "src"))

import numpy as np
from omnicloudmask import predict_from_array


def main():
    HF_CACHE_DIR.mkdir(parents=True, exist_ok=True)
    print(f"Downloading OmniCloudMask model weights into: {HF_CACHE_DIR}")
    print("(requires internet access)")

    # A single small-scene inference call is enough to trigger download of
    # every model component OmniCloudMask needs (its default ensemble uses
    # two backbones — confirmed in practice to both download from one call).
    dummy = np.zeros((3, 64, 64), dtype="float32")
    predict_from_array(
        dummy,
        export_confidence=True,
        softmax_output=True,
        inference_device="cpu",
        inference_dtype="fp32",
    )

    print("Done. Model weights cached locally.")
    print(f"Include the '{HF_CACHE_DIR.name}/' directory when packaging "
          f"this processor for deployment to an air-gapped target.")


if __name__ == "__main__":
    main()