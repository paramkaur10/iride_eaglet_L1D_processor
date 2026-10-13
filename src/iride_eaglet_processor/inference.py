# inference.py
"""
OmniCloudMask inference on the whole scene (no tiling).

The model weights are read from MODEL_DIR, hf_cache/omnicloudmask in the processor folder, filled
once by download_models.py. Left to itself, OmniCloudMask would look for them in the home folder of
the user running it (~/.local/share/omnicloudmask/<version>) and download them if missing; in PDGS
the processor runs as a user without a home folder and without network (ICD Appendix E), so the
weights travel with the processor and nothing is ever downloaded at run time.
"""

from pathlib import Path

import numpy as np
import torch
from omnicloudmask import predict_from_array

MODEL_DIR = Path(__file__).resolve().parents[2] / "hf_cache" / "omnicloudmask"


def model_weights(model_dir=MODEL_DIR):
    """the OmniCloudMask weight files in model_dir; an error when there are none"""
    files = sorted(p for p in Path(model_dir).glob("*") if p.is_file() and p.suffix in (".safetensors", ".pth"))
    if not files:
        raise FileNotFoundError(f"No OmniCloudMask model weights in {model_dir}: run download_models.py "
                                f"(once, with internet) before running or deploying the processor")
    return files


def limit_threads(n):
    """PyTorch uses at most n CPU threads. Returns (intra-op, inter-op) thread counts."""
    torch.set_num_threads(n)
    try:
        torch.set_num_interop_threads(n)
    except RuntimeError:            # can be set only once, before any inter-op work
        pass
    return torch.get_num_threads(), torch.get_num_interop_threads()


def run_omnicloud_on_array(image, device, divide_by_10000=False, inference_dtype="fp32", model_dir=MODEL_DIR):
    """
    image: CHW (3, H, W) or HWC (H, W, 3)
    model_dir: the folder with the model weights (see MODEL_DIR)
    Returns:
      mask_binary: H,W uint8
      cloud_conf:  H,W float32
      probs:       4,H,W float32
    """
    x = np.asarray(image)

    if x.ndim != 3:
        raise ValueError(f"Expected 3D image, got {x.shape}")

    if x.shape[0] == 3:
        pass
    elif x.shape[-1] == 3:
        x = np.transpose(x, (2, 0, 1))
    else:
        raise ValueError(f"Expected 3 channels, got {x.shape}")

    x = x.astype("float32")

    if divide_by_10000 and x.max() > 2:
        x = x / 10000.0

    probs = predict_from_array(
        x,
        export_confidence=True,
        softmax_output=True,
        inference_device=device,
        inference_dtype=inference_dtype,
        destination_model_dir=model_dir,
    )

    multiclass = np.argmax(probs, axis=0).astype("uint8")
    mask_binary = np.isin(multiclass, [1, 2]).astype("uint8")
    cloud_conf = (probs[1] + probs[2]).astype("float32")

    return mask_binary, cloud_conf, probs
