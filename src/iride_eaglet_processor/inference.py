# inference.py
"""
OmniCloudMask inference on the whole scene (no tiling).
"""

import numpy as np
from omnicloudmask import predict_from_array


def run_omnicloud_on_array(image, device, divide_by_10000=False, inference_dtype="fp32"):
    """
    image: CHW (3, H, W) or HWC (H, W, 3)
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
    )

    multiclass = np.argmax(probs, axis=0).astype("uint8")
    mask_binary = np.isin(multiclass, [1, 2]).astype("uint8")
    cloud_conf = (probs[1] + probs[2]).astype("float32")

    return mask_binary, cloud_conf, probs