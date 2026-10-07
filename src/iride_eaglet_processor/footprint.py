# footprint.py

import numpy as np


def get_valid_footprint(img, nodata_value=0, eps=1e-6):
    x = np.asarray(img).astype("float32")

    if x.ndim != 3:
        raise ValueError(f"Expected 3D image, got {x.shape}")

    # CHW -> HWC
    if x.shape[0] <= 20 and x.shape[-1] > 20:
        x = np.transpose(x, (1, 2, 0))

    finite = np.all(np.isfinite(x), axis=-1)
    nonzero = np.any(np.abs(x - nodata_value) > eps, axis=-1)

    return finite & nonzero