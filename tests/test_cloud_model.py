# test_cloud_model.py: the cloud model package (Clarity) as the processor reads it. Needs the package unzipped in
# models/ (models/README.md) and PyTorch; skipped without them.
import shutil

import numpy as np
import pytest

pytest.importorskip("torch")
from iride_eaglet_processor.inference import (  # noqa: E402
    MODEL_DIR, TO_PROCESSOR_ORDER, CloudModel, ModelError, check_package,
)

pytestmark = pytest.mark.skipif(not (MODEL_DIR / "SHA256SUMS").is_file(),
                                reason="no cloud model package in models/ (see models/README.md)")


@pytest.fixture(scope="module")
def model():
    return CloudModel()


def scene():
    """a small scene in reflectance: a bright cloud on dark land, and a no-data corner"""
    y, x = np.mgrid[0:300, 0:300] / 300
    land = 0.06 + 0.02 * np.sin(20 * x) * np.cos(15 * y)
    cloud = 0.6 * np.exp(-((x - 0.6) ** 2 + (y - 0.4) ** 2) / 0.02)
    img = np.stack([land + cloud, land * 1.1 + cloud, land * 1.3 + cloud]).astype("float32")
    img[:, :40, :40] = 0
    return img


def test_the_package_is_checked_file_by_file(tmp_path):
    copy = tmp_path / "clarity"
    shutil.copytree(MODEL_DIR, copy)
    assert len(check_package(copy)) >= 4
    net = copy / "clarity_eaglet2.pt"
    data = bytearray(net.read_bytes())
    data[len(data) // 2] ^= 0xFF                                  # one byte changed
    net.write_bytes(bytes(data))
    with pytest.raises(ModelError, match="not the delivered file"):
        CloudModel(copy)
    net.unlink()
    with pytest.raises(ModelError, match="missing"):
        CloudModel(copy)


def test_probabilities_in_the_processor_order(model):
    img = scene()
    probs, nothing = model.probabilities(img)
    assert probs.shape == (4, 300, 300) and probs.dtype == np.float32
    assert nothing[:40, :40].all() and not nothing[100:, 100:].any()
    np.testing.assert_allclose(probs[:, ~nothing].sum(0), 1, atol=1e-4)
    # the same reading as the package's own runner, with the classes put in the processor's order
    _, theirs = model.reader.classify(img, model.own, model.rough, model.factor, model.batch)
    np.testing.assert_array_equal(probs, np.asarray(theirs[TO_PROCESSOR_ORDER], dtype=np.float32))
    # the bright cloud is cloud (thick + thin), the dark land is not
    cloud = probs[1] + probs[2]
    assert cloud[120, 180] > 0.5 and cloud[280, 20] < 0.5
