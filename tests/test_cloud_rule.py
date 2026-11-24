import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from iride_eaglet_processor.masks import (CLOUD_THRESHOLD, probs_to_cloud_confidence,  # noqa: E402
                                          probs_to_four_class_map)

CASES = [  # clear, thick, thin, shadow -> class (0 clear, 1 thick, 2 thin, 3 shadow)
    ((0.40, 0.30, 0.30, 0.00), 1),   # cloud split between thick and thin: argmax said clear
    ((0.35, 0.40, 0.00, 0.25), 0),   # argmax said thick at 0.40: below 0.5, so clear
    ((0.20, 0.10, 0.20, 0.50), 3),   # not cloud, shadow beats clear
    ((0.10, 0.20, 0.70, 0.00), 2),   # thin
    ((0.50, 0.25, 0.25, 0.00), 1),   # exactly 0.5 counts as cloud; thick on a tie
    ((0.30, 0.00, 0.00, 0.30), 0),   # shadow only when strictly more likely than clear
    ((0.999, 0.001, 0.001, 0.001), 0),
    ((0.001, 0.999, 0.001, 0.001), 1),
]


def test_rule():
    p = np.array([c for c, _ in CASES], "float32").T.reshape(4, 2, 4)
    want = np.array([k for _, k in CASES]).reshape(2, 4)
    got = probs_to_four_class_map(p)
    assert got.dtype == np.uint8 and got.shape == (2, 4)
    assert (got == want).all(), (got, want)
    # the CLM is exactly COM >= threshold
    cloud = np.isin(got, (1, 2))
    assert (cloud == (probs_to_cloud_confidence(p) >= CLOUD_THRESHOLD)).all()


def test_random_consistency():
    rng = np.random.default_rng(1)
    p = rng.dirichlet((0.5, 0.5, 0.5, 0.5), size=(300, 200)).astype("float32")
    p = np.clip(np.moveaxis(p, -1, 0) + 0.001, 0.001, 0.999)           # as OmniCloudMask's softmax output
    got = probs_to_four_class_map(p)
    com = p[1] + p[2]
    assert ((got == 1) | (got == 2)).sum() == (com >= 0.5).sum()
    assert (got[(com >= 0.5) & (p[1] >= p[2])] == 1).all()
    assert (got[(com >= 0.5) & (p[1] < p[2])] == 2).all()
    assert (got[(com < 0.5) & (p[3] > p[0])] == 3).all()
    assert (got[(com < 0.5) & (p[3] <= p[0])] == 0).all()
    old = np.argmax(p, axis=0)
    print(f"random: old cloud {np.isin(old, (1, 2)).mean():.3f}, new cloud {(com >= 0.5).mean():.3f}")


if __name__ == "__main__":
    test_rule()
    test_random_consistency()
    print("cloud rule tests passed")
