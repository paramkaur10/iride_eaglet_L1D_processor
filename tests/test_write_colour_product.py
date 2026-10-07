# test_write_colour_product.py: the full-resolution writer on a small GeoTIFF (needs rasterio).
import numpy as np
import pytest

rasterio = pytest.importorskip("rasterio")
from rasterio.transform import from_origin

from iride_eaglet_processor.colour_correction import (
    ColourCorrection, write_colour_product, encode, percentiles_from_counts,
)

GAINS, OFFSETS = [1.6e-05, 1.4e-05, 1.87e-05], [-0.0006, 0.002, 0.0023]


@pytest.fixture
def l1c_tif(tmp_path):
    """a 3-band uint16 image, 0.92 m pixels (blocks do not fit evenly), with a no-data corner"""
    rs = np.random.default_rng(3)
    H, W = 700, 913
    dn = rs.integers(2000, 30000, (3, H, W)).astype("uint16")
    dn[:, :50, :120] = 0
    path = tmp_path / "IME03_1CSP__RGB3_TEST.tif"
    with rasterio.open(path, "w", driver="GTiff", width=W, height=H, count=3, dtype="uint16",
                       crs="EPSG:32633", transform=from_origin(500000, 4500000, 0.92, 0.92)) as dst:
        dst.write(dn)
    return path, dn


@pytest.mark.parametrize("method", ["LUT", "CLASS_MODEL", "HYBRID"])
def test_written_values_are_the_method_applied_per_pixel(tmp_path, aux_v4, l1c_tif, logger, method):
    path, dn = l1c_tif
    cc = ColourCorrection(aux_v4, method=method)
    out = tmp_path / f"out_{method}.L1D.tif"
    info = write_colour_product(path, GAINS, OFFSETS, out, cc, logger, tile=256, strip=100)
    assert out.exists() and not list(tmp_path.glob("*.part")) and not list(tmp_path.glob("*.tiles.tif"))
    with rasterio.open(out) as src:
        got = src.read()
        assert src.dtypes[0] == "uint16" and src.nodata == 0 and src.count == 3
        assert src.transform == from_origin(500000, 4500000, 0.92, 0.92)
        # a Cloud Optimized GeoTIFF, LZW, with internal overviews
        structure = src.tags(ns="IMAGE_STRUCTURE")
        assert structure.get("LAYOUT") == "COG" and structure.get("COMPRESSION") == "LZW"
        assert src.overviews(1) and src.block_shapes[0] == (512, 512)
        assert src.tags()["COLOUR_METHOD"] == method
    valid = (dn > 0).all(0)
    assert (got[:, ~valid] == 0).all() and (got[:, valid] > 0).all()
    assert info["valid_pixels"] == int(valid.sum()) and info["method"] == method
    counts = np.stack([np.bincount(got[b][valid], minlength=65536) for b in range(3)])
    assert info["rescale"] == percentiles_from_counts(counts)

    # recompute a sample of pixels the slow way, with an independent vegetation-around map
    rr, cc_ = np.nonzero(valid)
    pick = np.random.default_rng(0).choice(len(rr), 3000, replace=False)
    rr, cc_ = rr[pick], cc_[pick]
    rho = dn[:, rr, cc_].T.astype("float64") * GAINS + OFFSETS
    around = None
    if cc.needs_around:
        f = cc.block_factor(0.92)
        H, W = dn.shape[1:]
        X = np.where(valid, dn * np.array(GAINS)[:, None, None] + np.array(OFFSETS)[:, None, None], 0.0)
        nby, nbx = -(-H // f), -(-W // f)
        Xp = np.pad(X, ((0, 0), (0, nby * f - H), (0, nbx * f - W)))
        vp = np.pad(valid, ((0, nby * f - H), (0, nbx * f - W)))
        cnt = vp.reshape(nby, f, nbx, f).sum((1, 3))
        blocks = np.where((cnt > 0.5 * f * f)[..., None],
                          np.moveaxis(Xp.reshape(3, nby, f, nbx, f).sum((2, 4)), 0, -1) / np.maximum(cnt, 1)[..., None],
                          np.nan)
        around = cc.around_at(cc.around_map(blocks), rr, cc_, f)
    expected = encode(cc.apply(rho, around))
    assert np.abs(got[:, rr, cc_].T.astype(int) - expected.astype(int)).max() <= 1      # float32 block sums