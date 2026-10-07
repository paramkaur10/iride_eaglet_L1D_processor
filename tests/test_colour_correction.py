# test_colour_correction.py: the colour step's methods, switch, AUX checks and encoding.
import json

import numpy as np
import pytest
from scipy.interpolate import RegularGridInterpolator

from conftest import make_model, make_tables, write_aux, STEP, X_START, Y_START, NX, NY, NV
from iride_eaglet_processor.colour_correction import (
    ColourCorrection, ColourCorrectionError, encode, ENCODING_OFFSET, L1D_GAIN, L1D_OFFSET,
    percentiles_from_counts,
)


def enrico_apply(x, m):
    """apply() copied verbatim from Enrico's applica_L1C_RGB.py v2 (works on the JSON lists as they are)"""
    z = (x - m['center']) / m['scale']; logs = []
    for g in m['components']:
        v = []
        for mu, p, lw in zip(g['mean'], g['precision'], g['log_weight']):
            d = z - mu; v.append(lw - .5 * np.einsum('ni,ij,nj->n', d, p, d))
        v = np.stack(v, 1); mx = v.max(1); logs.append(mx + np.log(np.exp(v - mx[:, None]).sum(1)))
    logs = np.stack(logs, 1); w = np.exp(logs - logs.max(1)[:, None]); w /= w.sum(1)[:, None]
    out = np.zeros_like(x)
    for k, c in enumerate((1, 2, 3)):
        expert = m['experts'][str(c)]
        if c == 1: y = np.column_stack([np.interp(x.mean(1), expert['xq'], expert['yq'][j]) for j in range(3)])
        else: y = np.c_[(x - expert['mean']) / expert['scale'], np.ones(len(x))] @ np.array(expert['coef'])
        out += y * w[:, k, None]
    return out, w


@pytest.fixture
def pixels():
    rs = np.random.default_rng(7)
    rho = rs.uniform(0.02, 0.5, (5000, 3))
    around = rs.uniform(0, 1, 5000)
    return rho, around


def test_lut_matches_an_independent_interpolation(aux_v4, pixels):
    rho, around = pixels
    cc = ColourCorrection(aux_v4, method="LUT")
    q, _ = make_tables()
    grid = (X_START + STEP * np.arange(NX), Y_START + STEP * np.arange(NY), np.linspace(0, 1, NV))
    E = rho / np.array([0.92, 1.0, 1.29])
    pts = np.column_stack([np.log(E[:, 0] / E[:, 1]), np.log(E[:, 2] / E[:, 1]), around])
    pts = np.clip(pts, [g[0] for g in grid], [g[-1] for g in grid])          # clamped at the edges
    expected = np.column_stack([E[:, b] * RegularGridInterpolator(grid, q[b].astype("float64"))(pts) for b in range(3)])
    np.testing.assert_allclose(cc.apply(rho, around), expected, rtol=1e-10)


def test_class_model_is_exactly_enricos(aux_v4, pixels):
    rho, _ = pixels
    cc = ColourCorrection(aux_v4, method="CLASS_MODEL")
    ref, _ = enrico_apply(rho, make_model())
    np.testing.assert_array_equal(cc.apply(rho), np.maximum(ref, 0))


def test_hybrid_is_his_colour_on_vegetation_and_water_else_the_lut(aux_v4, pixels):
    rho, around = pixels
    rs = np.random.default_rng(11)
    outside = rs.uniform(0.85, 1.2, (500, 3))                 # beyond his training range: LUT only
    rho, around = np.vstack([rho, outside]), np.concatenate([around, rs.uniform(0, 1, 500)])
    lut = ColourCorrection(aux_v4, method="LUT").apply(rho, around)
    hyb_cc = ColourCorrection(aux_v4, method="HYBRID")
    his, w = hyb_cc.class_model(rho)
    m = hyb_cc.model
    inside = ~((rho < m["low"]) | (rho > m["high"])).any(1)
    alpha = np.where(inside, w[:, 0] + w[:, 1], 0.0)[:, None]
    s_his = his.sum(1, keepdims=True)
    ok = s_his[:, 0] > 1e-4
    expected = alpha * his * lut.sum(1, keepdims=True) / np.where(ok[:, None], s_his, 1.0) + (1 - alpha) * lut
    np.testing.assert_allclose(hyb_cc.apply(rho, around)[ok], expected[ok], rtol=1e-12)
    assert (alpha[:, 0] == 0).any() and (alpha[:, 0] > 0.5).any()      # the test covers both cases


def test_method_comes_from_the_file_and_can_be_overridden(tmp_path):
    p = write_aux(tmp_path / "aux.xml", method="CLASS_MODEL")
    assert ColourCorrection(p).method == "CLASS_MODEL"
    assert ColourCorrection(p, method="lut").method == "LUT"
    assert not ColourCorrection(p).needs_around
    with pytest.raises(ColourCorrectionError, match="unknown colour correction method"):
        ColourCorrection(p, method="ENRICO")


def test_version_3_file_is_lut_only(tmp_path):
    p = write_aux(tmp_path / "aux_v3.xml", with_model=False)
    assert ColourCorrection(p).method == "LUT"
    with pytest.raises(ColourCorrectionError, match="no Class_Model"):
        ColourCorrection(p, method="CLASS_MODEL")


@pytest.mark.parametrize("part", ["q_band2", "veg_share", "Class_Model"])
def test_corrupted_aux_is_refused(tmp_path, part):
    p = write_aux(tmp_path / "aux_bad.xml", corrupt=part)
    with pytest.raises(ColourCorrectionError, match="checksum mismatch"):
        ColourCorrection(p)


def test_missing_or_broken_file_raises_colour_error(tmp_path):
    with pytest.raises(ColourCorrectionError):
        ColourCorrection(tmp_path / "nothing.xml")
    (tmp_path / "broken.xml").write_text("<Earth_Explorer_File><Data_Block>")
    with pytest.raises(ColourCorrectionError):
        ColourCorrection(tmp_path / "broken.xml")


def test_build_colour_aux_adds_enricos_model_and_the_switch(tmp_path, capsys):
    from build_colour_aux import main as build_colour_aux
    v3 = write_aux(tmp_path / "v3.xml", with_model=False)
    script = tmp_path / "applica_L1C_RGB.py"                  # his script's MODEL line, as in the real one
    script.write_text("import json\nMODEL = json.loads(r'''" + json.dumps(make_model()) + "''')\n")
    build_colour_aux(["--lut-aux", str(v3), "--class-model", str(script), "--method", "HYBRID",
                      "--out-dir", str(tmp_path / "out")])
    assert "PROBLEM" not in capsys.readouterr().out
    out, = (tmp_path / "out").glob("IME___AXE1D_CLUT_20240101T000000_20990101T000000_*____________O_001.xml")
    cc = ColourCorrection(out)
    assert cc.method == "HYBRID" and cc.version == "4"
    np.testing.assert_array_equal(cc.q, ColourCorrection(v3).q)
    rho = np.random.default_rng(1).uniform(0.02, 0.5, (2000, 3))
    np.testing.assert_array_equal(ColourCorrection(out, method="CLASS_MODEL").apply(rho),
                                  np.maximum(enrico_apply(rho, make_model())[0], 0))


def test_encoding_follows_the_sentinel2_convention():
    rho = np.array([[0.1234, 0.0, -0.2], [1.5, 6.6, 0.00004]])
    out = encode(rho)
    assert out.dtype == np.uint16
    assert out[0].tolist() == [1234 + ENCODING_OFFSET, ENCODING_OFFSET, 1]    # never 0 (= no data) for a valid pixel
    assert out[1].tolist() == [16000, 65535, ENCODING_OFFSET]
    assert (L1D_GAIN, L1D_OFFSET) == (0.0001, -0.1)                            # what the XML and STAC get
    np.testing.assert_allclose(out[0, :2] * L1D_GAIN + L1D_OFFSET, [0.1234, 0.0], atol=1e-12)


def test_display_range_is_the_2nd_and_98th_percentile_of_each_band():
    rs = np.random.default_rng(5)
    values = rs.integers(1000, 5000, (3, 100000))
    counts = np.stack([np.bincount(v, minlength=65536) for v in values])
    got = percentiles_from_counts(counts)
    for b in range(3):
        v = np.sort(values[b])                       # the k-th smallest value, k = n * p / 100 rounded up
        assert got[b] == [int(v[int(np.ceil(v.size * p / 100.0)) - 1]) for p in (2, 98)]
    assert percentiles_from_counts(np.zeros((3, 65536), int)) == [[0, 0]] * 3


def test_vegetation_around_uses_the_window_and_ignores_no_data(aux_v4):
    cc = ColourCorrection(aux_v4)
    blocks = np.full((30, 30, 3), [0.30, 0.10, 0.12])        # band 1 / band 2 high: vegetation-like everywhere
    blocks[:, 15:] = [0.10, 0.12, 0.15]                       # right half: not vegetation
    blocks[0:5, 0:5] = np.nan                                 # some no data
    amap = cc.around_map(blocks)
    assert amap.shape == (30, 30)
    assert amap[15, 5] > 0.9 and amap[15, 25] < 0.2
    assert np.isfinite(amap).all() and amap.min() >= 0 and amap.max() <= 1
    # between block centres the value is interpolated, at the centres it is the block's own
    f = 10
    assert np.isclose(cc.around_at(amap, [15 * f + 5], [5 * f + 5], f)[0], amap[15, 5])