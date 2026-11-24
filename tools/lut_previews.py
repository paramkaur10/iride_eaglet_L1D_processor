#!/usr/bin/env python3
"""
lut_previews.py: whole-scene previews of every product of the LUT build (tools/build_lut.py), side by side:
    as delivered | current LUT | new LUT, one balance | new LUT, balance per satellite | Sentinel-2 L1C
The colour step is the processor's (divide by the balance, then the table with "vegetation around"), on the
image averaged to 20 m. Each EAGLET panel is clipped to what the L1D can store (reflectance -0.1 to 6.45) and
drawn like the L1D quicklooks: one linear stretch for its three bands, 1st to 99th percentile, so the colours
stay as they are. A product whose calibration is wrong comes out saturated (white), as its L1D would.

    python tools/lut_previews.py                 # after tools/build_lut.py; PNGs in <work>/previews
"""

import argparse
import sys
import warnings
from pathlib import Path

import numpy as np
import rasterio
from rasterio.enums import Resampling

sys.path.insert(0, str(Path(__file__).resolve().parent))
import build_lut as bl                                         # noqa: E402

L1D_RANGE = (-0.1, (65535 - 1000) / 10000)                     # reflectance the L1D's uint16 can hold


def stretch(img):
    """one linear stretch for the three bands (1st to 99th percentile), as the L1D quicklook; no data white"""
    ok = np.isfinite(img).all(0)
    out = np.ones(img.shape[1:] + (3,))
    if ok.sum() < 100:
        return out, True
    lo, hi = np.percentile(img[:, ok], [1, 99])
    if hi - lo < 1e-9:                                          # everything the same value: saturated
        out[ok] = 0.92
        return out, True
    rgb = np.clip((np.moveaxis(img, 0, -1) - lo) / (hi - lo), 0, 1)
    out[ok] = rgb[ok]
    return out, False


def satellite_balances(fit):
    """the colour balance of each satellite (median over its products with a good fit) and how many products"""
    if "prod_bal" not in fit:
        return {}
    by_sat = {}
    for sat, bal, ok in zip(fit["prod_sat"], fit["prod_bal"], fit["prod_bal_ok"]):
        if ok:
            by_sat.setdefault(str(sat), []).append(bal)
    return {sat: (np.median(b, axis=0), len(b)) for sat, b in by_sat.items()}


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--work", type=Path, default=bl.PROJECT / "lut_a01", help="the build_lut.py folder")
    ap.add_argument("--template", type=Path, help="the current colour AUX file (default: the newest in aux/)")
    ap.add_argument("--res", type=float, default=20.0, help="preview pixel size in metres (default: %(default)s)")
    a = ap.parse_args(argv)
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    prods = bl.load_inventory(a.work)
    tpl = bl.read_aux(a.template or bl.newest_aux(bl.REPO / "aux"))
    fit = bl.load_fit(a.work)
    sat_bal = satellite_balances(fit)
    if not sat_bal:
        print("!! lut_fit.npz has no balance per satellite (made by an older build_lut.py): that panel is left out")
    window = max(1, int(round(tpl["window"] * bl.RES / a.res)) | 1)            # about the same 70 m on the ground
    lut = bl.Lut(tpl["axes"], window)
    out_dir = a.work / "previews"
    out_dir.mkdir(exist_ok=True)

    for p in prods:
        with rasterio.open(p["tif"]) as src:
            f = max(1, int(round(a.res / src.res[0])))
            dn = src.read(out_shape=(3, max(1, src.height // f), max(1, src.width // f)),
                          resampling=Resampling.average).astype("float64")
        gains, offs = np.array(p["gains"])[:, None, None], np.array(p["offsets"])[:, None, None]
        refl = np.where((dn > 0).all(0), dn * gains + offs, np.nan)

        def corrected(q, veg_share, bal):
            bal = np.asarray(bal, "float64")[:, None, None]
            e = refl / bal
            around = lut.around_map(e, veg_share)
            ok = np.isfinite(e).all(0)
            out = np.full_like(e, np.nan)
            out[:, ok] = lut.apply(q, e[:, ok].T, around[ok]).T
            return out

        panels = [("as delivered", refl),
                  ("current LUT (in the processor now)", corrected(tpl["q"], tpl["veg_share"], tpl["balance"])),
                  ("new LUT, one balance " + "/".join(f"{v:.2f}" for v in fit["balance"]),
                   corrected(fit["q"], fit["veg_share"], fit["balance"]))]
        if p["sat"] in sat_bal:
            b, n = sat_bal[p["sat"]]
            panels.append((f"new LUT, {p['sat']} balance " + "/".join(f"{v:.2f}" for v in b)
                           + f" ({n} image{'s' if n > 1 else ''})", corrected(fit["q"], fit["veg_share"], b)))
        elif sat_bal:
            panels.append((f"new LUT, no balance measured for {p['sat']}", np.full_like(refl, np.nan)))
        s2, _ = bl.load_composite(a.work, p)
        if s2 is not None:
            k = max(1, int(round(a.res / bl.RES)))
            h, w = s2.shape[1] // k * k, s2.shape[2] // k * k
            with warnings.catch_warnings():                             # cells all cloud stay empty
                warnings.simplefilter("ignore", RuntimeWarning)
                s2 = np.nanmean(s2[:3, :h, :w].reshape(3, h // k, k, w // k, k), axis=(2, 4))
            panels.append(("Sentinel-2 L1C (closest clear pass)", s2))

        fig, axes = plt.subplots(1, len(panels), figsize=(4.6 * len(panels), 4.6 * refl.shape[1] / refl.shape[2] + 0.8),
                                 squeeze=False)
        for ax, (title, img) in zip(axes[0], panels):
            if not title.startswith("Sentinel-2"):
                img = np.clip(img, *L1D_RANGE)
            rgb, saturated = stretch(img)
            ax.imshow(rgb, interpolation="antialiased")
            ax.set_title(title + ("\n(saturated: the L1D would be white)" if saturated and np.isfinite(img).any() else ""),
                         loc="left", fontsize=9)
            ax.set_xticks([])
            ax.set_yticks([])
        note = "" if p.get("calibration_ok", True) else "   CALIBRATION LOOKS WRONG (XML gains)"
        fig.suptitle(f"{p['label']}  {p['stem']}{note}", x=0.01, ha="left", fontsize=10)
        fig.tight_layout(rect=(0, 0, 1, 0.94))
        path = out_dir / f"preview_{p['sat']}_{p['sensing']}.png"
        fig.savefig(path, dpi=110)
        plt.close(fig)
        print(f"{p['label']}: {path}")


if __name__ == "__main__":
    main()
