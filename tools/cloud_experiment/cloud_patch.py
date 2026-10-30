#!/usr/bin/env python3
# cloud_patch.py: EXPERIMENT, not a release. Changes the colour step of the L1D processor (01.01) so that clouds
# keep a neutral colour: where the processor's own cloud confidence (COM, OmniCloudMask's thick + thin cloud
# probability) is high, the pixel becomes grey at the brightness the LUT gives it; very bright pixels the cloud
# mask misses (snow) get one common factor on the three bands, so the LUT cannot change their hue. Darker ground
# keeps the full LUT. Applied to the processor folder given (inside the image build); stops without changing
# anything if a line it needs is not there exactly once.
#     python cloud_patch.py /opt/ipf-ime-msi-l1d/iride_eaglet_processor_v2-main
import sys
from pathlib import Path

ROOT = Path(sys.argv[1])
CC = ROOT / "src" / "iride_eaglet_processor" / "colour_correction.py"
PL = ROOT / "process_l1c.py"

CC_EDITS = [
    ("def write_colour_product(tif_path, gains, offsets, out_path, cc, logger, tile=1024, strip=512):\n",
     '''CLOUD_NEUTRAL = (0.3, 0.6)     # EXPERIMENT: cloud confidence from which a pixel is pulled to neutral grey, fully
BRIGHT_ONE_FACTOR = (0.45, 0.6)  # EXPERIMENT: balanced green from which the three bands get one common factor, fully


def keep_clouds_neutral(E, out, conf):
    """EXPERIMENT. E: the balanced input (n, 3); out: the method's output (n, 3); conf: cloud confidence (n,), 0 to 1.
    Bright pixels get one common factor on the three bands (the mean of the method's three), so the hue stays as
    delivered; cloud pixels become grey at that brightness. Both blend in gradually; pixels with a band <= 0 are
    left as the method made them"""
    res = out.copy()
    ok = (E > 0).all(1) & np.isfinite(out).all(1)
    if not ok.any():
        return res
    e, o = E[ok], out[ok]
    qm = (o / e).mean(1, keepdims=True)
    lo, hi = BRIGHT_ONE_FACTOR
    wb = np.clip((e[:, 1:2] - lo) / (hi - lo), 0, 1)
    r = (1 - wb) * o + wb * e * qm
    lo, hi = CLOUD_NEUTRAL
    wc = np.clip((conf[ok][:, None] - lo) / (hi - lo), 0, 1)
    res[ok] = (1 - wc) * r + wc * e.mean(1, keepdims=True) * qm
    return res


def write_colour_product(tif_path, gains, offsets, out_path, cc, logger, tile=1024, strip=512, cloud_path=None):
'''),
    ('            profile = dict(driver="GTiff", width=W, height=H, count=3, dtype="uint16", nodata=NODATA_OUT,\n',
     '''            com = None                     # EXPERIMENT: the cloud confidence map (COM), to keep clouds neutral
            if cloud_path is not None and cc.method in ("LUT", "HYBRID"):
                com = rasterio.open(cloud_path)
                if (com.height, com.width) != (H, W):
                    logger.warning(f"Colour: cloud confidence {com.width} x {com.height}, image {W} x {H}: "
                                   "clouds not kept neutral")
                    com.close()
                    com = None
                else:
                    logger.info(f"Colour: EXPERIMENT, clouds kept neutral: grey from cloud confidence "
                                f"{CLOUD_NEUTRAL[0]} (fully at {CLOUD_NEUTRAL[1]}); one factor for the three bands "
                                f"from balanced green {BRIGHT_ONE_FACTOR[0]} (fully at {BRIGHT_ONE_FACTOR[1]})")
            profile = dict(driver="GTiff", width=W, height=H, count=3, dtype="uint16", nodata=NODATA_OUT,
'''),
    ("                            values = encode(cc.apply(rho, around))\n",
     '''                            corrected = cc.apply(rho, around)
                            if com is not None:
                                conf = com.read(1, window=win)[rr, cc_].astype("float64") / 10000.0
                                corrected = keep_clouds_neutral(rho / cc.balance, corrected, conf)
                            values = encode(corrected)
'''),
    ("        t_corrected = time.perf_counter()\n",
     '''            if com is not None:
                com.close()
        t_corrected = time.perf_counter()
'''),
]
PL_EDITS = [
    ("def make_l1d_image(product, cc, logger):\n", "def make_l1d_image(product, cc, logger, cloud_path=None):\n"),
    ("        info = write_colour_product(tif_path, gains, offsets, l1d_path, cc, logger)\n",
     "        info = write_colour_product(tif_path, gains, offsets, l1d_path, cc, logger, cloud_path=cloud_path)\n"),
    ("    l1d = make_l1d_image(product, cc, logger)\n", "    l1d = make_l1d_image(product, cc, logger, cloud_path=com_path)\n"),
]

texts = {}
for path, edits in ((CC, CC_EDITS), (PL, PL_EDITS)):
    text = path.read_text()
    for old, new in edits:
        if text.count(old) != 1:
            sys.exit(f"{path.name}: a line to change is there {text.count(old)} times, not once: nothing changed\n{old}")
        text = text.replace(old, new)
    texts[path] = text
for path, text in texts.items():
    path.write_text(text)
    print(f"patched {path}")
