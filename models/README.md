# models/

The processor's cloud model, **Clarity for IRIDE Eaglet II (float32)**, goes here as its package,
unzipped exactly as delivered:

```
models/clarity_eaglet2_fp32/
    clarity_eaglet2.pt        the network for the 5 m grid (TorchScript, float32)
    clarity_eaglet2_15m.pt    the same network told a 15 m grid
    sensor.json               its bands (red, green, blue), its pixel (5 m) and its file names
    run_clarity.py            how a scene is read; the processor uses its functions
    MODEL.md                  the network, its checks and its scores
    README.md, requirements.txt
    SHA256SUMS                the SHA-256 of every file above
```

The package is not in git (this repository is public): the folder is git-ignored and only this README
is versioned. Put it in place once per checkout, from the delivered zip:

```bash
cd models
sha256sum clarity_eaglet2_fp32_20261124.zip    # bd272dba334756cde155070279ceff9898976b890c86b198174460c6ac056350
unzip clarity_eaglet2_fp32_20261124.zip
(cd clarity_eaglet2_fp32 && sha256sum -c SHA256SUMS)
```

The processor checks every file against `SHA256SUMS` at every run, before it opens the product: a
missing or changed file stops it with exit code 255. `docker/build_image.sh` copies the folder into
the Docker build context, and the image build checks `SHA256SUMS` again.

A new Clarity release replaces the folder (same name, `clarity_eaglet2_fp32`), with a new processor
version and Docker image.
