# Tools for the colour LUT

Offline tools to fit a new colour LUT for the processor and to check it, and the cloud experiment. They are not
part of the processor: `tools/` is left out of `git archive` (see `.gitattributes`), so it is neither in the Docker
build context nor in GitHub's Download ZIP. Clone the repository to get them.

| File | What it is |
|---|---|
| `build_lut_v4.py` | fits a new LUT (method v4) and writes a new colour AUX file: the current tool |
| `build_lut.py` | the previous method (LUT v3), which the built-in A01 LUT was made with; also the helpers the panel tools use |
| `lut_panels.py` | each product side by side: L1C, L1D with the current LUT, L1D with the new LUT, Sentinel-2 |
| `cloud_panels.py` | two L1D versions of the same products side by side, with a map of what changed |
| `run_l1d.sh` | runs L1C products through the processor's image as PDGS does, with a given AUX file |
| `lut_tool.sh` | runs one of the Python tools in the tools image |
| `docker/Dockerfile` | the tools image: the processor's image plus opencv, requests and matplotlib |
| `cloud_experiment/` | the cloud test of 30 Oct 2026: a test image that keeps clouds neutral |

## Fit a new LUT

`build_lut_v4.py` reads the L1C products of one processing baseline, takes Sentinel-2 L1C over each footprint from
the Copernicus Data Space, lines EAGLET up with Sentinel-2, measures the near-infrared share of each band and the
colour balance, fits the table and writes a new colour AUX file from the current one (Enrico's class model and
`<Method>` unchanged). Then it scores the current and the new LUT against Sentinel-2. The text at the top of the
file gives the steps in detail.

You need:

- the processor's image `ipf-ime-msi-l1d:01.01`, or the processor's Python environment with its model weights
  (`download_models.py`), plus opencv-python-headless, requests and matplotlib;
- a Copernicus Data Space client (Sentinel Hub OAuth client id and secret). The tools read them from
  `CDSE_CLIENT_ID` and `CDSE_CLIENT_SECRET`, or ask for them; they are never written to disk;
- the L1C zips, in one folder.

With Docker, from the repository folder (as on our server):

```bash
sudo docker build --network host -t eaglet-lut-tools tools/docker
TOOLS=$PWD/tools bash tools/lut_tool.sh build_lut_v4.py --work /lut/work/v5 --products /data --since 20261017
```

`lut_tool.sh` mounts `~/lut` as `/lut` (results: work folders go in `/lut/work`), the L1C zips of
`~/l1d_ohb/data` as `/data` and the processor's working folders `~/l1d_ohb/runs` as `/runs`. Set `LUT`, `DATA`
and `RUNS` to use other folders.

Without Docker, from the repository folder, in the processor's environment:

```bash
python tools/build_lut_v4.py --work ../lut_work --products <folder with the L1C zips> --since 20261017
```

The results are in the work folder: `aux/<new AUX file>`, `summary.txt` (the scores and a verdict), `scores.csv`
and `lut_check.png`. Each step keeps its results on disk; `--redo <step>` does a step and the ones after it again
(`--redo fit` is quick). `--exclude-sats` leaves satellites out of the fit, for example products made with old
calibration coefficients. `--help` lists every option.

## Check a new LUT

1. L1Ds of the same products with the processor's own AUX file (no tag) and with the new one (a tag), from a
   folder that holds a copy of `task_table.xml` and the L1C zips in `data/`:

   ```bash
   sudo bash <repository>/tools/run_l1d.sh data/*.zip
   sudo env AUX=<new AUX file .xml> TAG=LUT_V5 bash <repository>/tools/run_l1d.sh data/*.zip
   ```

2. The panels, in `<work>/final`:

   ```bash
   TOOLS=$PWD/tools bash tools/lut_tool.sh lut_panels.py --work /lut/work/v5 --runs /runs --new-tag LUT_V5
   ```

## The cloud experiment

`cloud_experiment/` holds the test of 30 Oct 2026 that keeps clouds neutral in the colour step: where the
processor's cloud confidence (COM) is high, the pixel becomes grey at the brightness the LUT gives it, and very
bright pixels get one factor for the three bands. It is not in the processor. `cloud_patch.py` changes the 01.01
code inside a test image built on top of `ipf-ime-msi-l1d:01.01` (`Dockerfile`), tagged
`ipf-ime-msi-l1d:01.01-cloudtest`, which never replaces 01.01. `run_cloudtest.sh` is the run as made on our server:
L1Ds of five products with the test image and the v4 AUX file, then `cloud_panels.py` against the same products
with the v4 AUX file alone. To build the test image again:

```bash
cd tools/cloud_experiment && sudo docker build -t ipf-ime-msi-l1d:01.01-cloudtest .
```
