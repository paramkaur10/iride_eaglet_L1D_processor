# Tools

Offline tools around the processor: fitting and checking the colour LUT, comparing L1D products, and figures. They
are not part of the processor: `tools/` is left out of `git archive` (`.gitattributes`), so it is neither in the
Docker build context nor in GitHub's Download ZIP. Clone the repository to get them.

| Tool | What it does |
|---|---|
| **Colour LUT** | |
| `build_lut_v4.py` | fits a new LUT (method v4) against Sentinel-2 L1C and writes a new colour AUX file: the current tool |
| `build_lut.py` | the previous method (v3), which the built-in A01 LUT was made with; also the helpers the other LUT tools use |
| `build_colour_aux.py` | turns a LUT-only AUX file (version 3) and the class model script into the version 4 AUX file the processor reads |
| `lut_panels.py` | each product side by side: L1C, L1D with the current LUT, L1D with the new LUT, Sentinel-2 |
| `lut_previews.py` | whole-scene previews of a LUT build: as delivered, current LUT, new LUT, Sentinel-2 |
| `inspect_l1c.py` | what a folder of L1C products holds, batch by batch, and whether their radiometric coefficients agree |
| `lut_tool.sh`, `docker/` | the tools image (the 01.01 processor image plus opencv, requests and matplotlib) and the script that runs a tool in it |
| **Running the processor** | |
| `make_job_order.py`, `sample_job_order.xml` | a Job Order for a local test run of the processor |
| `run_l1d.sh` | runs L1C products through the processor's image as PDGS does, with a given colour AUX file |
| **Comparing and showing L1D products** | |
| `compare_l1d.py` | two L1D products of the same L1C, file by file: what differs, and how much their cloud masks agree |
| `compare_panels.py` | two L1D versions of the same products side by side: quicklooks, cloud masks, cloud confidence and 2 km zooms where they differ |
| `cloud_panels.py` | two L1D versions side by side with a map of what changed in the image (the cloud experiment) |
| `readme_figure.py` | the README pictures: each L1D product in one row, L1C, Sentinel-2, L1D, cloud mask and cloud confidence |
| `l1d_panels.py` | poster panels of L1D products: L1C, L1D, cloud mask and confidence, the scene on a map, the product facts |
| `cloud_experiment/` | the test of 30 Oct 2026 that keeps clouds neutral in the colour step (not in the processor) |

The tools that only read L1D products (`compare_l1d.py`, `compare_panels.py`) run in the processor's own image,
for example:

```bash
docker run --rm -i --network none -v <folder A>:/a:ro -v <folder B>:/b:ro \
    ipf-ime-msi-l1d:01.02 python - /a /b < tools/compare_l1d.py
```

`readme_figure.py` runs in the processor's image; to download Sentinel-2 when no LUT build has it on disk
(`--fetch`) it also needs `requests`, installed for the run: the command is at the top of the file.

## Fit a new LUT

`build_lut_v4.py` reads the L1C products of one processing baseline, takes Sentinel-2 L1C over each footprint from
the Copernicus Data Space, aligns Eaglet-2 with Sentinel-2, measures the near-infrared share of each band and the
colour balance, fits the table and writes a new colour AUX file from the current one (the class model and
`<Method>` unchanged). Then it scores the current and the new LUT against Sentinel-2. The text at the top of the
file gives the steps in detail.

You need:

- the tools image, `eaglet-lut-tools`, built on the 01.01 processor image `ipf-ime-msi-l1d:01.01`: the LUT tools
  screen clouds in Eaglet-2 and in Sentinel-2 with OmniCloudMask, which that image has;
- a Copernicus Data Space client (Sentinel Hub OAuth client id and secret). The tools read them from
  `CDSE_CLIENT_ID` and `CDSE_CLIENT_SECRET`, or ask for them; they are never written to disk;
- the L1C zips, in one folder.

From the repository folder:

```bash
sudo docker build --network host -t eaglet-lut-tools tools/docker
TOOLS=$PWD/tools bash tools/lut_tool.sh build_lut_v4.py --work /lut/work/v5 --products /data --since 20261017
```

`lut_tool.sh` mounts `~/lut` as `/lut` (work folders go in `/lut/work`), the L1C zips of `~/l1d_ohb/data` as
`/data` and the processor's working folders `~/l1d_ohb/runs` as `/runs`. Set `LUT`, `DATA` and `RUNS` to use other
folders.

The results are in the work folder: `aux/<new AUX file>`, `summary.txt` (the scores and a verdict), `scores.csv`
and `lut_check.png`. Each step keeps its results on disk; `--redo <step>` does a step and the ones after it again
(`--redo fit` is quick). `--exclude-sats` leaves satellites out of the fit, for example products made with old
calibration coefficients. `--help` lists every option.

A new AUX file keeps the validity of the current one (from 17 Oct 2026, the first day of the A01 products) and gets
the time it was made as its creation time.

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
