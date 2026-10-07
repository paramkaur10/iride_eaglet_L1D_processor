# IRIDE EAGLET-2 L1C to L1D processor, version 2

Turns an EAGLET-2 L1C RGB3 product into a new **L1D** product: the L1C layout,
with the colour-corrected RGB (true colour) in `IMG_DATA` in place of the L1C
image, and the cloud mask and cloud confidence map in `AUX_DATA`. Follows the
IRIDE Processor ICD (PDGS-ID-XPR-IRIDE-0209 v2.4). The input product is never
modified. Python 3.10 or newer, CPU or GPU.

## Run

```bash
python process_l1c.py <absolute path to JobOrder.xml>
```

The Job Order is the only input: it gives the product zip (`1CSP__RGB3`), the
colour AUX file (`AXE1D_CLUT`, mandatory) and the logging level. The new
product is written as `<NEW_ID>.zip` next to the input zip. No network is
used at runtime (model weights from `hf_cache/`, see Offline deployment).

| Exit code | Meaning |
|---|---|
| `0` | done |
| `255` | failure, including a missing or unusable colour AUX file (there is no L1D image without it) |

At the end the log gives the time of each step, for example
`Processing times (s): unzip … | read + calibrate … | cloud mask … | colour correction … | COG writing … | metadata … | rename + checksums … | zip … | total …`,
followed by the image size, the device used for the cloud mask and the number
of CPUs.

## Output

`<NEW_ID>` is the input ID with `1CSP` changed to `1DSP` and the generation
time set to the time of the run. Every file in the product is renamed to it,
except files named after another product (`AUX_DATA/*.anc.xml`).

| In the L1D product | Content |
|---|---|
| `IMG_DATA/<NEW_ID>.tif` | **the L1D image**: colour-corrected RGB (true colour), Cloud Optimized GeoTIFF, LZW with predictor (lossless), 512 x 512 tiles, internal overviews; uint16 = reflectance x 10000 + 1000 (Sentinel-2 L1C convention), 0 = no data. It replaces the L1C image |
| `AUX_DATA/<NEW_ID>.CLM.tif` | cloud mask, uint8: 0 no data, 1 clear, 2 thick cloud, 3 thin cloud, 4 cloud shadow |
| `AUX_DATA/<NEW_ID>.COM.tif` | cloud confidence, uint16, confidence x 10000 |
| `AUX_DATA/<NEW_ID>.CLM.json` | class legend |
| `<NEW_ID>.json` | STAC: `eo:cloud_cover`; assets `clm`, `com`, `clm_aux`; the `rgb` asset describes the L1D image (COG media type, `raster:scale` 0.0001, `raster:offset` -0.1, `nodata` 0, method and AUX file) and its render range is recomputed (2nd / 98th percentile per band); new id and times; `file:checksum` / `file:size` of every asset recomputed |
| `<NEW_ID>.xml` | the L1C XML with `Rescaling_Factors` (GAIN 0.0001, OFFSET -0.1) and the image SHA-256 set for the L1D image; nothing else changed |
| `PREVIEW/*` | as in the L1C product, renamed |

Files that are already compressed (`.tif`, `.png`, `.jpg`) are stored in the
zip without compressing them again.

## Colour correction

EAGLET-2 band 1 picks up about 40% near-infrared, so vegetation looks purple.
The colour step brings the colours towards Sentinel-2 L1C. It runs after the
cloud mask, at full resolution, tile by tile, on the product's own grid.

**Input:** reflectance = DN x GAIN + OFFSET, with GAIN and OFFSET from the
L1C product XML (`Rescaling_Factors`, correct from 24 September 2026). The
rounded `raster:scale` of the STAC file is only a fallback. DN = 0 is no data.

**Configuration:** one AUX file,
`IME___AXE1D_CLUT_20240101T000000_20990101T000000_<created>____________O_001.xml`,
whose `<Method>` selects what the processor applies:

| `<Method>` | What it does | Brightness |
|---|---|---|
| `LUT` | colour balance, then a look-up table on ln(b1/b2), ln(b3/b2) and the share of vegetation in the ~70 m around each pixel; one table for the whole constellation | as delivered |
| `CLASS_MODEL` | Enrico Cadau's model (`applica_L1C_RGB.py` v2, stored unchanged): water / vegetation / built-up weights from Gaussian mixtures on the RGB triplet, one transform per class | close to Sentinel-2 |
| `HYBRID` | Enrico's colour where his model sees vegetation or water inside its training range, the LUT elsewhere | as delivered |

To switch, edit `<Method>` in the file. The tables and the model are protected
by SHA-256 checksums, `<Method>` is not. A file without `<Method>` (version 3)
works as `LUT`.

The file is built from the LUT notebook's file (version 3) and Enrico's
script, then read back and checked before it is kept:

```bash
python tools/build_colour_aux.py --lut-aux <version 3 AUX .xml> --class-model <applica_L1C_RGB.py> \
    --method LUT --out-dir <folder>
```

## Testing

From the repository folder:

```bash
pip install -r requirements.txt pytest
python download_models.py           # once: OmniCloudMask weights into hf_cache/
python -m pytest tests              # colour step, calibration, Job Order, tools
```

On a real product (the run folder can be anywhere outside the repository):

```bash
python tools/make_job_order.py <product.zip> --work-dir <run folder> --aux <AUX .xml> [--method HYBRID]
python process_l1c.py <run folder>/JobOrder.xml
```

`make_job_order.py` links the product zip into the run folder and writes the
Job Order from `sample_job_order.xml`. With `--method` it puts a copy of the
AUX file with that method in the run folder. The L1D product is written to the
run folder; `grep "Processing times"` in the log gives the time of each step.

## Offline deployment

`process_l1c.py` sets `HF_HOME=<repository>/hf_cache` and `HF_HUB_OFFLINE=1`
itself. Run `python download_models.py` once on a machine with internet, then
ship `hf_cache/` with the code:

```bash
zip -r iride_eaglet_processor_deploy.zip process_l1c.py job_order.py pdgs_logger.py \
    product_naming.py task_table.xml requirements.txt src/ hf_cache/
```

## PDGS interfaces

| Interface | File | ICD section |
|---|---|---|
| Task Table | `task_table.xml` | 3.1.1 |
| Job Order | `job_order.py` | 3.1.2 |
| Logging (one line per message, stdout / stderr) | `pdgs_logger.py` | 3.1.4 |
| Exit codes | `process_l1c.py` | 3.1.5 |

The processor never scans its working folder: every path comes from the Job
Order.

## Files

```
process_l1c.py              entry point
job_order.py                Job Order parsing (inputs told apart by File_Type)
pdgs_logger.py              PDGS log format
product_naming.py           L1C -> L1D names, product rename
download_models.py          fills hf_cache/ (once, with internet)
task_table.xml              processor definition for PDGS
sample_job_order.xml        Job Order template for local tests
requirements.txt
src/iride_eaglet_processor/
    raster_reader.py        reads the product
    calibration.py          GAIN / OFFSET from the product XML (STAC fallback); L1D values into the XML
    footprint.py            valid pixels
    resampling.py           to and from the 5 m inference grid
    inference.py            OmniCloudMask
    masks.py                CLM and COM rasters
    stats.py                cloud statistics
    metadata.py             STAC sidecar (L1D image, cloud outputs, checksums), CLM legend
    colour_correction.py    LUT, class model, hybrid; the L1D image writer (COG)
tools/
    build_colour_aux.py     builds the colour AUX file
    make_job_order.py       Job Order for a local test run
tests/                      pytest
```

## Open items (to confirm with Enrico / PDGS)

- `Processor_Name` / `Task_Name` (`EAGLET2_L1D_CLM_PROC` / `EAGLET2_L1D_CLM`)
  still say CLM although the processor now also corrects colours.
- `File_Type` codes: `1CSP__RGB3` in, `1DSP__RGB3` out, `AXE1D_CLUT` for the
  colour AUX file (mandatory), with its `Origin` and `Retrieval_Mode`.
- `PREVIEW`: still the L1C quicklook and thumbnail; to regenerate from the L1D
  image?
- The XML is the L1C one apart from `Rescaling_Factors` and the image checksum
  (Level-1C element names, the L1C product name inside, `NUMBER_OF_OUTLIERS`
  computed on the L1C).
- `Executable_File_Name` in `task_table.xml`.
- `Min_Disk_Space` (8192 MB) and `Processing_Timeout` (3600 s), to be measured
  on the largest product.

## Acknowledgements

Cloud masking by [OmniCloudMask](https://github.com/DPIRD-DMA/OmniCloudMask)
(MIT License).