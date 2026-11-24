# Changelog

All versions of the `EAGLET2_L1D_PROC` processor. The task version stays `01.00`, so the PDGS Job Orders do not
change between processor versions.

## 01.02 (24 Nov 2026)

- **Cloud mask by Clarity**, a sensor-agnostic cloud model that reads each scene on a 5 m and a 15 m grid. The CLM
  classes and rule and the meaning of the COM are unchanged. The model package is checked against its `SHA256SUMS` at
  every run and when the image is built.
- **Saturated pixels white**: where the L1C is saturated (DN 65535 in any band), the L1D pixel is white instead of
  purple. The log gives their number, and the image's `SATURATED` tag the rule. The rest of the L1D image is
  unchanged.
- **New quicklook stretch**: each band from its own dark level, one span for the three bands, then a gamma of 1.6.
  The blue, greyish haze of the single stretch is gone. The STAC display range follows the same rule.
- **Lower memory peak**: the full-resolution image is freed before the cloud model runs (about 15% lower in tests).
- **Colour AUX file valid from 17 Oct 2026**, the first day of the A01 products its LUT was fitted on, instead of
  1 Jan 2024. Its validity stop (2099) and its tables are unchanged:
  `IME___AXE1D_CLUT_20261017T000000_20990101T000000_20261019T112255____________O_001`.
- **Docker image**: the same base image as 01.01 (pinned by digest to the copy 01.01 was built on), so the same
  system packages. The previous cloud model and the Python packages only it used are no longer installed; every
  other package keeps its 01.01 version.
- **Processing time**: about 6.5 to 9.5 minutes per product with 2 CPUs, instead of 3 to 4.
- **Repository**: all processor modules in `src/iride_eaglet_processor/`; new README, changelog, tests and tools.

Known limitation: Clarity's networks are TorchScript files, and PyTorch 2.14 warns that `torch.jit.load` is not
supported on Python 3.14. With the image's pinned versions they load and run as they should, and every image build
runs them once, so a break would stop the build, not a PDGS run. A later Clarity package will use `torch.export`.

## 01.01 (20 Oct 2026)

- **Colour AUX file as PDGS gives it**: the processor reads the file zipped (`<ID>.zip` holding `<ID>/<ID>.xml` and
  its `<ID>/<ID>.hdr`), and checks it before anything else. 01.00 read only the bare XML, and its first PDGS run with
  an AUX file in the Job Order stopped with exit code 255.
- **Cloud rule**: a pixel is cloud when its cloud confidence (thick plus thin cloud) is at least 0.5. Before, each
  pixel took the most likely of the four classes, which left clear the clouds whose probability was split between
  thick and thin (clear 0.4, thick 0.3, thin 0.3 is 0.6 cloud).
- **A01 LUT built in**: fitted on seven A01 products with the corrected calibration coefficients (17 and
  18 Oct 2026). The first A01 products (13 to 15 Oct) had calibration problems and were not used.

## 01.00 (Oct 2026)

First version for the PDGS:

- the L1D product: the L1C layout, the colour-corrected image in `IMG_DATA`, the cloud mask and cloud confidence in
  `AUX_DATA`; the product name with the environment and processing baseline of the Job Order;
- the XML header of the L1D template; the colour AUX file named in `AUXILIARY_FILE` and kept out of `AUX_DATA`;
- the STAC L1D identity agreed for the marketplace;
- the quicklooks: the PNG transparent where there is no data, placed by the KML;
- the Docker image built as PDGS built the previous Eaglet-2 processor; 2 CPUs.
