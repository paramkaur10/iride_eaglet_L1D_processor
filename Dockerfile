# Eaglet-2 L1C to L1D processor (EAGLET2_L1D_PROC) as a Docker (OCI) image for IRIDE PDGS,
# following the ICD PDGS-ID-XPR-IRIDE-0209 v2.4, Appendices D, E and F.
#
# Build with docker/build_image.sh, after filling hf_cache/omnicloudmask with the model weights
# (python download_models.py, once, with internet). PDGS runs the image with its wrapper script as
#   docker run --user 1100:600 --cpus <N> -v <WD>:<WD>:rw -w <WD> ... <image>:01.00 EAGLET2_L1D <WD>/JobOrder.<id>.xml
# Nothing is downloaded at run time: the Python packages, the OmniCloudMask weights and the colour
# AUX file are all in the image, readable by any user.

# Python 3.12, as on the machine the processor was validated on
FROM python:3.12-slim

# PyTorch versions of that machine (the image has their CPU-only builds)
ARG TORCH_VERSION=2.14.1
ARG TORCHVISION_VERSION=0.29.1
# set by docker/build_image.sh from task_table.xml
ARG PROCESSOR_VERSION=unknown

LABEL org.opencontainers.image.title="EAGLET2_L1D_PROC" \
      org.opencontainers.image.description="Eaglet-2 L1C to L1D processor: cloud mask and colour correction" \
      org.opencontainers.image.version="${PROCESSOR_VERSION}"

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    PIP_ROOT_USER_ACTION=ignore

# PyTorch: the CPU-only build. The processor never uses a GPU, and the default build adds
# several GB of CUDA libraries.
RUN pip install --index-url https://download.pytorch.org/whl/cpu \
        "torch==${TORCH_VERSION}+cpu" "torchvision==${TORCHVISION_VERSION}+cpu"

# The other packages, pinned to the versions the processor was validated with
COPY docker/requirements-image.txt /tmp/requirements-image.txt
RUN pip install -r /tmp/requirements-image.txt \
 && rm /tmp/requirements-image.txt \
 && python -c "import torch; assert torch.__version__.endswith('+cpu'), 'not the CPU-only PyTorch: ' + torch.__version__"

# The one system library the Python packages need that the slim image leaves out: expat, for the
# GDAL inside rasterio (the packages bring everything else, or the image has it)
RUN apt-get update \
 && apt-get install -y --no-install-recommends libexpat1 \
 && rm -rf /var/lib/apt/lists/*

# The processor: code, colour AUX file and model weights, and its entry point
WORKDIR /opt/eaglet2_l1d
COPY process_l1c.py job_order.py pdgs_logger.py product_naming.py task_table.xml ./
COPY src/ src/
COPY aux/ aux/
COPY hf_cache/omnicloudmask/ hf_cache/omnicloudmask/
COPY docker/EAGLET2_L1D /usr/local/bin/EAGLET2_L1D
RUN python -m compileall -q /opt/eaglet2_l1d \
 && chmod -R a+rX /opt/eaglet2_l1d \
 && chmod 0755 /usr/local/bin/EAGLET2_L1D

# From here on as PDGS runs it: user 1100, group 600, no home folder. The check imports the
# processor and runs OmniCloudMask on a small blank image, offline, with the weights in the image.
USER 1100:600
RUN cd /opt/eaglet2_l1d && HF_HUB_OFFLINE=1 python -c "\
import sys; sys.path.insert(0, 'src'); \
import numpy as np, process_l1c; \
from iride_eaglet_processor.inference import run_omnicloud_on_array, model_weights; \
print('model weights:', ', '.join(p.name for p in model_weights())); \
run_omnicloud_on_array(np.zeros((3, 64, 64), 'float32'), 'cpu'); \
print('the processor imports and its models run offline, as user 1100')"

CMD ["EAGLET2_L1D"]
