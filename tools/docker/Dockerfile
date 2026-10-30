# eaglet-lut-tools: the processor's image plus what the LUT tools need besides the processor's packages.
#     sudo docker build --network host -t eaglet-lut-tools tools/docker
# (--network host where Docker's own network cannot look up names, as on our server)
FROM ipf-ime-msi-l1d:01.01
RUN pip install --no-cache-dir opencv-python-headless requests matplotlib
