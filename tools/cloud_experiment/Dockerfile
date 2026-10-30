# EXPERIMENT image, not a release: the delivered 01.01 with cloud_patch.py applied to its colour step (clouds kept
# neutral). Tagged ipf-ime-msi-l1d:01.01-cloudtest; it never replaces 01.01 and is never pushed or delivered.
FROM ipf-ime-msi-l1d:01.01
COPY --chown=1100:600 cloud_patch.py /opt/ipf-ime-msi-l1d/cloud_patch.py
RUN python /opt/ipf-ime-msi-l1d/cloud_patch.py /opt/ipf-ime-msi-l1d/iride_eaglet_processor_v2-main \
 && rm /opt/ipf-ime-msi-l1d/cloud_patch.py \
 && python -m compileall -q /opt/ipf-ime-msi-l1d/iride_eaglet_processor_v2-main \
 && cd /opt/ipf-ime-msi-l1d/iride_eaglet_processor_v2-main \
 && HF_HUB_OFFLINE=1 python -c "import sys; sys.path.insert(0, 'src'); import process_l1c; from iride_eaglet_processor import colour_correction as c; print('EXPERIMENT image: clouds kept neutral', c.CLOUD_NEUTRAL, c.BRIGHT_ONE_FACTOR)"
