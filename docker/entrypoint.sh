#!/bin/bash
# Activate the conda environment
source /opt/conda/etc/profile.d/conda.sh
conda activate aerialgym

# Start Xvfb on :99 as a virtual-display fallback.
# Isaac Gym's viewer requires an X11 display even when the primary display
# is forwarded from the host; Xvfb guarantees one is always available.
Xvfb :99 -screen 0 1920x1080x24 -nolisten tcp &

# Use the host-forwarded display if the socket exists; otherwise use :99.
DISP_NUM="${DISPLAY#:}"
DISP_NUM="${DISP_NUM%%.*}"
if [ -z "${DISPLAY}" ] || [ ! -S "/tmp/.X11-unix/X${DISP_NUM}" ]; then
    export DISPLAY=:99
fi

exec "$@"
