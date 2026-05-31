## Docker Quick Start

A self-contained Docker environment is provided under [`docker/`](/docker/) for running the simulator without a manual local installation.

### Prerequisites

- [Docker](https://docs.docker.com/get-docker/) with the [NVIDIA Container Toolkit](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/install-guide.html) (`nvidia-docker2`)
- An NVIDIA GPU with a driver that supports CUDA 11.7+

### Build the image

```bash
cd docker/
docker compose build
```

This builds the `aerialgym:latest` image, which bundles:
- Miniconda + Python 3.8
- PyTorch 1.13.1 (CUDA 11.7), Warp 1.0.0, PyTorch3D 0.7.3
- NVIDIA Isaac Gym Preview 4 (downloaded automatically)
- All Python and system dependencies needed to run the examples

### Run the container

```bash
cd docker/
docker compose up -d        # start in the background
docker exec -it aerialgym /bin/bash
```

The source tree is bind-mounted at `/opt/aerialgym` inside the container, so edits you make on the host are immediately reflected inside and vice-versa.

### X11 display forwarding (viewer window)

The compose file forwards the host X11 socket and Xauthority cookie automatically.  
On hosts that use Wayland/Xwayland, make sure `$DISPLAY` and `$XAUTHORITY` are set before calling `docker compose up`:

```bash
echo $DISPLAY        # should be :0, :1, or similar
echo $XAUTHORITY     # should point to a valid file
```

If no graphical display is available (e.g. a headless server), pass `--headless True` to any example:

```bash
python3 aerial_gym/examples/position_control_example.py --headless True
```

The container entrypoint automatically starts a virtual framebuffer (`Xvfb :99`) as a fallback when the host display socket is absent.

### Run the examples

Inside the container, all examples under `aerial_gym/examples/` can be run directly:

```bash
# Headless — no viewer window
python3 aerial_gym/examples/position_control_example.py --headless True

# With viewer (requires X11 forwarding from the host)
python3 aerial_gym/examples/position_control_example.py --headless False

# Camera stream with depth + segmentation images
python3 aerial_gym/examples/save_camera_stream.py --headless True

# RL environment loop
python3 aerial_gym/examples/rl_env_example.py --headless True
```

### Stop the container

```bash
cd docker/
docker compose down
```
