FROM ubuntu:20.04

SHELL ["/bin/bash", "-lc"]

# Install python3.8, conda and pip3
RUN apt-get update && apt-get install -y \
    wget \
    bzip2 \
    ca-certificates \
    python3 \
    python3-pip \
    git \
    # Vulkan loader + Mesa OpenGL — required for Isaac Gym's rendering backend
    libvulkan1 \
    vulkan-utils \
    libgl1-mesa-glx \
    libegl1-mesa \
    libgles2-mesa \
    # Runtime libs used by Pillow/matplotlib in camera and plotting examples
    libtiff5 \
    # X11 client lib so the viewer can connect to a forwarded display
    libx11-6 \
    xauth \
    # Virtual framebuffer — lets Isaac Gym open a window even without a real display
    xvfb \
    && rm -rf /var/lib/apt/lists/*

RUN wget --quiet https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh -O /tmp/miniconda.sh && \
    /bin/bash /tmp/miniconda.sh -b -p /opt/conda && \
    rm /tmp/miniconda.sh && \
    /opt/conda/bin/conda clean -afy

# Set PATH for conda
ENV PATH="/opt/conda/bin:${PATH}"

# Accept Anaconda channel Terms of Service for non-interactive builds.
RUN conda tos accept --override-channels --channel https://repo.anaconda.com/pkgs/main && \
    conda tos accept --override-channels --channel https://repo.anaconda.com/pkgs/r

# Create a conda environment and install dependencies
RUN conda create -n aerialgym python=3.8
RUN conda run -n aerialgym conda install -y -c pytorch -c conda-forge -c defaults -c nvidia \
    pytorch==1.13.1 \
    torchvision==0.14.1 \
    torchaudio==0.13.1 \
    pytorch-cuda=11.7

RUN conda run -n aerialgym conda install -y -c pytorch -c conda-forge -c defaults -c nvidia \
        pyyaml==6.0 \
        tensorboard==2.13.0

RUN conda run -n aerialgym conda install -y -c fvcore -c iopath -c conda-forge \
        fvcore \
        iopath \
        urdfpy

RUN conda run -n aerialgym pip3 install \
    warp-lang==1.0.0 \
    trimesh \
    gymnasium \
    rl-games \
    sample-factory

# Install a PyTorch3D wheel matching python3.8 + CUDA 11.7 + PyTorch 1.13.1.
RUN conda run -n aerialgym pip3 install \
    "pytorch3d==0.7.3" \
    -f https://dl.fbaipublicfiles.com/pytorch3d/packaging/wheels/py38_cu117_pyt1131/download.html

# Download and unpack the isaac gym source code (https://developer.nvidia.com/isaac-gym-preview-4)
RUN wget --quiet https://developer.nvidia.com/isaac-gym-preview-4 -O /tmp/isaacgym.tar.gz && \
    tar -xzf /tmp/isaacgym.tar.gz -C /opt/ && \
    rm /tmp/isaacgym.tar.gz

# Set the working directory
WORKDIR /opt/isaacgym

# Install the isaac gym python bindings
RUN cd python && \
    conda run -n aerialgym pip3 install -e .

# Isaac Gym needs the conda env's lib directory on LD_LIBRARY_PATH so it can
# find libcudart, libnvrtc, and other CUDA runtime libraries at startup.
ENV LD_LIBRARY_PATH="/opt/conda/envs/aerialgym/lib:${LD_LIBRARY_PATH}"

# Mirror the change in .bashrc for interactive shells.
RUN echo "export LD_LIBRARY_PATH=/opt/conda/envs/aerialgym/lib:\$LD_LIBRARY_PATH" >> ~/.bashrc

# Copy the aerialgym code to the container
COPY . /opt/aerialgym

# Set the working directory to the aerialgym code
WORKDIR /opt/aerialgym

# Install the aerialgym package
RUN conda run -n aerialgym pip3 install --no-deps -e .

# Isaac Gym's torch_utils.py uses np.float which was removed in NumPy 1.24.
# Pin numpy to the last compatible version as the very last pip step so nothing
# else can silently re-upgrade it.
RUN conda run -n aerialgym pip3 install "numpy==1.23.5"
RUN conda run -n aerialgym pip3 install "matplotlib"

# Setup the bashrc to activate the conda environment
RUN echo "source /opt/conda/etc/profile.d/conda.sh && conda activate aerialgym" >> ~/.bashrc

# Copy the entrypoint script that starts Xvfb and activates the conda env.
COPY docker/entrypoint.sh /entrypoint.sh
RUN chmod +x /entrypoint.sh

# Set the entrypoint to our startup script; default command is an interactive shell.
ENTRYPOINT ["/entrypoint.sh"]
CMD ["/bin/bash"]