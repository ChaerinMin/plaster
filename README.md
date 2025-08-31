# plaster

`plaster` comes after [`BRICs`](https://github.com/brown-ivl/brics) and [`mortar`](https://github.com/brown-ivl/mortar). `plaster` does **plastering** which includes the following:

- For each category of BRICS data (e.g., BRICS Baby, BRICS Studio, BRICS Mini), it computes high-level statistics about how many days of data exists, and caches them to the recorded data directory.
- Within each day of capture, it computes statistics on durations, timestamps, and checks for and flags inconsistencies.
- Index and cache all the captured data as AVLTrees for time synced frame matching.
- Handles both old linux time-based captures, and the new GPS time-based captures.
- Colmap-based data intrinsic and extrinsic calibration and caching.
- *TBD* Color calibration

## Installation

`plaster` is written in Python and some C++. The goal is to have minimal dependencies and run off of most linux machines. It's designed to be constantly running in the background to perform `plastering` activities as data from `mortar` and BRICS systems stream into a server.

The typical workflow is to have a standalone machine (e.g., on AWS or GCP) run `plaster`.

### Install Anaconda

- Download the Anaconda installer script, or any other python environment you want:

```bash
curl -O https://repo.anaconda.com/archive/Anaconda3-2025.06-0-Linux-x86_64.sh
bash Anaconda3-2025.06-0-Linux-x86_64.sh
source ~/anaconda3/bin/activate
```

For more details, see the [official Anaconda documentation](https://docs.anaconda.com/anaconda/install/linux/).

- Setup a conda environment for this project
```bash
# create and activate an isolated environment
conda create -n plaster python=3.11 -y
conda activate plaster

# Install primer and timetree
pip install "git+ssh://git@github.com/brown-ivl/primer.git@main"
pip install "git+ssh://git@github.com/brown-ivl/timetree.git@main"

# Install colmap
pip install pycolmap
```
