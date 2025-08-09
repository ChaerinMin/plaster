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

1. Download the Anaconda installer script:

```bash
wget https://repo.anaconda.com/archive/Anaconda3-latest-Linux-x86_64.sh
```

2. Run the installer:

```bash
bash Anaconda3-latest-Linux-x86_64.sh
```

3. Follow the prompts to complete installation.

4. Activate Anaconda:

```bash
source ~/anaconda3/bin/activate
```

For more details, see the [official Anaconda documentation](https://docs.anaconda.com/anaconda/install/linux/).
```

5. Setup a conda environment for this project
```bash
# create and activate an isolated environment
conda create -n plaster python=3.11 -y
conda activate plaster

# tools needed
conda install -c conda-forge pybind11 -y
# If you don’t have a compiler:
conda install -c conda-forge gxx_linux-64 -y
```

6. Build the timetree extension

```
python setup.py build_ext --inplace
pip install -e .
```

### Running `plaster`

- First install `pybind`: `conda install -c conda-forge pybind`
