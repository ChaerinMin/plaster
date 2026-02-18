# Plaster

Plaster assumes BRICS data with timestamp files.

Plaster provides the following:

- `plaster.json` file. This file includes sync information.
- Undistortion information by `pycolmap`
- Calibration either by `pycolmap` (stage 1: OPENCV, stage 2: PINHOLE) or VGGT (stage 3: PINHOLE).

## Installation

```bash
pip install "git+ssh://git@github.com/brown-ivl/primer.git@main"
pip install "git+ssh://git@github.com/brown-ivl/timetree.git@main"
pip uninstall pyceres
pip install pycolmap==3.12.4  # Must be 3.12.4
pip install pyceres
# If CCV,
module load ffmpeg
```

### OPTIONAL: Use VGGT for spatial calibration

VGGT might sometimes work better for spatial calibration in settings where COLMAP fails. Plaster Space supports using VGGT for a "stage3" calibration. In order to use this feature, follow the installation [instructions for VGGT here](https://github.com/facebookresearch/vggt). Specifically:

```
cd ~/code/vggt
pip install -r requirements.txt
pip install -r requirements_demo.txt
pip install -e .
```

Note that VGGT Torch dependencies are not compatible with `plaster` and `primer`. But it should not matter. `plaster` is designed to be compatible with `<torch-2.0` but will also work with newer versions.


## Usage

```bash
python plaster_time.py -s /oscar/data/ssrinath/public/brics-mini
python plaster_space.py -s /oscar/data/ssrinath/brics/non-pii/brics-mini -o "/oscar/data/ssrinath/public/brics-mini" -d yyyy-mm-dd
```