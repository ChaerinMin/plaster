try:
    # When built in-place, the compiled module is timetree/timetree*.so
    from timetree import timetree as timetree_ext
except Exception:
    # If installed as a top-level extension
    import timetree as timetree_ext
import os
import glob
from source import SensorMetadata


TIMETREE_FILENAME = 'plaster.timetree'

class TimeCache:
    def __init__(self, sensor_dir):
        self.sensor_dir = sensor_dir
        self.time_tree = None
        self.init()

    def init(self):
        # First check if a pre-built timetree exists
        self.time_tree_path = os.path.join(self.sensor_dir, TIMETREE_FILENAME)
        if os.path.exists(self.time_tree_path) and os.path.getsize(self.time_tree_path) > 0:
            print(f"Loaded existing TimeTree from {TIMETREE_FILENAME}")
            # static method on the class
            self.time_tree = timetree_ext.TimeTree.load(self.time_tree_path)
            return

        # If no tree exists, concat all txt timestamp files and create a new one
        print('No existing time tree found (or it is empty), creating a new one.')
        metadata_files = sorted(glob.glob(os.path.join(self.sensor_dir, "*.txt")))
        print('Found metadata files:', metadata_files)
        self.time_tree = timetree_ext.TimeTree()
        for metadata_path in metadata_files:
            self.time_tree.appendAVLTree(metadata_path)

        self.time_tree.save(self.time_tree_path)

import argparse

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Initialize TimeCache for a sensor directory.")
    parser.add_argument("-d", "--sensor-dir", type=str, required=True, help="Path to the sensor directory")

    args = parser.parse_args()
    time_cache_instance = TimeCache(args.sensor_dir)