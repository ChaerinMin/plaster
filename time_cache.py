import timetree
import os
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
        if os.path.exists(self.time_tree_path):
            print(f"Loaded existing TimeTree from {TIMETREE_FILENAME}")
            self.time_tree = timetree.load(self.time_tree_path)
            return

        # If no tree exists, concat all txt timestamp files and create a new one
        metadata_files = [f for f in os.listdir(self.sensor_dir) if f.endswith("*.txt")]
        self.time_tree = timetree.TimeTree()
        for metadata_file in metadata_files:
            self.time_tree.appendAVLTree(os.path.join(self.sensor_dir, metadata_file))

        self.time_tree.save(self.time_tree_path)

import argparse

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Initialize TimeCache for a sensor directory.")
    parser.add_argument("-d", "--sensor-dir", type=str, required=True, help="Path to the sensor directory")

    args = parser.parse_args()
    time_cache_instance = TimeCache(args.sensor_dir)
    print("TimeCache initialized successfully.")