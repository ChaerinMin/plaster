# Add file directory to import path
import os, sys
file_dir = os.path.dirname(os.path.abspath(__file__))
if file_dir not in sys.path:
    sys.path.insert(0, file_dir)
from timetree import timetree
import glob
from source import SensorMetadata

TIMETREE_FILENAME = 'plaster.timetree'

class TimeCache:
    def __init__(self, sensor_dir, force_recompute=False):
        self.sensor_dir = sensor_dir
        self.time_tree = None
        self.force_recompute = force_recompute
        self.init()

    def print_stats(self):
        # Print tree stats
        print("TimeTree statistics:")
        print(f" - Total nodes: {self.time_tree.getTotalNodes()}")
        print(f" - Leaf nodes: {self.time_tree.countLeafNodes()}")
        print(f" - Tree depth: {self.time_tree.getTreeDepth()}")

    def init(self):
        # First check if a pre-built timetree exists
        self.time_tree_path = os.path.join(self.sensor_dir, TIMETREE_FILENAME)
        if (os.path.exists(self.time_tree_path) and os.path.getsize(self.time_tree_path) > 0) and self.force_recompute == False:
            print(f"Loaded existing TimeTree from {TIMETREE_FILENAME}")
            # static method on the class
            self.time_tree = timetree.TimeTree.load(self.time_tree_path)
        else:
            # If no tree exists, concat all txt timestamp files and create a new one
            if self.force_recompute:
                print('Force recompute flag is set, creating a new TimeTree.')
            else:
                print('No existing time tree found (or it is empty), creating a new one.')
            metadata_files = sorted(glob.glob(os.path.join(self.sensor_dir, "*.txt")))
            self.time_tree = timetree.TimeTree()
            for metadata_path in metadata_files:
                self.time_tree.appendAVLTree(metadata_path)

            self.time_tree.save(self.time_tree_path)
            print(f"Created new TimeTree and saved to {TIMETREE_FILENAME}")

        self.print_stats()

import argparse
import os
import sys

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Initialize TimeCache for a sensor directory.")
    parser.add_argument("-d", "--sensor-dir", type=str, required=True, help="Path to the sensor directory")
    parser.add_argument("-t", "--timestamp", type=int, required=True, help="Timestamp to retrieve from the TimeTree")
    parser.add_argument("-mt", "--max_threshold", type=int, default=1000, help="Threshold for timestamp retrieval (in same units as timestamp)")
    # Force tree recomputation
    parser.add_argument("-f", "--force-recompute", action="store_true", help="Force recomputation of the TimeTree")

    args = parser.parse_args()
    time_cache_instance = TimeCache(args.sensor_dir, force_recompute=args.force_recompute)
    # Try finding a node
    node_details = time_cache_instance.time_tree.get(args.timestamp, threshold=args.max_threshold)
    print(f"Node details for timestamp {args.timestamp}: {node_details}")
