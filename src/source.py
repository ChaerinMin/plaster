import os
import re
import json
from datetime import datetime
from glob import glob

class SensorMetadata:
    """
    A class representing for a sensor.
    This is usually recorded in a .txt file alongside the sensor's data files.
    """
    def __init__(self, name, metadata_file):
        self.name = name
        self.metadata_file = metadata_file
        self.timestamps = []
        self.frame_nums = []
        self.valid = False
        self.init()

    def init(self):
        """
        Initializes the metadata by loading its data from the metadata file.
        """
        # print(f"Initializing SensorMetadata: {self.name} from {self.metadata_file}")
        if os.path.exists(self.metadata_file):
            with open(self.metadata_file, 'r') as file:
                # Format is a txt file: frame_<TIMESTAMP>[_<FRAMENUM>]. The last bit within [] is optional.
                for line in file:
                    line = line.strip()
                    if line:
                        match = re.match(r'frame_(\d+)(?:_(\d+))?', line)
                        if match:
                            timestamp = int(match.group(1))
                            frame_num = int(match.group(2)) if match.group(2) else 0
                            self.timestamps.append(timestamp)
                            self.frame_nums.append(frame_num)
                        else:
                            print(f"Line '{line}' in {self.metadata_file} does not match expected format.")
                # If no timestamps were found, mark as invalid
                if self.timestamps:
                    self.valid = True
                else:
                    self.valid = False
                    print(f"No valid timestamps found in {self.metadata_file}.")
        else:
            print(f"Metadata file {self.metadata_file} does not exist.")

class Sequence:
    """
    A class representing a contiguous sequence of data captured by a sensor.
    """
    def __init__(self):
        self.sensor_data = dict()

    def insert(self, name, sensor_metadata):
        """
        Inserts metadata into the sequence.
        This method should handle the logic of adding metadata to the sequence.
        """
        self.sensor_data[name] = sensor_metadata

class Sensor:
    """
    A class representing a sensor that captures data.
    """
    def __init__(self, source_path, date, sensor_name, force_reserialize=False):
        self.source_path = source_path
        self.date = date
        self.name = sensor_name
        self.path = os.path.join(self.source_path, self.date, self.name)
        self.plaster_path = os.path.join(self.path, 'plaster.json')
        self.force_reserialize = force_reserialize
        self.sequences = []
        self.THRESHOLD = 2000  # Threshold in milliseconds for sequence continuity
        self.init()

    def init(self):
        """
        Initializes the sensor by loading its data from the source path.
        We do the following:
        - Identify sequences and write the sequence info into the plaster.json file
        - Check for consistency of videos/audio and txt file
        - Identify sequences from all the video files
        """
        # First, get all the files
        self.video_files = sorted(glob(os.path.join(self.path, '*.mp4')))
        if len(self.video_files) == 0:
            self.video_files = sorted(glob(os.path.join(self.path, '*.avi'))) # If it is not yet processed
        self.metadata_files = sorted(glob(os.path.join(self.path, '*.txt')))

        # Check if the number of video files and metadata files match
        assert len(self.video_files) == len(self.metadata_files), \
            f"Number of video files ({len(self.video_files)}) does not match number of metadata files ({len(self.metadata_files)}) in {self.path}"

        # Load metadata for each sensor
        self.metadata = []
        for metadata_file in self.metadata_files:
            metadata_name = os.path.basename(metadata_file).replace('.txt', '')
            sensor_metadata = SensorMetadata(metadata_name, metadata_file)
            if sensor_metadata.valid:
                self.metadata.append(sensor_metadata)

        # Next divide them into sequences
        seq_ctr = 0
        for ctr in range(len(self.metadata)):
            if ctr == 0:
                sequence = Sequence()
                sequence_name = "sequence" + str(seq_ctr).zfill(6)
                sequence.insert(sequence_name, self.metadata[ctr])
            else:
                # Check if the current metadata is contiguous with the previous one
                prev_metadata = self.metadata[ctr - 1]
                curr_metadata = self.metadata[ctr]
                assert len(prev_metadata.timestamps) > 0 and len(curr_metadata.timestamps) > 0, "Metadata timestamps cannot be empty."

                print(f"Checking sequence continuity: {prev_metadata.timestamps[-1]} -> {curr_metadata.timestamps[0]}")
                if abs(curr_metadata.timestamps[0] - prev_metadata.timestamps[-1]) <= self.THRESHOLD:
                    sequence.insert(curr_metadata)
                else:
                    self.sequences.append(sequence)
                    sequence = Sequence()
                    seq_ctr += 1
                    sequence_name = "sequence" + str(ctr).zfill(6)
                    sequence.insert(sequence_name, curr_metadata)

            self.serialize(self.plaster_path)

    def serialize(self, plaster_path):
        """
        Serializes the sensor's data to a JSON format in the sensor's directory.
        """
        sensor_data = {
            "name": self.name,
            "date": self.date,
            "sensor": self.name,
            "sequences": {
                f"sequence{str(idx).zfill(6)}": list(seq.sensor_data.keys())
                for idx, seq in enumerate(self.sequences)
            },
            "plaster_timestamp": datetime.now().isoformat()
        }
        with open(plaster_path, 'w') as json_file:
            json.dump(sensor_data, json_file, indent=4)

class Day:
    """
    A class representing a day of data captured by a BRICS rig.
    """
    def __init__(self, date, source_path, force_reserialize=False):
        self.source_path = source_path
        self.date = date
        self.path = os.path.join(source_path, date)
        self.plaster_path = os.path.join(self.path, 'plaster.json')
        self.sensors = []
        self.force_reserialize = force_reserialize
        self.init()

    def init(self):
        """
        Initializes the day by listing all the sensors that were capturing data on that day.
        """
        sensor_pattern = re.compile(r'^[A-Za-z0-9_\-]+$')
        if not os.path.exists(self.path):
            self.sensors = []
            print(f"Day directory {self.path} does not exist.")
            return
        
        sensor_names = [entry for entry in os.listdir(self.path)
                        if os.path.isdir(os.path.join(self.path, entry)) and sensor_pattern.match(entry)]

        if os.path.exists(self.plaster_path) and not self.force_reserialize:
            with open(self.plaster_path, 'r') as json_file:
                try:
                    data = json.load(json_file)
                    json_sensors = data.get('sensors', [])
                except Exception:
                    json_sensors = []

            if set(json_sensors) == set(sensor_names):
                self.sensors = json_sensors
                return

        self.sensors = [Sensor(self.source_path, self.date, sensor, self.force_reserialize) for sensor in sensor_names]
        self.serialize(self.plaster_path)

    def serialize(self, plaster_path):
        """Serializes the day's data to a JSON format in the day's directory.
        """
        sensor_names = [sensor.name for sensor in self.sensors]
        json_obj = json.dumps({
            "date": self.date,
            "sensors": sensor_names,
            "plaster_timestamp": datetime.now().isoformat()
        }, indent=4)
        with open(plaster_path, 'w') as json_file:
            json_file.write(json_obj)

class Source:
    """
    A class representing the source of data, usually captured by a single BRICS rig (e.g., BRICS Mini, BRICS Studio).
    """
    def __init__(self, path, force_reserialize=False):
        self.name = os.path.basename(os.path.normpath(path))
        self.path = path
        print(f"Initializing Source: {self.name} at {self.path}")
        self.days = []
        self.plaster_path = os.path.join(self.path, 'plaster.json')
        self.force_reserialize = force_reserialize
        if self.force_reserialize:
            print("Forcing reserialization of the source.")
        self.init()

    def init(self):
        """
        Initializes the source by computing the days of data captured.
        """
        date_pattern = re.compile(r'^\d{4}-\d{2}-\d{2}$')
        dir_days = [entry for entry in os.listdir(self.path)
                    if os.path.isdir(os.path.join(self.path, entry)) and date_pattern.match(entry)]

        if os.path.exists(self.plaster_path) and not self.force_reserialize:
            with open(self.plaster_path, 'r') as json_file:
                try:
                    data = json.load(json_file)
                    json_days = data.get('days', [])
                except Exception:
                    json_days = []

            if set(json_days) == set(dir_days):
                self.days = [Day(date, self.path, self.force_reserialize) for date in json_days]
                return
            # else, update JSON below

        self.days = [Day(date, self.path, self.force_reserialize) for date in dir_days]
        self.serialize(self.plaster_path)

        print(f"All done.")

    def serialize(self, plaster_path):
        """
        Serializes the source data to a JSON format in the top-level directory.
        """
        json_obj = json.dumps({
            "source": self.name,
            "days": [day.date for day in self.days],
            "plaster_timestamp": datetime.now().isoformat()
        }, indent=4)
        with open(plaster_path, 'w') as json_file:
            json_file.write(json_obj)
    