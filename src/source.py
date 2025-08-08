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
        self.stats = {
            "start_time": -1,
            "end_time": -1,
            "duration": 0,
            "avg_frame_rate": 0,
            "num_frames": 0
        }

    def compute_stats(self):
        """
        Computes statistics for the sequence.
        This method should handle the logic of computing statistics for the sequence.
        """
        # Compute duration, number of frames, etc.
        if not self.sensor_data:
            return None
        timestamps = []
        for metadata in self.sensor_data.values():
            timestamps.extend(metadata.timestamps)
        if not timestamps:
            return None
        start_time = min(timestamps)
        end_time = max(timestamps)
        duration = (end_time - start_time)*1e-9  # Convert to seconds
        num_frames = sum(len(metadata.timestamps) for metadata in self.sensor_data.values())
        self.stats = {
            "start_time": start_time,
            "end_time": end_time,
            "duration": duration,
            "avg_frame_rate": num_frames / duration if duration > 0 else 0,
            "num_frames": num_frames
        }

    def insert(self, name, sensor_metadata):
        """
        Inserts metadata into the sequence.
        This method should handle the logic of adding metadata to the sequence.
        """
        self.sensor_data[name] = sensor_metadata
        self.compute_stats()

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
        self.THRESHOLD = 2*1e9  # Threshold in nanoseconds for sequence continuity
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
        self.sensor_data_files = sorted(glob(os.path.join(self.path, '*.mp4')))
        if len(self.sensor_data_files) == 0:
            self.sensor_data_files = sorted(glob(os.path.join(self.path, '*.avi'))) # If it is not yet processed
        if len(self.sensor_data_files) == 0:
            self.sensor_data_files = sorted(glob(os.path.join(self.path, '*.flac'))) # If it is audio data
        if len(self.sensor_data_files) == 0:
            self.sensor_data_files = sorted(glob(os.path.join(self.path, '*.wav'))) # If it is unprocessed audio data            
        self.metadata_files = sorted(glob(os.path.join(self.path, '*.txt')))

        # Check if the number of video files and metadata files match
        assert len(self.sensor_data_files) == len(self.metadata_files), \
            f"Number of video files ({len(self.sensor_data_files)}) does not match number of metadata files ({len(self.metadata_files)}) in {self.path}"

        # Load metadata for each sensor
        self.metadata = []
        for metadata_file in self.metadata_files:
            metadata_name = os.path.basename(metadata_file).replace('.txt', '')
            sensor_metadata = SensorMetadata(metadata_name, metadata_file)
            if sensor_metadata.valid:
                self.metadata.append(sensor_metadata)

        # Next divide them into sequences
        sequence = Sequence()
        for ctr in range(len(self.metadata)):
            if ctr == 0:
                sequence.insert(self.metadata[ctr].name, self.metadata[ctr])
            else:
                # Check if the current metadata is contiguous with the previous one
                prev_metadata = self.metadata[ctr - 1]
                curr_metadata = self.metadata[ctr]
                assert len(prev_metadata.timestamps) > 0 and len(curr_metadata.timestamps) > 0, "Metadata timestamps cannot be empty."

                # print(f"Checking sequence continuity: {prev_metadata.timestamps[-1]} -> {curr_metadata.timestamps[0]}")
                if abs(curr_metadata.timestamps[0] - prev_metadata.timestamps[-1]) <= self.THRESHOLD:
                    sequence.insert(self.metadata[ctr].name, curr_metadata)
                else:
                    self.sequences.append(sequence)
                    sequence = Sequence()
                    sequence.insert(self.metadata[ctr].name, curr_metadata)

        # Add the last sequence if it exists
        if sequence.sensor_data:
            self.sequences.append(sequence)
        # If no sequences were found, print a warning
        if not self.sequences:
            print(f"No valid sequences found for sensor {self.name} on date {self.date}")
            return

        self.serialize(self.plaster_path)

    def serialize(self, plaster_path):
        """
        Serializes the sensor's data to a JSON format in the sensor's directory.
        """
        sequences_list = []
        for idx, seq in enumerate(self.sequences, start=1):
            sequences_list.append({
                "name": f"sequence{str(idx).zfill(6)}",
                "sensor_data": list(seq.sensor_data.keys()),
                "start_time": seq.stats["start_time"],
                "end_time": seq.stats["end_time"],
                "duration": seq.stats["duration"],
                "avg_frame_rate": seq.stats["avg_frame_rate"],
                "num_frames": seq.stats["num_frames"]
            })

        sensor_data = {
            "source": os.path.basename(os.path.normpath(self.source_path)),
            "day": self.date,
            "sensor": self.name,
            "sequences": sequences_list,
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
        self.multisequences = []
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

        self.sensors = [Sensor(self.source_path, self.date, sensor, self.force_reserialize) for sensor in sensor_names]
        # Identify multi-sensor overlapping sequences for this day
        try:
            self.multisequences = self.identify_multi_sequence(self.sensors)
        except Exception as e:
            # Do not fail day initialization if grouping fails; log and continue
            print(f"Failed to identify multisequences for {self.date}: {e}")
            self.multisequences = []

        self.serialize(self.plaster_path)

    def identify_multi_sequence(self, sensors):
        """
        Identify cross-sensor multisequences by overlapping time windows.

        Approach:
        - Collect all (sensor, sequence) intervals [start_time, end_time].
        - Build an undirected graph where an edge connects two sequences from
          different sensors if their intervals overlap at all.
        - Connected components of this graph are multisequences. This naturally
          ensures each sensor sequence belongs to at most one multisequence.

        Returns: list of multisequences. Each multisequence is a dict:
        {
            "start_time": <min_start_of_members>,
            "end_time": <max_end_of_members>,
            "duration": <seconds>,
            "members": [
                { "sensor": str, "sequence_index": int,
                  "start_time": int, "end_time": int, "duration": float }, ...
            ]
        }
        """

        if sensors is None:
            sensors = self.sensors

        # Collect intervals per sensor and a flat list of nodes
        nodes = []  # (node_id, sensor_name, seq_index, start, end)
        per_sensor = {}
        node_id = 0

        for sensor in sensors:
            seq_list = []
            for idx, seq in enumerate(sensor.sequences):
                st = seq.stats.get("start_time", -1)
                en = seq.stats.get("end_time", -1)
                if st is None or en is None or st < 0 or en < 0:
                    continue
                if st > en:
                    st, en = en, st
                seq_list.append((idx, seq, st, en))
                nodes.append((node_id, sensor.name, idx, st, en))
                node_id += 1
            # Sort sequences within each sensor by start time (helps determinism)
            seq_list.sort(key=lambda x: x[2])
            per_sensor[sensor.name] = seq_list

    # Do not early-exit; even a single sequence should form its own multisequence

        # Build adjacency list for overlaps across sensors
        adj = {nid: set() for nid, *_ in nodes}

        # Index nodes by sensor for efficient cross comparison
        by_sensor = {}
        for nid, sname, sidx, st, en in nodes:
            by_sensor.setdefault(sname, []).append((nid, st, en))
        # Sort each sensor's list by start for efficient merging
        for sname in by_sensor:
            by_sensor[sname].sort(key=lambda x: x[1])

        sensor_names = list(by_sensor.keys())
        # Compare only across different sensors using two-pointer sweep
        for i in range(len(sensor_names)):
            for j in range(i + 1, len(sensor_names)):
                A = by_sensor[sensor_names[i]]  # list of (nid, st, en)
                B = by_sensor[sensor_names[j]]
                pa = pb = 0
                while pa < len(A) and pb < len(B):
                    nid_a, sa, ea = A[pa]
                    nid_b, sb, eb = B[pb]
                    # Overlap if sa <= eb and sb <= ea
                    if sa <= eb and sb <= ea:
                        adj[nid_a].add(nid_b)
                        adj[nid_b].add(nid_a)
                    # Advance pointer with smaller end
                    if ea < eb:
                        pa += 1
                    else:
                        pb += 1

        # Find connected components (ignore isolated nodes since they don't overlap)
        visited = set()
        nid_to_meta = {nid: (sname, sidx, st, en) for nid, sname, sidx, st, en in nodes}
        components = []
        for nid in adj:
            if nid in visited:
                continue
            # BFS/DFS to collect component (includes isolated nodes)
            stack = [nid]
            comp = set()
            visited.add(nid)
            while stack:
                cur = stack.pop()
                comp.add(cur)
                for nei in adj[cur]:
                    if nei not in visited:
                        visited.add(nei)
                        stack.append(nei)
            if comp:
                components.append(comp)

        multisequences = []
        for comp in components:
            members = []
            comp_start = None
            comp_end = None
            for nid in sorted(comp):  # stable order by node id
                sname, sidx, st, en = nid_to_meta[nid]
                members.append({
                    "sensor": sname,
                    "sequence_index": sidx,
                    "start_time": st,
                    "end_time": en,
                    "duration": (en - st) * 1e-9
                })
                comp_start = st if comp_start is None else min(comp_start, st)
                comp_end = en if comp_end is None else max(comp_end, en)

            # Sort members by start time to satisfy "sort sequences inside each sensor"
            members.sort(key=lambda m: (m["start_time"], m["sensor"]))

            multisequences.append({
                "start_time": comp_start,
                "end_time": comp_end,
                "duration": (comp_end - comp_start) * 1e-9 if comp_end is not None and comp_start is not None else 0,
                "members": members
            })

        # Sort multisequences by start time for determinism
        multisequences.sort(key=lambda ms: ms["start_time"]) 

        # Assign deterministic names and put 'name' first in each dict for readability
        for i, ms in enumerate(multisequences, start=1):
            name = f"multisequence{str(i).zfill(6)}"
            multisequences[i - 1] = {
                "name": name,
                "start_time": ms.get("start_time"),
                "end_time": ms.get("end_time"),
                "duration": ms.get("duration"),
                "members": ms.get("members", [])
            }

        self.multisequences = multisequences
        return multisequences

    def serialize(self, plaster_path):
        """
        Serializes the day's data to a JSON format in the day's directory.
        """
        sensor_names = [sensor.name for sensor in self.sensors]
        json_obj = json.dumps({
            "source": os.path.basename(os.path.normpath(self.source_path)),
            "day": self.date,
            "multisequences": self.multisequences,
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
    