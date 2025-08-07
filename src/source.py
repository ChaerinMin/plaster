import os
import re
import json
from datetime import datetime

class Sequence:
    """
    A class representing a contiguous sequence of data captured by a sensor.
    """
    def __init__(self, source_path):
        self.source_path = source_path
        self.name = os.path.basename(os.path.normpath(source_path))
        self.init()

    def init(self):
        """
        Initializes the sequence by loading its data from the source path.
        """
        sequence_data_path = os.path.join(self.source_path, f"{self.name}.json")
        # if os.path.exists(sequence_data_path):
        #     with open(sequence_data_path, 'r') as json_file:
        #         try:
        #             self.data = json.load(json_file)
        #         except Exception as e:
        #             print(f"Error loading data for sequence {self.name}: {e}")

class Sensor:
    """
    A class representing a sensor that captures data.
    """
    def __init__(self, source_path, date):
        self.source_path = source_path
        self.date = date
        self.path = os.path.join(source_path, date)
        self.plaster_path = os.path.join(self.path, 'plaster.json')
        self.name = os.path.basename(os.path.normpath(self.path))
        self.init()

    def init(self):
        """
        Initializes the sensor by loading its data from the source path.
        """
        sensor_data_path = os.path.join(self.source_path, f"{self.name}.json")
        # if os.path.exists(sensor_data_path):
        #     with open(sensor_data_path, 'r') as json_file:
        #         try:
        #             self.data = json.load(json_file)
        #         except Exception as e:
        #             print(f"Error loading data for sensor {self.name}: {e}")

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

        if os.path.exists(self.plaster_path):
            with open(self.plaster_path, 'r') as json_file:
                try:
                    data = json.load(json_file)
                    json_sensors = data.get('sensors', [])
                except Exception:
                    json_sensors = []

            if set(json_sensors) == set(sensor_names):
                self.sensors = json_sensors
                return
            
        self.sensors = [Sensor(self.source_path, self.date) for sensor in sensor_names]
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
    