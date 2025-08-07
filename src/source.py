import os
import re
import json
from datetime import datetime

class Sensor:
    """
    A class representing a sensor that captures data.
    """
    def __init__(self, name, source_path):
        self.name = name
        self.source_path = source_path
        self.data = []
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
    def __init__(self, date, source_path):
        self.source_path = source_path
        self.date = date
        self.sensors = []
        self.init()

    def init(self):
        """
        Initializes the day by listing all the sensors that were capturing data on that day.
        """
        sensor_pattern = re.compile(r'^[A-Za-z0-9_\-]+$')
        sensor_dir = os.path.join(self.source_path, self.date)
        if not os.path.exists(sensor_dir):
            self.sensors = []
            return

        self.sensors = [entry for entry in os.listdir(sensor_dir)
                        if os.path.isdir(os.path.join(sensor_dir, entry)) and sensor_pattern.match(entry)]

        plaster_path = os.path.join(sensor_dir, 'plaster.json')
        if os.path.exists(plaster_path):
            with open(plaster_path, 'r') as json_file:
                try:
                    data = json.load(json_file)
                    json_sensors = data.get('sensors', [])
                except Exception:
                    json_sensors = []

            if set(json_sensors) == set(self.sensors):
                self.sensors = json_sensors
                return

        json_obj = json.dumps({
            "date": self.date,
            "sensors": self.sensors,
            "plaster_timestamp": datetime.now().isoformat()
        }, indent=4)
        with open(plaster_path, 'w') as json_file:
            json_file.write(json_obj)

    def get_date(self):
        return self.date

class Source:
    """
    A class representing the source of data, usually captured by a single BRICS rig (e.g., BRICS Mini, BRICS Studio).
    """
    def __init__(self, path):
        self.name = os.path.basename(os.path.normpath(path))
        self.path = path
        print(f"Initializing Source: {self.name} at {self.path}")
        self.days = []
        self.plaster_path = os.path.join(self.path, 'plaster.json')
        self.init()

    def init(self):
        """
        Initializes the source by computing the days of data captured.
        """
        date_pattern = re.compile(r'^\d{4}-\d{2}-\d{2}$')
        dir_days = [entry for entry in os.listdir(self.path)
                    if os.path.isdir(os.path.join(self.path, entry)) and date_pattern.match(entry)]

        if os.path.exists(self.plaster_path):
            with open(self.plaster_path, 'r') as json_file:
                try:
                    data = json.load(json_file)
                    json_days = data.get('days', [])
                except Exception:
                    json_days = []

            if set(json_days) == set(dir_days):
                self.days = [Day(date) for date in json_days]
                return
            # else, update JSON below

        self.days = [Day(date, self.path) for date in dir_days]
        self.serialize()

    def serialize(self):
        """
        Serializes the source data to a JSON format in the top-level directory.
        """
        json_obj    = json.dumps({
            "name": self.name,
            "days": [day.get_date() for day in self.days],
            "plaster_timestamp": datetime.now().isoformat()
        }, indent=4)
        with open(self.plaster_path, 'w') as json_file:
            json_file.write(json_obj)
        print(f"Serialized Source to {self.plaster_path}")
    