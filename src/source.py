import os
import re
import json
import datetime

class Day:
    """
    A class representing a day of data captured by a BRICS rig.
    """
    def __init__(self, date):
        self.date = date

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

        self.days = [Day(date) for date in dir_days]
        self.serialize()

    def serialize(self):
        """
        Serializes the source data to a JSON format in the top-level directory.
        """
        json_obj    = json.dumps({
            "name": self.name,
            "path": self.path,
            "days": [day.get_date() for day in self.days]
            ,
            "plastered_at": datetime.now().isoformat()
        }, indent=4)
        with open(self.plaster_path, 'w') as json_file:
            json_file.write(json_obj)
        print(f"Serialized Source to {self.plaster_path}")
    