import os
import re

class Day:
    """
    A class representing a day of data captured by a BRICS rig.
    """
    def __init__(self, date):
        self.date = date
        self.sources = []

    def add_source(self, source):
        self.sources.append(source)

    def get_sources(self):
        return self.sources

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

        self.compute()

    def name(self):
        return self.name

    def path(self):
        return self.path

    def compute(self):
        """
        Computes the days of data captured by this source.
        """
        date_pattern = re.compile(r'^\d{4}-\d{2}-\d{2}$')
        if os.path.isdir(self.path):
            for entry in os.listdir(self.path):
                full_path = os.path.join(self.path, entry)
                if os.path.isdir(full_path) and date_pattern.match(entry):
                    day = Day(entry)
                    day.add_source(self)
                    self.days.append(day)
                    print(f"Found day: {day.get_date()}")
    
    