"""Streaming CSV logger for fly movement data."""

import os

from sociSleep.config import RESULTS_DIR


class CSVLogger:
    """
    Append one row per second with fly positions and movement flags.

    Header columns adapt to the active plate design.
    """

    def __init__(self, camera_id, plate_design):
        os.makedirs(RESULTS_DIR, exist_ok=True)
        self.output_file = os.path.join(RESULTS_DIR, f"camera{camera_id}.csv")
        self._header_written = False
        self._plate_design = plate_design

    def set_design(self, plate_design):
        """Update design and reset header so the next write uses new columns."""
        self._plate_design = plate_design
        self._header_written = False

    def write_row(self, row):
        """Append a single CSV row (list of values)."""
        header = self._plate_design.csv_header()
        if not self._header_written:
            with open(self.output_file, "w", encoding="utf-8") as f:
                f.write(",".join(header) + "\n")
            self._header_written = True
        with open(self.output_file, "a", encoding="utf-8") as f:
            f.write(",".join(map(str, row)) + "\n")

    @property
    def path(self):
        return self.output_file
