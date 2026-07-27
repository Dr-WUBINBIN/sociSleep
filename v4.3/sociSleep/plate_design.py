"""Plate layout definitions for sociSleep experiments."""

from dataclasses import dataclass


@dataclass(frozen=True)
class PlateDesign:
    """
    Describes how many flies occupy each arena and which labels they receive.

    Labels are assigned left-to-right within each arena (largest blob first).
    """

    id: str
    name: str
    left_count: int
    right_count: int

    @property
    def left_labels(self):
        """Fly identity labels in the left arena."""
        if self.left_count == 1:
            return ["A"]
        if self.left_count == 2:
            return ["A", "B"]
        raise ValueError(f"Unsupported left_count: {self.left_count}")

    @property
    def right_labels(self):
        """Fly identity labels in the right arena."""
        if self.right_count == 1:
            return ["B"] if self.left_count == 1 else ["C"]
        if self.right_count == 2:
            return ["B", "C"] if self.left_count == 1 else ["C", "D"]
        raise ValueError(f"Unsupported right_count: {self.right_count}")

    @property
    def all_labels(self):
        """All active fly labels for this design."""
        return self.left_labels + self.right_labels

    @property
    def supports_merge_detection(self):
        """True when an arena can hold multiple flies (merged-blob logic applies)."""
        return self.left_count > 1 or self.right_count > 1

    def label_side(self, label):
        """Return 'left' or 'right' for a fly label."""
        if label in self.left_labels:
            return "left"
        if label in self.right_labels:
            return "right"
        raise KeyError(label)

    def csv_header(self):
        """Build CSV column names for this design."""
        cols = ["Time"]
        for label in self.all_labels:
            cols.extend([f"{label}_x", f"{label}_y", f"{label}_movement"])
        return cols


SOLO_VS_SOLO = PlateDesign(
    id="solo_vs_solo",
    name="Solo vs Solo",
    left_count=1,
    right_count=1,
)

SOLO_VS_GROUP = PlateDesign(
    id="solo_vs_group",
    name="Solo vs Group",
    left_count=1,
    right_count=2,
)

GROUP_VS_GROUP = PlateDesign(
    id="group_vs_group",
    name="Group vs Group",
    left_count=2,
    right_count=2,
)

ALL_DESIGNS = [SOLO_VS_SOLO, SOLO_VS_GROUP, GROUP_VS_GROUP]

DEFAULT_DESIGN = SOLO_VS_GROUP

# Marker colors per label (BGR)
LABEL_COLORS = {
    "A": (0, 0, 255),      # red
    "B": (255, 0, 0),      # blue
    "C": (0, 255, 0),      # green
    "D": (0, 255, 255),    # yellow
}
