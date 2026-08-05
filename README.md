# sociSleep

<p align="center">
  <img src="docs/dashboard.gif" width="900">
</p>

Identity-preserving tracking of individual Drosophila behavior in social environments.

**sociSleep** is an identity-preserving behavioral tracking system for quantifying sleep, locomotion, and social dynamics of individual *Drosophila* housed in social environments.

Unlike conventional Drosophila sleep systems that require physical isolation of animals, sociSleep continuously tracks individuals within shared arenas, allowing long-term measurement of sleep while preserving social interactions.

---

## Features

- Identity-preserving tracking for individual flies
- Quantification of sleep and locomotor activity
- Simultaneous tracking of social groups
- Automatic detection of merged flies
- Support for multiple experimental plate layouts
- Multi-camera acquisition (1–2 USB cameras)
- Independent output for each camera
- Interactive OpenCV graphical interface
- Automatic CSV data logging

---

## Experimental Designs

Currently supported plate configurations:

| Design | Left Arena | Right Arena |
|---------|------------|-------------|
| Solo vs Solo | 1 fly | 1 fly |
| Solo vs Group | 1 fly | 2 flies |
| Group vs Group | 2 flies | 2 flies |

Additional designs can be added by modifying `plate_design.py`.

---

## System Architecture

```
USB Camera(s)
        │
        ▼
 FlyTracker Thread(s)
        │
        ▼
 Identity Tracking
        │
        ▼
 Behavior Detection
        │
        ▼
 CSV Logger
```

Each USB camera is handled by an independent tracking thread.

The dashboard combines all active cameras into a single live display.

Example:

```
+----------------------+----------------------+
|     Camera 1         |      Camera 2        |
|                      |                      |
|   Arena A  Arena B   |   Arena A  Arena B   |
|                      |                      |
+----------------------+----------------------+
```

---

## Directory Structure

```
sociSleep/

├── main.py
├── fly_tracker.py
├── dashboard.py
├── camera_detector.py
├── arena_config.py
├── design_selector.py
├── csv_logger.py
├── plate_design.py
├── config.py
└── ...
```

---

## Requirements

- Python 3.11+
- OpenCV-python=4.5.0
- NumPy
- pygrabber
- SciPy
- Pandas

Install dependencies:

```bash
pip install -r requirements.txt
```

---

## Running sociSleep

```bash
camera_names.py
```

The program automatically detects connected USB cameras, and output camera names.


```bash
camera_detector.py
```

Modify camera names (just keyword) at Line 7.


```bash
run_sociSleep.py
```

Launch sociSleep tracker.

---

## Multi-camera Support

The current version supports one or two USB cameras simultaneously.

Each camera operates independently while sharing the same experimental plate design.

Output files are generated separately:

```
results_camera1/
results_camera2/
```

Each directory contains:

- Tracking CSV
- Experiment log
- Optional snapshots

---

## Plate Design Selection

Before tracking begins, select one of the available plate layouts:

- Solo vs Solo
- Solo vs Group
- Group vs Group

The graphical control panel provides:

- Large clickable buttons
- Current design highlight
- Arena summary
- READY / LOCKED status

Once tracking starts, the plate design becomes locked to prevent accidental changes.

---

## Output

Each experiment generates CSV files containing:

- Timestamp
- Fly coordinates
- Fly movement (0:immobile; 1: moving)
- Arena assignment

Example:

| Time | A_x | A_y | A_movement | B_x | B_y | B_movement |
|------|-----|-----|------------|-----|-----|------------|

---

## Identity Tracking

sociSleep preserves fly identity during long-term recordings using:

- Kalman filter + nearest assignment
- Blob area estimation
- Merge detection
- Arena-specific constraints

Identity switches are minimized during brief social interactions.

---

## GUI

The OpenCV interface consists of:

### Dashboard

- Live camera feeds
- Arena overlays
- Fly identities
- Tracking status

### Plate Design

- Clickable plate layout selection
- Arena information
- Tracking status

### Arena Calibration

Each camera has an independent calibration window for selecting arena positions before tracking.

---

## Typical Workflow

1. Connect USB camera(s).
2. Launch `run_sociSleep.py`.
3. Select the experimental plate design.
4. Calibrate arena locations and camera parameters.
5. Press **START**.
6. Monitor tracking in the live dashboard.
7. CSV files are automatically saved during acquisition.

---

## Citation

If you use sociSleep in your research, please cite:

> Binbin Wu, William Ja. Rencent social loss, not chronic isolation, reshapes sleep in Drosophila.
> Doi: https://doi.org/10.64898/2026.07.12.738068

---

## License

GPL-3.0 open-source License

---

## Contact

Binbin Wu

The Wertheim UF Scripps Institute

For questions, bug reports, or feature requests, please open a GitHub Issue.
