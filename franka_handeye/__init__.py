"""
Franka Hand-Eye Calibration Package

A modular toolkit for performing hand-eye calibration with a Franka robot
and Intel RealSense camera using ChArUco board detection.

Modules
-------
camera : RealSense camera interface
detector : ChArUco board detection
robot : Franka robot control
calibration : Hand-eye calibration computation
"""

from .camera import RealSenseCamera
from .detector import CharucoDetector
from .robot import RobotController
from .calibration import (
    NumpyEncoder,
    load_captured_data,
    compute_hand_eye_calibration,
    compute_consistency_metrics,
    save_calibration_result,
    load_calibration_result,
    compute_alignment_pose,
)

__version__ = "0.2.0"
__all__ = [
    # Camera
    "RealSenseCamera",
    # Detector
    "CharucoDetector",
    # Robot
    "RobotController",
    # Calibration
    "NumpyEncoder",
    "load_captured_data",
    "compute_hand_eye_calibration",
    "compute_consistency_metrics",
    "save_calibration_result",
    "load_calibration_result",
    "compute_alignment_pose",
]

