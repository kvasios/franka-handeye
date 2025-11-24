#!/usr/bin/env python3
"""
Franka Hand-Eye Calibration - Data Capture Script

Automatically captures calibration data at predefined joint poses.
Can be run standalone or imported as a module.

Usage:
    python scripts/capture_data.py --host 172.16.0.2 --output data/captured-data
"""

import sys
import time
import argparse
import yaml
import json
import shutil
from pathlib import Path

# Add parent directory to path for standalone execution
sys.path.insert(0, str(Path(__file__).parent.parent))

from franka_handeye import (
    RealSenseCamera,
    CharucoDetector,
    RobotController,
    NumpyEncoder,
)
import numpy as np
import cv2


def validate_joint_poses(joint_poses: list) -> bool:
    """
    Validate that joint poses are properly formatted.
    
    Parameters
    ----------
    joint_poses : list
        List of joint pose arrays.
    
    Returns
    -------
    bool
        True if valid.
    
    Raises
    ------
    ValueError
        If validation fails.
    """
    if not joint_poses:
        raise ValueError("No joint poses found")
    
    for i, pose in enumerate(joint_poses):
        if not isinstance(pose, list):
            raise ValueError(f"Pose {i} is not a list")
        if len(pose) != 7:
            raise ValueError(f"Pose {i} has {len(pose)} joints, expected 7")
        for j, val in enumerate(pose):
            if not isinstance(val, (int, float)):
                raise ValueError(f"Pose {i}, joint {j} is not a number")
    
    print(f"✓ Validated {len(joint_poses)} joint poses")
    return True


def capture_single_pose(
    robot: RobotController,
    camera: RealSenseCamera,
    detector: CharucoDetector,
    K: np.ndarray,
    D: np.ndarray,
    pose_idx: int,
    output_dir: Path
) -> bool:
    """
    Capture data for a single pose.
    
    Parameters
    ----------
    robot : RobotController
        Robot controller instance.
    camera : RealSenseCamera
        Camera instance.
    detector : CharucoDetector
        Charuco detector instance.
    K : np.ndarray
        Camera intrinsics matrix.
    D : np.ndarray
        Distortion coefficients.
    pose_idx : int
        Pose index for naming.
    output_dir : Path
        Output directory.
    
    Returns
    -------
    bool
        True if capture succeeded.
    """
    # Capture frame
    frame = camera.get_frame()
    if frame is None:
        print("WARNING: Failed to capture frame")
        return False
    
    # Get robot state using our RobotController
    robot_state = robot.get_state()
    q = robot_state['q']
    O_T_EE = robot_state['O_T_EE'].flatten().tolist()
    
    # Detect Charuco board
    valid, rvec, tvec, _ = detector.detect(frame, K, D)
    
    if valid:
        print("✓ Charuco board detected")
        cv2.drawFrameAxes(frame, K, D, rvec, tvec, 0.1)
    else:
        print("✗ WARNING: Charuco board NOT detected")
    
    # Save data
    pose_dir = output_dir / f"pose_{pose_idx:02d}"
    pose_dir.mkdir(exist_ok=True)
    cv2.imwrite(str(pose_dir / "image.png"), frame)
    
    data = {
        "joint_pose": q,
        "O_T_EE": O_T_EE,
        "camera_intrinsics": K.tolist() if hasattr(K, 'tolist') else K,
        "dist_coeffs": D.tolist() if hasattr(D, 'tolist') else D,
        "charuco_detected": valid,
    }
    
    if valid:
        data["T_cam_target_rvec"] = rvec.tolist() if hasattr(rvec, 'tolist') else rvec
        data["T_cam_target_tvec"] = tvec.tolist() if hasattr(tvec, 'tolist') else tvec
    
    with open(pose_dir / "data.json", 'w') as f:
        json.dump(data, f, indent=4, cls=NumpyEncoder)
    
    print(f"✓ Saved to {pose_dir.name}")
    return True


def capture_poses(
    robot: RobotController,
    camera: RealSenseCamera,
    detector: CharucoDetector,
    joint_poses: list,
    output_dir: Path,
    K: np.ndarray,
    D: np.ndarray,
    stabilization_delay: float = 0.5
) -> int:
    """
    Capture data at each joint pose automatically.
    
    Parameters
    ----------
    robot : RobotController
        Robot controller instance.
    camera : RealSenseCamera
        Camera instance.
    detector : CharucoDetector
        Charuco detector instance.
    joint_poses : list
        List of joint poses to visit.
    output_dir : Path
        Output directory.
    K : np.ndarray
        Camera intrinsics matrix.
    D : np.ndarray
        Distortion coefficients.
    stabilization_delay : float
        Delay after reaching pose before capture (seconds).
    
    Returns
    -------
    int
        Number of successful captures.
    """
    print(f"\nStarting automatic capture of {len(joint_poses)} poses...\n")
    
    # Clean output directory
    if output_dir.exists():
        shutil.rmtree(output_dir)
    output_dir.mkdir(parents=True)
    
    successful_captures = 0
    
    for i, pose in enumerate(joint_poses):
        print(f"--- Pose {i+1}/{len(joint_poses)} ---")
        try:
            # Move to joint pose using RobotController
            robot.move_joints(pose)
            time.sleep(stabilization_delay)
            
            if capture_single_pose(robot, camera, detector, K, D, i, output_dir):
                successful_captures += 1
            print()
            
        except Exception as e:
            print(f"✗ Error at pose {i}: {e}\n")
    
    print(f"Capture complete: {successful_captures}/{len(joint_poses)} successful")
    return successful_captures


def main():
    parser = argparse.ArgumentParser(
        description="Franka Hand-Eye Calibration Data Capture - Automatic Mode"
    )
    parser.add_argument("--host", default="172.16.0.2", help="Robot FCI IP address")
    parser.add_argument("--output", default="data/captured-data", help="Output directory")
    parser.add_argument("--poses", default="config/joint_poses.yaml", help="Joint poses config")
    parser.add_argument("--board", default="config/calibration_board_parameters.yaml", 
                        help="Charuco board parameters")
    args = parser.parse_args()

    output_dir = Path(args.output)

    print("=" * 60)
    print("Franka Hand-Eye Calibration - Data Capture")
    print("=" * 60)

    # Load and validate joint poses
    poses_path = Path(args.poses)
    if not poses_path.exists():
        print(f"ERROR: {poses_path} not found")
        return 1
    
    with open(poses_path, 'r') as f:
        data = yaml.safe_load(f)
    
    if not data or 'joint_poses' not in data:
        print("ERROR: joint_poses not found in config")
        return 1
    
    joint_poses = data['joint_poses']
    
    try:
        validate_joint_poses(joint_poses)
    except ValueError as e:
        print(f"ERROR: {e}")
        return 1

    # Initialize camera
    print("\nInitializing RealSense camera...")
    try:
        camera = RealSenseCamera()
        K, D = camera.get_intrinsics_matrix()
        print("✓ Camera initialized")
    except Exception as e:
        print(f"ERROR: Failed to initialize camera: {e}")
        return 1
    
    # Initialize Charuco detector
    print("Loading Charuco board parameters...")
    try:
        detector = CharucoDetector(args.board)
        print("✓ Charuco detector initialized")
    except Exception as e:
        print(f"ERROR: Failed to load Charuco parameters: {e}")
        camera.stop()
        return 1
    
    # Connect to robot using RobotController
    print(f"\nConnecting to robot at {args.host}...")
    try:
        robot = RobotController(args.host, dynamics_factor=0.2)
        print("✓ Robot connected")
    except Exception as e:
        print(f"ERROR: Failed to connect to robot: {e}")
        camera.stop()
        return 1

    # Capture data
    try:
        successful = capture_poses(robot, camera, detector, joint_poses, output_dir, K, D)
        
        if successful == len(joint_poses):
            print("\n" + "=" * 60)
            print("SUCCESS: All poses captured successfully!")
            print("=" * 60)
            print(f"\nData saved to: {output_dir}")
            print("\nNext step: Run calibration with:")
            print("  python scripts/compute_calibration.py")
            return 0
        else:
            print("\n" + "=" * 60)
            print(f"WARNING: Only {successful}/{len(joint_poses)} poses captured successfully")
            print("=" * 60)
            return 1
            
    except KeyboardInterrupt:
        print("\n\nCapture interrupted by user")
        return 1
    except Exception as e:
        print(f"\nERROR: {e}")
        return 1
    finally:
        camera.stop()
        print("\nCamera stopped")


if __name__ == "__main__":
    exit(main())
