#!/usr/bin/env python3
"""
Franka Hand-Eye Calibration - Verification Script

Verifies calibration by moving robot to align with charuco board.
Can be run standalone or imported as a module.

Usage:
    python scripts/verify_calibration.py --host 172.16.0.2 --offset 0.06
"""

import sys
import time
import argparse
import numpy as np
import cv2
from pathlib import Path
from scipy.spatial.transform import Rotation as R

# Add parent directory to path for standalone execution
sys.path.insert(0, str(Path(__file__).parent.parent))

from franka_handeye import (
    RealSenseCamera,
    CharucoDetector,
    RobotController,
    load_calibration_result,
    compute_alignment_pose,
)


def plot_verification(T_gripper_base, T_cam_gripper, T_target_cam):
    """
    Plot the frames after verification movement.
    
    Parameters
    ----------
    T_gripper_base : np.ndarray
        4x4 matrix (Gripper in Base).
    T_cam_gripper : np.ndarray
        4x4 matrix (Camera in Gripper - Calibration result).
    T_target_cam : np.ndarray
        4x4 matrix (Target in Camera - Detection result).
    """
    import matplotlib.pyplot as plt
    from mpl_toolkits.mplot3d import Axes3D
    
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection='3d')
    
    def plot_frame(T, label, scale=0.1):
        R_mat = T[:3, :3]
        t = T[:3, 3]
        ax.quiver(t[0], t[1], t[2], R_mat[0,0], R_mat[1,0], R_mat[2,0], length=scale, color='r')
        ax.quiver(t[0], t[1], t[2], R_mat[0,1], R_mat[1,1], R_mat[2,1], length=scale, color='g')
        ax.quiver(t[0], t[1], t[2], R_mat[0,2], R_mat[1,2], R_mat[2,2], length=scale, color='b')
        ax.text(t[0], t[1], t[2], label)

    # Plot Base Frame (0,0,0)
    plot_frame(np.eye(4), "Base", scale=0.2)
    
    # Plot Gripper Frame
    plot_frame(T_gripper_base, "Gripper")
    
    # Plot Camera Frame
    T_cam_base = T_gripper_base @ T_cam_gripper
    plot_frame(T_cam_base, "Camera")
    
    # Plot Target (Charuco) Frame
    T_target_base = T_cam_base @ T_target_cam
    plot_frame(T_target_base, "Charuco")
    
    # Set labels and auto-scale
    ax.set_xlabel('X (m)')
    ax.set_ylabel('Y (m)')
    ax.set_zlabel('Z (m)')
    
    # Collect all points to set axes limits
    points = np.vstack([
        np.zeros(3), 
        T_gripper_base[:3, 3], 
        T_cam_base[:3, 3], 
        T_target_base[:3, 3]
    ])
    
    center = np.mean(points, axis=0)
    radius = np.max(np.linalg.norm(points - center, axis=1)) + 0.1
    
    ax.set_xlim(center[0] - radius, center[0] + radius)
    ax.set_ylim(center[1] - radius, center[1] + radius)
    ax.set_zlim(center[2] - radius, center[2] + radius)
    
    plt.title("Verification: Frame Alignment")
    plt.legend(['X', 'Y', 'Z'])
    plt.show(block=True)


def move_to_board_position(
    robot: RobotController,
    T_gripper_base_current: np.ndarray,
    T_cam_gripper: np.ndarray,
    rvec: np.ndarray,
    tvec: np.ndarray,
    offset: float,
    target_point: list,
    position_name: str = "target"
):
    """
    Move robot to a position relative to the charuco board.
    
    Parameters
    ----------
    robot : RobotController
        Robot controller instance.
    T_gripper_base_current : np.ndarray
        Current gripper pose.
    T_cam_gripper : np.ndarray
        Camera-to-gripper calibration.
    rvec : np.ndarray
        Board rotation vector.
    tvec : np.ndarray
        Board translation vector.
    offset : float
        Distance offset from board.
    target_point : list
        Target point in board frame [x, y, z].
    position_name : str
        Name for logging.
    """
    T_gripper_base_desired = compute_alignment_pose(
        T_gripper_base_current, T_cam_gripper, rvec, tvec, offset, target_point
    )
    
    translation = T_gripper_base_desired[:3, 3].tolist()
    quaternion = R.from_matrix(T_gripper_base_desired[:3, :3]).as_quat().tolist()
    
    print(f"🤖 Moving to {position_name}...")
    robot.move_cartesian(translation, quaternion)
    print(f"✓ Reached {position_name}")


def run_verification(
    host: str = "172.16.0.2",
    calibration_path: str = "data/hand-eye-calibration-output/calibration_result.json",
    offset: float = 0.06,
    board_params_path: str = "config/calibration_board_parameters.yaml",
    show_plot: bool = True,
    tour_corners: bool = True
) -> bool:
    """
    Run the verification procedure.
    
    Parameters
    ----------
    host : str
        Robot FCI IP address.
    calibration_path : str
        Path to calibration result JSON.
    offset : float
        Distance offset from board in meters.
    board_params_path : str
        Path to board parameters YAML.
    show_plot : bool
        Whether to show preview plot.
    tour_corners : bool
        Whether to tour all corners.
    
    Returns
    -------
    bool
        True if verification completed successfully.
    """
    # Load calibration
    calib_path = Path(calibration_path)
    if not calib_path.exists():
        print(f"\n❌ ERROR: Calibration file not found: {calib_path}")
        print("   Run: python scripts/compute_calibration.py")
        return False
    
    print(f"\n✓ Loading calibration from {calib_path}")
    T_cam_gripper = load_calibration_result(calib_path)
    
    # Initialize camera
    print("✓ Initializing RealSense camera...")
    try:
        camera = RealSenseCamera()
        K, D = camera.get_intrinsics_matrix()
        print("✓ Camera initialized")
    except Exception as e:
        print(f"❌ ERROR: Failed to initialize camera: {e}")
        return False
    
    # Initialize Charuco detector
    print("✓ Loading Charuco board parameters...")
    try:
        detector = CharucoDetector(board_params_path)
        print("✓ Charuco detector initialized")
    except Exception as e:
        print(f"❌ ERROR: Failed to load Charuco parameters: {e}")
        camera.stop()
        return False
    
    # Connect to robot using RobotController
    print(f"✓ Connecting to robot at {host}...")
    try:
        robot = RobotController(host, dynamics_factor=0.1)
        print("✓ Robot connected")
        
        # Move to Home Pose
        print("🤖 Moving to home pose...")
        robot.go_home()
        print("✓ Reached home pose")
        
        # Home and close gripper
        print("🏠 Homing gripper...")
        robot.home_gripper()
        print("✓ Gripper homed")
        
        print("🤏 Closing gripper...")
        robot.close_gripper()
        print("✓ Gripper closed")
        
    except Exception as e:
        print(f"❌ ERROR: Failed to connect to robot: {e}")
        camera.stop()
        return False
    
    try:
        # Detect charuco board
        print("\n📷 Detecting charuco board from current position...")
        time.sleep(1.0)
        frame = camera.get_frame()
        if frame is None:
            print("❌ ERROR: Failed to capture frame")
            return False
        
        valid, rvec, tvec, _ = detector.detect(frame, K, D)
        
        if not valid:
            print("❌ ERROR: Charuco board not detected!")
            print("   Make sure the board is visible in the camera view.")
            return False
        
        print("✓ Charuco board detected")
        print(f"   Position in camera frame: {tvec.flatten()}")
        
        # Get current gripper pose using RobotController
        robot_state = robot.get_state()
        T_gripper_base_current = robot_state['O_T_EE']
        print("✓ Current gripper pose obtained")
        
        # Get board center
        center_point = detector.get_board_center()
        board_width, board_height = detector.board_dimensions
        
        print(f"\n🧮 Computing alignment pose at board center ({offset}m offset)...")
        print(f"   Board dimensions: {board_width:.3f}m x {board_height:.3f}m")
        print(f"   Center point in board frame: {center_point}")
        
        # Show preview plot
        if show_plot:
            T_gripper_base_desired = compute_alignment_pose(
                T_gripper_base_current, T_cam_gripper, rvec, tvec, offset, center_point
            )
            
            R_target_cam, _ = cv2.Rodrigues(rvec)
            T_target_cam = np.eye(4)
            T_target_cam[:3, :3] = R_target_cam
            T_target_cam[:3, 3] = tvec.flatten()
            
            T_cam_base = T_gripper_base_current @ T_cam_gripper
            T_target_base = T_cam_base @ T_target_cam
            
            T_cam_base_preview = T_gripper_base_desired @ T_cam_gripper
            T_target_cam_preview = np.linalg.inv(T_cam_base_preview) @ T_target_base
            
            print("\n📊 Displaying preview of center alignment...")
            print("   (Close the plot window to continue)")
            plot_verification(T_gripper_base_desired, T_cam_gripper, T_target_cam_preview)
        
        # Move to center
        move_to_board_position(
            robot, T_gripper_base_current, T_cam_gripper, rvec, tvec,
            offset, center_point, "board center"
        )
        
        print("\n✅ Center alignment complete!")
        
        # Tour corners if requested
        if tour_corners:
            print("\n🎯 Now visiting the 4 corners of the charuco board...")
            corners = detector.get_board_corners()
            corners.append(corners[0])  # Return to first corner
            corner_names = ["Top-Left", "Top-Right", "Bottom-Right", "Bottom-Left", "Top-Left (return)"]
            
            for i, (corner, corner_name) in enumerate(zip(corners, corner_names)):
                print(f"\n--- Corner {i+1}/5: {corner_name} ---")
                move_to_board_position(
                    robot, T_gripper_base_current, T_cam_gripper, rvec, tvec,
                    offset, corner, f"{corner_name} corner"
                )
                time.sleep(0.3)
            
            print("\n✅ Corner tour complete!")
        
        # Return to home
        print("\n🏠 Returning to home position...")
        robot.go_home()
        robot.home_gripper()
        print("✓ Returned to home position")
        
        return True
        
    except KeyboardInterrupt:
        print("\n\n❌ Aborted by user")
        return False
    except Exception as e:
        print(f"\n❌ ERROR: {e}")
        import traceback
        traceback.print_exc()
        return False
    finally:
        camera.stop()
        print("\n✓ Camera stopped")


def main():
    parser = argparse.ArgumentParser(
        description="Verify Hand-Eye Calibration by Aligning with Charuco Board"
    )
    parser.add_argument("--host", default="172.16.0.2", help="Robot FCI IP address")
    parser.add_argument("--calibration", default="data/hand-eye-calibration-output/calibration_result.json",
                        help="Path to calibration result")
    parser.add_argument("--offset", type=float, default=0.06,
                        help="Distance from board in meters (default: 0.06)")
    parser.add_argument("--no-plot", action="store_true", help="Skip preview plot")
    parser.add_argument("--no-tour", action="store_true", help="Skip corner tour")
    args = parser.parse_args()

    print("=" * 70)
    print("Franka Hand-Eye Calibration - Verification")
    print("=" * 70)
    
    print("\n⚠️  WARNING: This script will move the robot!")
    print("\n📋 Before proceeding:")
    print("   1. Ensure the charuco board is visible to the camera")
    print("   2. Ensure the workspace is clear and safe")
    print("   3. Have the emergency stop readily accessible")
    print(f"\nThe robot will align the end effector {args.offset}m above the board.")
    print("\n" + "=" * 70)
    
    try:
        input("\n➤ Press ENTER to continue or Ctrl+C to abort: ")
    except KeyboardInterrupt:
        print("\n\nAborted.")
        return 1
    
    success = run_verification(
        host=args.host,
        calibration_path=args.calibration,
        offset=args.offset,
        show_plot=not args.no_plot,
        tour_corners=not args.no_tour
    )
    
    if success:
        print("\n" + "=" * 70)
        print("🎉 Verification complete!")
        print("=" * 70)
        return 0
    else:
        return 1


if __name__ == "__main__":
    exit(main())
