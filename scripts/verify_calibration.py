import os
import argparse
import yaml
import time
import numpy as np
import cv2
import pyrealsense2 as rs
import json
from pathlib import Path

# Set default server IP
os.environ.setdefault("FRANKY_SERVER_IP", "192.168.122.100")

from scipy.spatial.transform import Rotation as R
from franky import Robot, CartesianMotion, Affine, JointMotion, Gripper, ReferenceType
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

class RealSenseCamera:
    def __init__(self):
        self.pipeline = rs.pipeline()
        self.config = rs.config()
        self.config.enable_stream(rs.stream.color, 1280, 720, rs.format.bgr8, 30)
        self.profile = self.pipeline.start(self.config)
        self.color_stream = self.profile.get_stream(rs.stream.color)
        self.intrinsics = self.color_stream.as_video_stream_profile().get_intrinsics()

    def get_frame(self):
        frames = self.pipeline.wait_for_frames()
        color_frame = frames.get_color_frame()
        if not color_frame:
            return None
        return np.asanyarray(color_frame.get_data())

    def get_intrinsics_matrix(self):
        K = np.array([
            [self.intrinsics.fx, 0, self.intrinsics.ppx],
            [0, self.intrinsics.fy, self.intrinsics.ppy],
            [0, 0, 1]
        ])
        dist_coeffs = np.array(self.intrinsics.coeffs)
        return K, dist_coeffs

    def stop(self):
        self.pipeline.stop()

class CharucoDetector:
    def __init__(self, params_path):
        with open(params_path, 'r') as f:
            params = yaml.safe_load(f)
        self.board_size = params['board_size']
        self.square_length = params['square_length']
        self.board = cv2.aruco.CharucoBoard(
            (params['board_size'][0], params['board_size'][1]),
            params['square_length'],
            params['marker_length'],
            cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_6X6_250)
        )
        self.dictionary = self.board.getDictionary()
        self.params = cv2.aruco.DetectorParameters()
    
    def get_board_dimensions(self):
        """Get physical dimensions of the board in meters."""
        width = self.board_size[0] * self.square_length
        height = self.board_size[1] * self.square_length
        return width, height

    def detect(self, image, K, D):
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        corners, ids, _ = cv2.aruco.detectMarkers(gray, self.dictionary, parameters=self.params)
        if ids is not None and len(ids) > 0:
            _, charuco_corners, charuco_ids = cv2.aruco.interpolateCornersCharuco(corners, ids, gray, self.board)
            if charuco_corners is not None and len(charuco_corners) > 3:
                valid, rvec, tvec = cv2.aruco.estimatePoseCharucoBoard(charuco_corners, charuco_ids, self.board, K, D, None, None)
                return valid, rvec, tvec
        return False, None, None

def load_calibration(calib_path):
    """Load the hand-eye calibration result."""
    with open(calib_path, 'r') as f:
        calib = json.load(f)
    
    T_cam_gripper = np.array(calib['T_cam_gripper'])
    return T_cam_gripper

def compute_alignment_pose(T_gripper_base_current, T_cam_gripper, rvec, tvec, offset_distance=0.1, target_point_in_board=[0, 0, 0]):
    """
    Compute the desired gripper pose to align END EFFECTOR with charuco board.
    
    Args:
        T_gripper_base_current: Current gripper pose in base frame (4x4)
        T_cam_gripper: Camera to gripper transform from calibration (4x4)
        rvec: Rotation vector of target in camera frame
        tvec: Translation vector of target in camera frame
        offset_distance: Distance to maintain from board (meters)
        target_point_in_board: [x, y, z] point in board frame to align with (default: origin)
    
    Returns:
        T_gripper_base_desired: Desired gripper pose in base frame (4x4)
    """
    # Convert rvec, tvec to transformation matrix
    R_target_cam, _ = cv2.Rodrigues(rvec)
    t_target_cam = tvec.flatten()
    
    T_target_cam = np.eye(4)
    T_target_cam[:3, :3] = R_target_cam
    T_target_cam[:3, 3] = t_target_cam
    
    # Current camera pose in base frame
    T_cam_base_current = T_gripper_base_current @ T_cam_gripper
    
    # Target (charuco board) pose in base frame
    T_target_base = T_cam_base_current @ T_target_cam
    
    # Desired gripper pose: aligned with charuco board at target_point, offset along Z axis (normal)
    # The gripper frame should match the board's orientation
    # Offset by offset_distance along the board's Z-axis (pointing away from board)
    T_gripper_target_desired = np.eye(4)
    T_gripper_target_desired[:3, 3] = [
        target_point_in_board[0], 
        target_point_in_board[1], 
        target_point_in_board[2] - offset_distance
    ]
    
    # Desired gripper pose in base frame
    T_gripper_base_desired = T_target_base @ T_gripper_target_desired
    
    return T_gripper_base_desired

def get_board_corners(board_width, board_height):
    """
    Get the 4 corners of the charuco board in the board's frame.
    
    OpenCV charuco frame: origin at top-left, X right, Y down, Z out.
    
    Args:
        board_width: Width of board in meters
        board_height: Height of board in meters
    
    Returns:
        List of 4 corner positions [x, y, z] in board frame
    """
    corners = [
        [0, 0, 0],                          # Top-left
        [board_width, 0, 0],                # Top-right
        [board_width, board_height, 0],     # Bottom-right
        [0, board_height, 0]                # Bottom-left
    ]
    return corners

def plot_verification(T_gripper_base, T_cam_gripper, T_target_cam):
    """
    Plot the frames after verification movement.
    
    Args:
        T_gripper_base: 4x4 matrix (Gripper in Base)
        T_cam_gripper: 4x4 matrix (Camera in Gripper - Calibration result)
        T_target_cam: 4x4 matrix (Target in Camera - Detection result)
    """
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection='3d')
    
    def plot_frame(T, label, scale=0.1):
        R = T[:3, :3]
        t = T[:3, 3]
        ax.quiver(t[0], t[1], t[2], R[0,0], R[1,0], R[2,0], length=scale, color='r')
        ax.quiver(t[0], t[1], t[2], R[0,1], R[1,1], R[2,1], length=scale, color='g')
        ax.quiver(t[0], t[1], t[2], R[0,2], R[1,2], R[2,2], length=scale, color='b')
        ax.text(t[0], t[1], t[2], label)

    # Plot Base Frame (0,0,0)
    plot_frame(np.eye(4), "Base", scale=0.2)
    
    # Plot Gripper Frame
    plot_frame(T_gripper_base, "Gripper")
    
    # Plot Camera Frame
    # T_cam_base = T_gripper_base * T_cam_gripper
    T_cam_base = T_gripper_base @ T_cam_gripper
    plot_frame(T_cam_base, "Camera")
    
    # Plot Target (Charuco) Frame
    # T_target_base = T_cam_base * T_target_cam
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

def main():
    parser = argparse.ArgumentParser(
        description="Verify Hand-Eye Calibration by Aligning with Charuco Board"
    )
    parser.add_argument("--host", default="172.16.0.2", help="Robot FCI IP address")
    parser.add_argument("--calibration", default="data/hand-eye-calibration-output/calibration_result.json",
                        help="Path to calibration result")
    parser.add_argument("--offset", type=float, default=0.06,
                        help="Distance from board in meters (default: 0.06)")
    args = parser.parse_args()

    print("=" * 70)
    print("Franka Hand-Eye Calibration - Verification")
    print("=" * 70)
    
    # Warning message
    print("\n⚠️  WARNING: This script will move the robot!")
    print("\n📋 Before proceeding, you should:")
    print("   1. Verify the calibration visually with:")
    print("      python scripts/compute_calibration.py --plot")
    print("   2. Ensure the charuco board is visible to the camera")
    print("   3. Ensure the workspace is clear and safe for robot motion")
    print("   4. Have the emergency stop readily accessible")
    print(f"\nThis script will move the robot to align the end effector {args.offset}m")
    print("above the detected charuco board, visiting the center and all 4 corners.")
    print("\n" + "=" * 70)
    
    response = input("\n➤ Press ENTER to continue or Ctrl+C to abort: ")
    
    # Load calibration
    calib_path = Path(args.calibration)
    if not calib_path.exists():
        print(f"\n❌ ERROR: Calibration file not found: {calib_path}")
        print("   Run: python scripts/compute_calibration.py")
        return 1
    
    print(f"\n✓ Loading calibration from {calib_path}")
    T_cam_gripper = load_calibration(calib_path)
    
    # Initialize camera
    print("✓ Initializing RealSense camera...")
    try:
        camera = RealSenseCamera()
        K, D = camera.get_intrinsics_matrix()
        print("✓ Camera initialized")
    except Exception as e:
        print(f"❌ ERROR: Failed to initialize camera: {e}")
        return 1
    
    # Initialize Charuco detector
    print("✓ Loading Charuco board parameters...")
    try:
        detector = CharucoDetector("config/calibration_board_parameters.yaml")
        print("✓ Charuco detector initialized")
    except Exception as e:
        print(f"❌ ERROR: Failed to load Charuco parameters: {e}")
        camera.stop()
        return 1
    
    # Connect to robot
    print(f"✓ Connecting to robot at {args.host}...")
    try:
        robot = Robot(args.host)
        robot.recover_from_errors()
        robot.relative_dynamics_factor = 0.1  # Slow and safe
        
        print("✓ Robot connected")
        
        # Initialize gripper
        gripper = Gripper(args.host)
        print("✓ Gripper initialized")
        
        # Move to Home Pose
        print("🤖 Moving to home pose...")
        robot.move(JointMotion([0.0, 0.0, 0.0, -2.2, 0.0, 2.2, 0.7]))
        print("✓ Reached home pose")
        
        # Home and close gripper
        print("🏠 Homing gripper...")
        gripper.homing()
        print("✓ Gripper homed")
        
        print("🤏 Closing gripper...")
        gripper.move(0, 0.1)  # width=0 (closed), speed=0.1
        print("✓ Gripper closed")
        
    except Exception as e:
        print(f"❌ ERROR: Failed to connect to robot: {e}")
        camera.stop()
        return 1
    
    try:
        # Get current robot state
        print("\n📷 Detecting charuco board from current position...")
        time.sleep(1.0) # Wait for camera/robot to stabilize
        frame = camera.get_frame()
        if frame is None:
            print("❌ ERROR: Failed to capture frame")
            return 1
        
        valid, rvec, tvec = detector.detect(frame, K, D)
        
        if not valid:
            print("❌ ERROR: Charuco board not detected!")
            print("   Make sure the board is visible in the camera view.")
            return 1
        
        print("✓ Charuco board detected")
        print(f"   Position in camera frame: {tvec.flatten()}")
        
        # Get current gripper pose
        state = robot.state
        O_T_EE_affine = state.O_T_EE
        O_T_EE_matrix_remote = O_T_EE_affine.matrix
        
        T_gripper_base_current = np.zeros((4, 4))
        for r in range(4):
            row_remote = O_T_EE_matrix_remote[r]
            for c in range(4):
                T_gripper_base_current[r, c] = float(row_remote[c])
        
        print("✓ Current gripper pose obtained")
        
        # Get board dimensions for center calculation
        board_width, board_height = detector.get_board_dimensions()
        center_point = [board_width / 2, board_height / 2, 0]
        
        # Compute desired pose (center of board)
        print(f"\n🧮 Computing alignment pose at board center ({args.offset}m offset)...")
        print(f"   Board dimensions: {board_width:.3f}m x {board_height:.3f}m")
        print(f"   Center point in board frame: {center_point}")
        T_gripper_base_desired = compute_alignment_pose(
            T_gripper_base_current, T_cam_gripper, rvec, tvec, args.offset, center_point
        )
        
        # Convert to lists for RPyC/pybind11 compatibility
        # pybind11 on the server side should automatically convert these lists 
        # to the required numpy arrays
        translation_list = T_gripper_base_desired[:3, 3].tolist()
        rotation_matrix = T_gripper_base_desired[:3, :3]
        quaternion_list = R.from_matrix(rotation_matrix).as_quat().tolist() # [x, y, z, w]
        
        print("✓ Target pose computed")
        print("\nDesired gripper position (base frame):")
        print(f"   xyz: {translation_list}")
        print(f"   quat (xyzw): {quaternion_list}")
        
        # Show preview plot of planned alignment
        print("\n📊 Displaying preview of center alignment...")
        print("   (Close the plot window to continue)")
        
        # Calculate frames for preview
        R_target_cam_initial, _ = cv2.Rodrigues(rvec)
        T_target_cam_initial = np.eye(4)
        T_target_cam_initial[:3, :3] = R_target_cam_initial
        T_target_cam_initial[:3, 3] = tvec.flatten()
        
        T_cam_base_initial = T_gripper_base_current @ T_cam_gripper
        T_target_base = T_cam_base_initial @ T_target_cam_initial
        
        # For preview, use the TARGET gripper pose we computed
        T_gripper_base_preview = T_gripper_base_desired
        T_cam_base_preview = T_gripper_base_preview @ T_cam_gripper
        T_target_cam_preview = np.linalg.inv(T_cam_base_preview) @ T_target_base
        
        plot_verification(T_gripper_base_preview, T_cam_gripper, T_target_cam_preview)
        
        # Confirmation after viewing plot
        print("\n" + "=" * 70)
        print("The robot will visit:")
        print("  1. Center of the board")
        print("  2. All 4 corners (Top-Left, Top-Right, Bottom-Right, Bottom-Left)")
        response = input("\n➤ Proceed with corner tour? Press ENTER to continue or Ctrl+C to abort: ")
        
        # Execute absolute motion to center
        print("\n🤖 Moving robot to board center...")
        target_affine = Affine(translation_list, quaternion_list)
        motion = CartesianMotion(target_affine, ReferenceType.Absolute, 0.1)
        robot.move(motion)
        
        print("✓ Reached board center!")
        
        print("\n" + "=" * 70)
        print("✅ Center alignment complete!")
        print("=" * 70)
        
        # Now visit the 4 corners
        print("\n🎯 Now visiting the 4 corners of the charuco board...")
        board_width, board_height = detector.get_board_dimensions()
        corners = get_board_corners(board_width, board_height)
        corner_names = ["Top-Left", "Top-Right", "Bottom-Right", "Bottom-Left"]
        
        for i, (corner, corner_name) in enumerate(zip(corners, corner_names)):
            print(f"\n--- Corner {i+1}/4: {corner_name} ---")
            print(f"   Position in board frame: {corner}")
            
            # Compute alignment pose for this corner
            T_gripper_base_corner = compute_alignment_pose(
                T_gripper_base_current, T_cam_gripper, rvec, tvec, args.offset, corner
            )
            
            # Convert to lists for RPyC
            translation_corner = T_gripper_base_corner[:3, 3].tolist()
            rotation_corner = T_gripper_base_corner[:3, :3]
            quaternion_corner = R.from_matrix(rotation_corner).as_quat().tolist()
            
            print(f"   Target gripper position: {translation_corner}")
            
            # Move to corner
            print(f"🤖 Moving to {corner_name} corner...")
            target_affine_corner = Affine(translation_corner, quaternion_corner)
            motion_corner = CartesianMotion(target_affine_corner, ReferenceType.Absolute, 0.1)
            robot.move(motion_corner)
            print(f"✓ Reached {corner_name} corner")
            
            # Small delay between corners
            time.sleep(0.3)
        
        print("\n" + "=" * 70)
        print("✅ SUCCESS: All corners visited successfully!")
        print("=" * 70)
        print("\nThe robot has visited the center and all 4 corners of the charuco board.")
        print(f"Each position maintained a {args.offset}m offset from the board surface.")
        print("\nIf all positions aligned correctly, your hand-eye calibration is accurate!")
        
        # Return to home position
        print("\n🏠 Returning to home position...")
        robot.move(JointMotion([0.0, 0.0, 0.0, -2.2, 0.0, 2.2, 0.7]))
        print("✓ Returned to home position")
        
        print("\n" + "=" * 70)
        print("🎉 Verification complete!")
        print("=" * 70)
        
        return 0
        
    except KeyboardInterrupt:
        print("\n\n❌ Aborted by user")
        return 1
    except Exception as e:
        print(f"\n❌ ERROR: {e}")
        import traceback
        traceback.print_exc()
        return 1
    finally:
        camera.stop()
        print("\n✓ Camera stopped")

if __name__ == "__main__":
    exit(main())

