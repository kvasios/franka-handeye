import os
import time
import argparse
import yaml
import numpy as np
import cv2
import pyrealsense2 as rs
import json
from pathlib import Path
import shutil

# Set default server IP
os.environ.setdefault("FRANKY_SERVER_IP", "192.168.122.100")

from scipy.spatial.transform import Rotation as R
from franky import Robot, JointMotion

# Helper for JSON serialization of numpy types
class NumpyEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        if isinstance(obj, np.floating):
            return float(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return super(NumpyEncoder, self).default(obj)

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
        self.board = cv2.aruco.CharucoBoard(
            (params['board_size'][0], params['board_size'][1]),
            params['square_length'],
            params['marker_length'],
            cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_6X6_250)
        )
        self.dictionary = self.board.getDictionary()
        self.params = cv2.aruco.DetectorParameters()

    def detect(self, image, K, D):
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        corners, ids, _ = cv2.aruco.detectMarkers(gray, self.dictionary, parameters=self.params)
        if ids is not None and len(ids) > 0:
            _, charuco_corners, charuco_ids = cv2.aruco.interpolateCornersCharuco(corners, ids, gray, self.board)
            if charuco_corners is not None and len(charuco_corners) > 3:
                valid, rvec, tvec = cv2.aruco.estimatePoseCharucoBoard(charuco_corners, charuco_ids, self.board, K, D, None, None)
                return valid, rvec, tvec, charuco_corners
        return False, None, None, None

def validate_joint_poses(joint_poses):
    """Validate that joint poses are exactly 12 and properly formatted."""
    if not joint_poses:
        raise ValueError("No joint poses found in config/joint_poses.yaml")
    
    if len(joint_poses) != 12:
        raise ValueError(f"Expected exactly 12 joint poses, got {len(joint_poses)}")
    
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

def capture_poses(robot, camera, detector, joint_poses, output_dir, K, D):
    """Capture data at each joint pose automatically."""
    print(f"\nStarting automatic capture of {len(joint_poses)} poses...\n")
    
    # Clean output directory
    if output_dir.exists():
        shutil.rmtree(output_dir)
    output_dir.mkdir(parents=True)
    
    successful_captures = 0
    
    for i, pose in enumerate(joint_poses):
        print(f"--- Pose {i+1}/{len(joint_poses)} ---")
        try:
            # Move to joint pose
            robot.move(JointMotion(pose))
            time.sleep(0.5)  # Stabilization delay
            
            # Capture frame
            frame = camera.get_frame()
            if frame is None:
                print("WARNING: Failed to capture frame")
                continue
            
            # Get robot state
            state = robot.state
            
            # Extract joint positions (manual extraction to avoid pickling issues)
            q_remote = state.q
            q = [float(q_remote[i]) for i in range(len(q_remote))]
            
            # Extract end-effector pose
            O_T_EE_affine = state.O_T_EE
            O_T_EE_matrix_remote = O_T_EE_affine.matrix
            O_T_EE = []
            for r in range(4):
                row_remote = O_T_EE_matrix_remote[r]
                row = [float(row_remote[c]) for c in range(4)]
                O_T_EE.extend(row)
            
            # Detect Charuco board
            valid, rvec, tvec, _ = detector.detect(frame, K, D)
            
            if valid:
                print("✓ Charuco board detected")
                cv2.drawFrameAxes(frame, K, D, rvec, tvec, 0.1)
            else:
                print("✗ WARNING: Charuco board NOT detected")
                
            # Save data
            pose_dir = output_dir / f"pose_{i:02d}"
            pose_dir.mkdir(exist_ok=True)
            cv2.imwrite(str(pose_dir / "image.png"), frame)
            
            data = {
                "joint_pose": q,
                "O_T_EE": O_T_EE,
                "camera_intrinsics": K.tolist() if isinstance(K, np.ndarray) else K,
                "dist_coeffs": D.tolist() if isinstance(D, np.ndarray) else D,
                "charuco_detected": valid,
            }
            
            if valid:
                data["T_cam_target_rvec"] = rvec.tolist() if hasattr(rvec, 'tolist') else rvec
                data["T_cam_target_tvec"] = tvec.tolist() if hasattr(tvec, 'tolist') else tvec
                
            with open(pose_dir / "data.json", 'w') as f:
                json.dump(data, f, indent=4, cls=NumpyEncoder)
            
            successful_captures += 1
            print(f"✓ Saved to {pose_dir.name}\n")
                
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
    args = parser.parse_args()

    output_dir = Path(args.output)

    print("=" * 60)
    print("Franka Hand-Eye Calibration - Data Capture")
    print("=" * 60)

    # Load and validate joint poses
    poses_path = Path("config/joint_poses.yaml")
    if not poses_path.exists():
        print(f"ERROR: {poses_path} not found")
        return 1
    
    with open(poses_path, 'r') as f:
        data = yaml.safe_load(f)
    
    if not data or 'joint_poses' not in data:
        print("ERROR: joint_poses not found in config/joint_poses.yaml")
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
        detector = CharucoDetector("config/calibration_board_parameters.yaml")
        print("✓ Charuco detector initialized")
    except Exception as e:
        print(f"ERROR: Failed to load Charuco parameters: {e}")
        camera.stop()
        return 1
    
    # Connect to robot
    print(f"\nConnecting to robot at {args.host}...")
    try:
        robot = Robot(args.host)
        robot.recover_from_errors()
        robot.relative_dynamics_factor = 0.2  # Moderate speed for safety
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
