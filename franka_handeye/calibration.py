"""
Hand-eye calibration computation and verification.
"""

import json
import numpy as np
import cv2
from pathlib import Path
from scipy.spatial.transform import Rotation as R


class NumpyEncoder(json.JSONEncoder):
    """JSON encoder that handles numpy types."""
    
    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        if isinstance(obj, np.floating):
            return float(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return super().default(obj)


def load_captured_data(data_dir: str | Path) -> tuple[list, list, list, list]:
    """
    Load captured calibration data from directory.
    
    Parameters
    ----------
    data_dir : str or Path
        Directory containing pose_XX subdirectories with data.json files.
    
    Returns
    -------
    tuple
        (R_gripper2base, t_gripper2base, R_target2cam, t_target2cam)
        Lists of rotation matrices and translation vectors.
    """
    data_dir = Path(data_dir)
    pose_dirs = sorted([
        d for d in data_dir.iterdir() 
        if d.is_dir() and d.name.startswith("pose_")
    ])
    
    R_gripper2base = []
    t_gripper2base = []
    R_target2cam = []
    t_target2cam = []
    
    valid_poses = 0
    
    for p_dir in pose_dirs:
        json_path = p_dir / "data.json"
        if not json_path.exists():
            continue
            
        with open(json_path, 'r') as f:
            data = json.load(f)
            
        if not data.get("charuco_detected", False):
            print(f"Skipping {p_dir.name}: Charuco not detected.")
            continue
            
        # Extract Robot Pose (T_gripper2base = O_T_EE from Franka)
        O_T_EE = np.array(data["O_T_EE"])
        T_g2b = O_T_EE.reshape(4, 4)
        
        # Verify it's a valid transformation matrix
        if not np.allclose(T_g2b[3, :], [0, 0, 0, 1]):
            print(f"Warning: {p_dir.name} has invalid homogeneous row")
            continue
        
        R_g2b = T_g2b[:3, :3]
        t_g2b = T_g2b[:3, 3]
        
        # Extract Target Pose from OpenCV solvePnP
        rvec = np.array(data["T_cam_target_rvec"]).flatten()
        tvec = np.array(data["T_cam_target_tvec"]).flatten()
        
        R_t2c, _ = cv2.Rodrigues(rvec)
        t_t2c = tvec
        
        R_gripper2base.append(R_g2b)
        t_gripper2base.append(t_g2b)
        R_target2cam.append(R_t2c)
        t_target2cam.append(t_t2c)
        
        valid_poses += 1
    
    print(f"Loaded {valid_poses} valid poses for calibration.")
    return R_gripper2base, t_gripper2base, R_target2cam, t_target2cam


def compute_hand_eye_calibration(
    R_gripper2base: list,
    t_gripper2base: list,
    R_target2cam: list,
    t_target2cam: list,
    method: int = cv2.CALIB_HAND_EYE_DANIILIDIS
) -> tuple[np.ndarray, np.ndarray]:
    """
    Compute hand-eye calibration using OpenCV.
    
    Parameters
    ----------
    R_gripper2base : list
        List of 3x3 rotation matrices (gripper to base).
    t_gripper2base : list
        List of translation vectors (gripper to base).
    R_target2cam : list
        List of 3x3 rotation matrices (target to camera).
    t_target2cam : list
        List of translation vectors (target to camera).
    method : int
        OpenCV calibration method. Default is DANIILIDIS.
    
    Returns
    -------
    tuple
        (R_cam2gripper, t_cam2gripper) rotation matrix and translation vector.
    """
    if len(R_gripper2base) < 3:
        raise ValueError("Need at least 3 poses for calibration.")
    
    R_cam2gripper, t_cam2gripper = cv2.calibrateHandEye(
        R_gripper2base=R_gripper2base,
        t_gripper2base=t_gripper2base,
        R_target2cam=R_target2cam,
        t_target2cam=t_target2cam,
        method=method
    )
    
    return R_cam2gripper, t_cam2gripper


def compute_consistency_metrics(
    R_gripper2base: list,
    t_gripper2base: list,
    R_target2cam: list,
    t_target2cam: list,
    T_cam2gripper: np.ndarray
) -> tuple[float, float]:
    """
    Compute consistency/repeatability metrics for calibration.
    
    The target (charuco board) position in base frame should be constant
    across all poses if calibration is accurate.
    
    Parameters
    ----------
    R_gripper2base : list
        List of gripper-to-base rotation matrices.
    t_gripper2base : list
        List of gripper-to-base translations.
    R_target2cam : list
        List of target-to-camera rotation matrices.
    t_target2cam : list
        List of target-to-camera translations.
    T_cam2gripper : np.ndarray
        4x4 camera-to-gripper transformation matrix.
    
    Returns
    -------
    tuple
        (mean_error, std_error) position errors in meters.
    """
    target_positions = []
    
    for i in range(len(R_gripper2base)):
        T_g2b = np.eye(4)
        T_g2b[:3, :3] = R_gripper2base[i]
        T_g2b[:3, 3] = t_gripper2base[i].flatten()
        
        T_t2c = np.eye(4)
        T_t2c[:3, :3] = R_target2cam[i]
        T_t2c[:3, 3] = t_target2cam[i].flatten()
        
        # T_target2base = T_gripper2base * T_cam2gripper * T_target2cam
        T_t2b = T_g2b @ T_cam2gripper @ T_t2c
        target_positions.append(T_t2b[:3, 3])
    
    target_positions = np.array(target_positions)
    mean_pos = np.mean(target_positions, axis=0)
    pos_errors = np.linalg.norm(target_positions - mean_pos, axis=1)
    
    return float(np.mean(pos_errors)), float(np.std(pos_errors))


def save_calibration_result(
    output_path: str | Path,
    R_cam2gripper: np.ndarray,
    t_cam2gripper: np.ndarray,
    consistency_error_mean: float = None,
    consistency_error_std: float = None
) -> dict:
    """
    Save calibration result to JSON file.
    
    Parameters
    ----------
    output_path : str or Path
        Output file path.
    R_cam2gripper : np.ndarray
        3x3 rotation matrix.
    t_cam2gripper : np.ndarray
        Translation vector.
    consistency_error_mean : float, optional
        Mean consistency error.
    consistency_error_std : float, optional
        Std of consistency error.
    
    Returns
    -------
    dict
        The saved result dictionary.
    """
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    # Create homogeneous matrix
    T_cam2gripper = np.eye(4)
    T_cam2gripper[:3, :3] = R_cam2gripper
    T_cam2gripper[:3, 3] = t_cam2gripper.flatten()
    
    # Convert to quaternion
    quat = R.from_matrix(R_cam2gripper).as_quat()  # [x, y, z, w]
    
    result = {
        "T_cam_gripper": T_cam2gripper.tolist(),
        "xyz": t_cam2gripper.flatten().tolist(),
        "quaternion_xyzw": quat.tolist(),
    }
    
    if consistency_error_mean is not None:
        result["consistency_error_mean"] = consistency_error_mean
    if consistency_error_std is not None:
        result["consistency_error_std"] = consistency_error_std
    
    with open(output_path, 'w') as f:
        json.dump(result, f, indent=4)
    
    return result


def load_calibration_result(calib_path: str | Path) -> np.ndarray:
    """
    Load calibration result from JSON file.
    
    Parameters
    ----------
    calib_path : str or Path
        Path to calibration JSON file.
    
    Returns
    -------
    np.ndarray
        4x4 camera-to-gripper transformation matrix.
    """
    with open(calib_path, 'r') as f:
        calib = json.load(f)
    
    return np.array(calib['T_cam_gripper'])


def compute_alignment_pose(
    T_gripper_base_current: np.ndarray,
    T_cam_gripper: np.ndarray,
    rvec: np.ndarray,
    tvec: np.ndarray,
    offset_distance: float = 0.1,
    target_point_in_board: list = None
) -> np.ndarray:
    """
    Compute desired gripper pose to align with charuco board.
    
    Parameters
    ----------
    T_gripper_base_current : np.ndarray
        Current gripper pose in base frame (4x4).
    T_cam_gripper : np.ndarray
        Camera to gripper transform from calibration (4x4).
    rvec : np.ndarray
        Rotation vector of target in camera frame.
    tvec : np.ndarray
        Translation vector of target in camera frame.
    offset_distance : float
        Distance to maintain from board (meters).
    target_point_in_board : list
        [x, y, z] point in board frame to align with. Default: origin.
    
    Returns
    -------
    np.ndarray
        Desired gripper pose in base frame (4x4).
    """
    if target_point_in_board is None:
        target_point_in_board = [0, 0, 0]
    
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
    
    # Desired gripper pose: aligned with charuco board, offset along Z axis
    T_gripper_target_desired = np.eye(4)
    T_gripper_target_desired[:3, 3] = [
        target_point_in_board[0],
        target_point_in_board[1],
        target_point_in_board[2] - offset_distance
    ]
    
    # Desired gripper pose in base frame
    T_gripper_base_desired = T_target_base @ T_gripper_target_desired
    
    return T_gripper_base_desired

