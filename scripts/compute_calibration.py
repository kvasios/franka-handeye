import argparse
import json
import numpy as np
import cv2
from pathlib import Path
from scipy.spatial.transform import Rotation as R
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

def load_captured_data(data_dir):
    data_dir = Path(data_dir)
    pose_dirs = sorted([d for d in data_dir.iterdir() if d.is_dir() and d.name.startswith("pose_")])
    
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
            
        # Extract Robot Pose (T_gripper2base)
        O_T_EE = np.array(data["O_T_EE"])
        
        if O_T_EE.shape == (16,):
            T_g2b = O_T_EE.reshape(4, 4).T
        elif O_T_EE.shape == (4, 4):
            if np.allclose(O_T_EE[3, :], [0, 0, 0, 1]):
                T_g2b = O_T_EE
            elif np.allclose(O_T_EE[:, 3], [0, 0, 0, 1]):
                 T_g2b = O_T_EE.T
            else:
                 T_g2b = O_T_EE 
        else:
             continue

        R_g2b = T_g2b[:3, :3]
        t_g2b = T_g2b[:3, 3]
        
        # Extract Target Pose (T_target2cam)
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

def plot_frames(R_g2b, t_g2b, R_t2c, t_t2c, T_cam2gripper):
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    
    def plot_frame(R, t, label, scale=0.1):
        t = t.flatten()
        ax.quiver(t[0], t[1], t[2], R[0,0], R[1,0], R[2,0], length=scale, color='r')
        ax.quiver(t[0], t[1], t[2], R[0,1], R[1,1], R[2,1], length=scale, color='g')
        ax.quiver(t[0], t[1], t[2], R[0,2], R[1,2], R[2,2], length=scale, color='b')
        ax.text(t[0], t[1], t[2], label)

    # Plot Base Frame (0,0,0)
    plot_frame(np.eye(3), np.zeros(3), "Base")
    
    # Plot Last Pose Frames
    # T_gripper2base
    last_idx = -1
    R_last_g2b = R_g2b[last_idx]
    t_last_g2b = t_g2b[last_idx]
    plot_frame(R_last_g2b, t_last_g2b, "Gripper")
    
    # Calculate Camera Frame in Base
    # T_cam2base = T_gripper2base * T_cam2gripper
    T_g2b = np.eye(4)
    T_g2b[:3, :3] = R_last_g2b
    T_g2b[:3, 3] = t_last_g2b
    
    T_c2b = T_g2b @ T_cam2gripper
    plot_frame(T_c2b[:3, :3], T_c2b[:3, 3], "Camera")
    
    # Calculate Target (Charuco) Frame in Base
    # T_target2base = T_cam2base * T_target2cam
    R_last_t2c = R_t2c[last_idx]
    t_last_t2c = t_t2c[last_idx]
    
    T_t2c = np.eye(4)
    T_t2c[:3, :3] = R_last_t2c
    T_t2c[:3, 3] = t_last_t2c
    
    T_t2b = T_c2b @ T_t2c # Note: T_target2cam is pose of target IN camera frame -> T_c2b * T_t2c transforms target point to base? No.
    # T_target2cam means transformation FROM target TO camera.
    # So Point_cam = T_target2cam * Point_target
    # Point_base = T_cam2base * Point_cam = T_cam2base * T_target2cam * Point_target
    # So T_target2base = T_cam2base * T_target2cam
    # Correct.
    
    plot_frame(T_t2b[:3, :3], T_t2b[:3, 3], "Charuco")

    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    # Auto scale
    all_points = np.vstack([np.zeros(3), t_last_g2b, T_c2b[:3, 3], T_t2b[:3, 3]])
    min_xyz = np.min(all_points, axis=0)
    max_xyz = np.max(all_points, axis=0)
    ax.set_xlim(min_xyz[0]-0.2, max_xyz[0]+0.2)
    ax.set_ylim(min_xyz[1]-0.2, max_xyz[1]+0.2)
    ax.set_zlim(min_xyz[2]-0.2, max_xyz[2]+0.2)
    
    plt.title("Frames Visualization (Last Pose)")
    plt.show()

def compute_error_metrics(R_g2b, t_g2b, R_t2c, t_t2c, T_cam2gripper):
    # Error metric:
    # The transform from Base to Target (Charuco) should be constant across all poses.
    # T_target2base_i = T_gripper2base_i * T_cam2gripper * T_target2cam_i
    
    target_positions = []
    target_orientations = [] # Quaternions
    
    for i in range(len(R_g2b)):
        T_g2b = np.eye(4)
        T_g2b[:3, :3] = R_g2b[i]
        T_g2b[:3, 3] = t_g2b[i].flatten()
        
        T_t2c = np.eye(4)
        T_t2c[:3, :3] = R_t2c[i]
        T_t2c[:3, 3] = t_t2c[i].flatten() # Note: OpenCV returns tvec as column, flatten ensures consistent shape
        
        # T_target2base = T_gripper2base * T_cam2gripper * T_target2cam ? 
        # Wait. T_target2cam usually means "Pose of Target in Camera Frame". 
        # So P_cam = T_target2cam * P_target.
        # P_gripper = T_cam2gripper * P_cam
        # P_base = T_gripper2base * P_gripper
        # So P_base = T_gripper2base * T_cam2gripper * T_target2cam * P_target
        # Yes.
        
        T_t2b = T_g2b @ T_cam2gripper @ T_t2c
        
        target_positions.append(T_t2b[:3, 3])
        target_orientations.append(R.from_matrix(T_t2b[:3, :3]).as_quat())

    target_positions = np.array(target_positions)
    mean_pos = np.mean(target_positions, axis=0)
    pos_errors = np.linalg.norm(target_positions - mean_pos, axis=1)
    
    # Orientation error (approximate angle difference)
    # Calculate mean orientation? A bit complex for quats.
    # Let's just take the first one as reference or mean if small dispersion.
    # Pairwise difference from mean position is good enough for now.
    
    print("\n--- Repeatability / Consistency Metrics ---")
    print(f"Mean Target Position (Base Frame): {mean_pos}")
    print(f"Position Error (Std Dev): {np.std(pos_errors):.6f} m")
    print(f"Max Position Error: {np.max(pos_errors):.6f} m")
    
    # Angular error logic could be added here
    
    return np.mean(pos_errors), np.std(pos_errors)

def main():
    parser = argparse.ArgumentParser(description="Compute Hand-Eye Calibration")
    parser.add_argument("--data", default="data/captured-data", help="Directory with captured data")
    parser.add_argument("--method", default="daniilidis", help="Calibration method (daniilidis, tsai, etc.)")
    parser.add_argument("--plot", action="store_true", help="Show 3D plot of frames")
    args = parser.parse_args()

    R_g2b, t_g2b, R_t2c, t_t2c = load_captured_data(args.data)

    if len(R_g2b) < 3:
        print("Error: Need at least 3 poses for calibration.")
        return

    print("Running calibration...")
    
    method_dict = {
        "daniilidis": cv2.CALIB_HAND_EYE_DANIILIDIS,
        "tsai": cv2.CALIB_HAND_EYE_TSAI,
        "park": cv2.CALIB_HAND_EYE_PARK,
        "horaud": cv2.CALIB_HAND_EYE_HORAUD,
        "andrei": cv2.CALIB_HAND_EYE_ANDREI
    }
    
    method = method_dict.get(args.method.lower(), cv2.CALIB_HAND_EYE_DANIILIDIS)
    
    R_cam2gripper, t_cam2gripper = cv2.calibrateHandEye(
        R_gripper2base=R_g2b,
        t_gripper2base=t_g2b,
        R_target2cam=R_t2c,
        t_target2cam=t_t2c,
        method=method
    )

    print("\nCalibration Result (T_cam_gripper):")
    print("Rotation Matrix (R_cam2gripper):")
    print(R_cam2gripper)
    print("\nTranslation Vector (t_cam2gripper):")
    print(t_cam2gripper.flatten())
    
    # Create Homogeneous Matrix
    T_cam2gripper = np.eye(4)
    T_cam2gripper[:3, :3] = R_cam2gripper
    T_cam2gripper[:3, 3] = t_cam2gripper.flatten()
    
    print("\nHomogeneous Matrix T_cam_gripper:")
    print(T_cam2gripper)
    
    # Convert to Quaternion
    quat = R.from_matrix(R_cam2gripper).as_quat() # [x, y, z, w]
    print("\nQuaternion [x, y, z, w]:")
    print(quat)
    
    # Compute Consistency
    mean_err, std_err = compute_error_metrics(R_g2b, t_g2b, R_t2c, t_t2c, T_cam2gripper)

    # Save result
    result = {
        "T_cam_gripper": T_cam2gripper.tolist(),
        "xyz": t_cam2gripper.flatten().tolist(),
        "quaternion_xyzw": quat.tolist(),
        "consistency_error_mean": float(mean_err),
        "consistency_error_std": float(std_err)
    }
    
    output_dir = Path("data/hand-eye-calibration-output")
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / "calibration_result.json"

    with open(output_path, 'w') as f:
        json.dump(result, f, indent=4)
    print(f"\nSaved result to {output_path}")

    if args.plot:
        plot_frames(R_g2b, t_g2b, R_t2c, t_t2c, T_cam2gripper)

if __name__ == "__main__":
    main()
