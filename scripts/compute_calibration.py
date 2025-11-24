#!/usr/bin/env python3
"""
Franka Hand-Eye Calibration - Compute Calibration Script

Computes hand-eye calibration from captured data.
Can be run standalone or imported as a module.

Usage:
    python scripts/compute_calibration.py --data data/captured-data --plot
"""

import sys
import argparse
import numpy as np
import cv2
from pathlib import Path
from scipy.spatial.transform import Rotation as R

# Add parent directory to path for standalone execution
sys.path.insert(0, str(Path(__file__).parent.parent))

from franka_handeye import (
    load_captured_data,
    compute_hand_eye_calibration,
    compute_consistency_metrics,
    save_calibration_result,
)


def plot_frames(R_g2b, t_g2b, R_t2c, t_t2c, T_cam2gripper):
    """
    Plot 3D visualization of coordinate frames.
    
    Parameters
    ----------
    R_g2b : list
        Gripper-to-base rotation matrices.
    t_g2b : list
        Gripper-to-base translations.
    R_t2c : list
        Target-to-camera rotation matrices.
    t_t2c : list
        Target-to-camera translations.
    T_cam2gripper : np.ndarray
        4x4 camera-to-gripper transformation.
    """
    import matplotlib.pyplot as plt
    from mpl_toolkits.mplot3d import Axes3D
    
    fig = plt.figure(figsize=(10, 8))
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
    last_idx = -1
    R_last_g2b = R_g2b[last_idx]
    t_last_g2b = t_g2b[last_idx]
    plot_frame(R_last_g2b, t_last_g2b, "Gripper")
    
    # Calculate Camera Frame in Base
    T_g2b = np.eye(4)
    T_g2b[:3, :3] = R_last_g2b
    T_g2b[:3, 3] = t_last_g2b
    
    T_c2b = T_g2b @ T_cam2gripper
    plot_frame(T_c2b[:3, :3], T_c2b[:3, 3], "Camera")
    
    # Calculate Target (Charuco) Frame in Base
    R_last_t2c = R_t2c[last_idx]
    t_last_t2c = t_t2c[last_idx]
    
    T_t2c = np.eye(4)
    T_t2c[:3, :3] = R_last_t2c
    T_t2c[:3, 3] = t_last_t2c
    
    T_t2b = T_c2b @ T_t2c
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


def run_calibration(
    data_dir: str = "data/captured-data",
    output_dir: str = "data/hand-eye-calibration-output",
    show_plot: bool = False,
    method: int = cv2.CALIB_HAND_EYE_DANIILIDIS
) -> dict | None:
    """
    Run the complete calibration pipeline.
    
    Parameters
    ----------
    data_dir : str
        Directory containing captured data.
    output_dir : str
        Directory to save calibration result.
    show_plot : bool
        Whether to show 3D visualization.
    method : int
        OpenCV calibration method.
    
    Returns
    -------
    dict or None
        Calibration result dictionary, or None if failed.
    """
    # Load data
    R_g2b, t_g2b, R_t2c, t_t2c = load_captured_data(data_dir)
    
    if len(R_g2b) < 3:
        print("Error: Need at least 3 poses for calibration.")
        return None
    
    print(f"Running calibration using DANIILIDIS method...")
    
    # Compute calibration
    R_cam2gripper, t_cam2gripper = compute_hand_eye_calibration(
        R_g2b, t_g2b, R_t2c, t_t2c, method=method
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
    quat = R.from_matrix(R_cam2gripper).as_quat()
    print("\nQuaternion [x, y, z, w]:")
    print(quat)
    
    # Compute Consistency
    mean_err, std_err = compute_consistency_metrics(
        R_g2b, t_g2b, R_t2c, t_t2c, T_cam2gripper
    )
    
    print("\n--- Repeatability / Consistency Metrics ---")
    print(f"Position Error (Mean): {mean_err:.6f} m")
    print(f"Position Error (Std Dev): {std_err:.6f} m")
    
    # Save result
    output_path = Path(output_dir) / "calibration_result.json"
    result = save_calibration_result(
        output_path, R_cam2gripper, t_cam2gripper, mean_err, std_err
    )
    print(f"\nSaved result to {output_path}")
    
    if show_plot:
        plot_frames(R_g2b, t_g2b, R_t2c, t_t2c, T_cam2gripper)
    
    return result


def main():
    parser = argparse.ArgumentParser(description="Compute Hand-Eye Calibration")
    parser.add_argument("--data", default="data/captured-data", help="Directory with captured data")
    parser.add_argument("--output", default="data/hand-eye-calibration-output", help="Output directory")
    parser.add_argument("--plot", action="store_true", help="Show 3D plot of frames")
    args = parser.parse_args()

    print("=" * 60)
    print("Franka Hand-Eye Calibration - Compute")
    print("=" * 60)
    
    result = run_calibration(args.data, args.output, args.plot)
    
    if result:
        print("\n" + "=" * 60)
        print("SUCCESS: Calibration complete!")
        print("=" * 60)
        print("\nNext step: Verify calibration with:")
        print("  python scripts/verify_calibration.py")
        return 0
    else:
        return 1


if __name__ == "__main__":
    exit(main())
