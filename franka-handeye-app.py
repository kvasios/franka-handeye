#!/usr/bin/env python3
"""
Franka Hand-Eye Calibration Application

A complete GUI application for hand-eye calibration with Franka robot
and Intel RealSense camera using ChArUco board detection.

Features:
- Live camera feed with ChArUco detection
- Robot jogging controls
- Data capture with pose management
- Calibration computation
- Verification with board alignment

Usage:
    python franka-handeye-app.py --host 172.16.0.2
"""

import gc
import importlib
import os
import sys
import time
import argparse
import json
import shutil
import threading
import yaml
from pathlib import Path

import numpy as np
import cv2
import dearpygui.dearpygui as dpg
from scipy.spatial.transform import Rotation as R

# Set default server IP before importing franky
os.environ.setdefault("FRANKY_SERVER_IP", "192.168.122.100")

from franka_handeye import (
    RealSenseCamera,
    CharucoDetector,
    RobotController,
    NumpyEncoder,
    load_captured_data,
    compute_hand_eye_calibration,
    compute_consistency_metrics,
    save_calibration_result,
    load_calibration_result,
    compute_alignment_pose,
)


# =============================================================================
# Theme & Styling
# =============================================================================

class Theme:
    """Application color theme - Dark industrial aesthetic."""
    
    # Base colors
    BG_DARK = (18, 18, 22)
    BG_MEDIUM = (28, 28, 35)
    BG_LIGHT = (38, 38, 48)
    
    # Accent colors
    ACCENT_PRIMARY = (0, 180, 216)      # Cyan
    ACCENT_SECONDARY = (255, 107, 107)  # Coral
    ACCENT_SUCCESS = (46, 213, 115)     # Green
    ACCENT_WARNING = (255, 193, 7)      # Amber
    ACCENT_DANGER = (255, 71, 87)       # Red
    
    # Text colors
    TEXT_PRIMARY = (240, 240, 245)
    TEXT_SECONDARY = (160, 160, 175)
    TEXT_MUTED = (100, 100, 115)
    
    # Status colors
    CONNECTED = ACCENT_SUCCESS
    DISCONNECTED = ACCENT_DANGER
    DETECTING = ACCENT_PRIMARY
    NOT_DETECTING = ACCENT_WARNING


def setup_theme():
    """Configure DearPyGui theme."""
    with dpg.theme() as global_theme:
        with dpg.theme_component(dpg.mvAll):
            # Window
            dpg.add_theme_color(dpg.mvThemeCol_WindowBg, Theme.BG_DARK)
            dpg.add_theme_color(dpg.mvThemeCol_ChildBg, Theme.BG_MEDIUM)
            dpg.add_theme_color(dpg.mvThemeCol_PopupBg, Theme.BG_MEDIUM)
            
            # Frame/Input
            dpg.add_theme_color(dpg.mvThemeCol_FrameBg, Theme.BG_LIGHT)
            dpg.add_theme_color(dpg.mvThemeCol_FrameBgHovered, (48, 48, 60))
            dpg.add_theme_color(dpg.mvThemeCol_FrameBgActive, (58, 58, 72))
            
            # Buttons
            dpg.add_theme_color(dpg.mvThemeCol_Button, Theme.BG_LIGHT)
            dpg.add_theme_color(dpg.mvThemeCol_ButtonHovered, (58, 58, 72))
            dpg.add_theme_color(dpg.mvThemeCol_ButtonActive, Theme.ACCENT_PRIMARY)
            
            # Header
            dpg.add_theme_color(dpg.mvThemeCol_Header, Theme.BG_LIGHT)
            dpg.add_theme_color(dpg.mvThemeCol_HeaderHovered, (48, 48, 60))
            dpg.add_theme_color(dpg.mvThemeCol_HeaderActive, Theme.ACCENT_PRIMARY)
            
            # Tab
            dpg.add_theme_color(dpg.mvThemeCol_Tab, Theme.BG_LIGHT)
            dpg.add_theme_color(dpg.mvThemeCol_TabHovered, Theme.ACCENT_PRIMARY)
            dpg.add_theme_color(dpg.mvThemeCol_TabActive, Theme.ACCENT_PRIMARY)
            
            # Text
            dpg.add_theme_color(dpg.mvThemeCol_Text, Theme.TEXT_PRIMARY)
            dpg.add_theme_color(dpg.mvThemeCol_TextDisabled, Theme.TEXT_MUTED)
            
            # Progress bar
            dpg.add_theme_color(dpg.mvThemeCol_PlotHistogram, Theme.ACCENT_PRIMARY)
            
            # Separator
            dpg.add_theme_color(dpg.mvThemeCol_Separator, Theme.BG_LIGHT)
            
            # Scrollbar
            dpg.add_theme_color(dpg.mvThemeCol_ScrollbarBg, Theme.BG_DARK)
            dpg.add_theme_color(dpg.mvThemeCol_ScrollbarGrab, Theme.BG_LIGHT)
            
            # Rounding
            dpg.add_theme_style(dpg.mvStyleVar_FrameRounding, 4)
            dpg.add_theme_style(dpg.mvStyleVar_WindowRounding, 0)
            dpg.add_theme_style(dpg.mvStyleVar_ChildRounding, 4)
            dpg.add_theme_style(dpg.mvStyleVar_PopupRounding, 4)
            dpg.add_theme_style(dpg.mvStyleVar_ScrollbarRounding, 4)
            dpg.add_theme_style(dpg.mvStyleVar_GrabRounding, 4)
            dpg.add_theme_style(dpg.mvStyleVar_TabRounding, 4)
            
            # Padding
            dpg.add_theme_style(dpg.mvStyleVar_FramePadding, 8, 6)
            dpg.add_theme_style(dpg.mvStyleVar_ItemSpacing, 8, 6)
            dpg.add_theme_style(dpg.mvStyleVar_WindowPadding, 12, 12)
    
    dpg.bind_theme(global_theme)
    
    # Create accent button theme
    with dpg.theme() as accent_btn_theme:
        with dpg.theme_component(dpg.mvButton):
            dpg.add_theme_color(dpg.mvThemeCol_Button, Theme.ACCENT_PRIMARY)
            dpg.add_theme_color(dpg.mvThemeCol_ButtonHovered, (0, 200, 236))
            dpg.add_theme_color(dpg.mvThemeCol_ButtonActive, (0, 160, 196))
    
    with dpg.theme() as danger_btn_theme:
        with dpg.theme_component(dpg.mvButton):
            dpg.add_theme_color(dpg.mvThemeCol_Button, Theme.ACCENT_DANGER)
            dpg.add_theme_color(dpg.mvThemeCol_ButtonHovered, (255, 91, 107))
            dpg.add_theme_color(dpg.mvThemeCol_ButtonActive, (235, 51, 67))
    
    with dpg.theme() as success_btn_theme:
        with dpg.theme_component(dpg.mvButton):
            dpg.add_theme_color(dpg.mvThemeCol_Button, Theme.ACCENT_SUCCESS)
            dpg.add_theme_color(dpg.mvThemeCol_ButtonHovered, (66, 233, 135))
            dpg.add_theme_color(dpg.mvThemeCol_ButtonActive, (26, 193, 95))
    
    return {
        'accent': accent_btn_theme,
        'danger': danger_btn_theme,
        'success': success_btn_theme,
    }


# =============================================================================
# Application State
# =============================================================================

class AppState:
    """Central application state management."""
    
    def __init__(self):
        # Hardware
        self.robot: RobotController | None = None
        self.camera: RealSenseCamera | None = None
        self.detector: CharucoDetector | None = None
        self.K: np.ndarray | None = None
        self.D: np.ndarray | None = None
        
        # Paths
        self.output_dir = Path("data/captured-data")
        self.calibration_dir = Path("data/hand-eye-calibration-output")
        self.calibration_file = self.calibration_dir / "calibration_result.json"
        self.board_params_path = Path("config/calibration_board_parameters.yaml")
        self.poses_config_path = Path("config/joint_poses.yaml")
        
        # Capture state
        self.captured_count = 0
        self.target_captures = 12
        self.captured_poses = []
        
        # Detection state
        self.last_frame: np.ndarray | None = None
        self.current_detection = (False, None, None)
        
        # Calibration state
        self.calibration_result: dict | None = None
        self.T_cam_gripper: np.ndarray | None = None
        
        # Jogging state
        self.jog_buttons_pressed = {}
        
        # Initialization flags
        self._initialized = False
        self._camera_init_attempted = False
        self.host = "172.16.0.2"  # Store host for reconnection
        self.robot_error: str | None = None  # Store last robot error
        self.is_auto_capturing = False  # Flag for auto-capture sequence
    
    def initialize(self, host: str) -> bool:
        """Initialize hardware connections."""
        self.host = host
        
        if self._initialized:
            return True
        
        try:
            # Camera (lazy init)
            print("Initializing Camera...")
            self.camera = RealSenseCamera(lazy=True)
            
            # Robot - attempt connection but don't fail if it doesn't work
            self.connect_robot()
            
            # Detector
            if self.board_params_path.exists():
                self.detector = CharucoDetector(self.board_params_path)
            else:
                print(f"Warning: Board params not found at {self.board_params_path}")
            
            # Prepare directories
            self.output_dir.mkdir(parents=True, exist_ok=True)
            self.calibration_dir.mkdir(parents=True, exist_ok=True)
            
            # Load existing calibration if available
            if self.calibration_file.exists():
                self.T_cam_gripper = load_calibration_result(self.calibration_file)
                with open(self.calibration_file, 'r') as f:
                    self.calibration_result = json.load(f)
                print("Loaded existing calibration.")
            
            # Count existing captures
            if self.output_dir.exists():
                existing = [d for d in self.output_dir.iterdir() 
                           if d.is_dir() and d.name.startswith("pose_")]
                self.captured_count = len(existing)
            
            self._initialized = True
            return True
            
        except Exception as e:
            print(f"Initialization Error: {e}")
            return False
    
    def connect_robot(self) -> bool:
        """
        Attempt to connect/reconnect to the robot.
        
        Always creates a fresh connection from scratch, properly cleaning up
        any existing connection first. Forces reimport of franky since it
        establishes gRPC connection at import time.
        
        Returns
        -------
        bool
            True if connection succeeded.
        """
        self.robot_error = None
        
        # Clean up existing robot connection if any
        if self.robot is not None:
            print("Cleaning up existing robot connection...")
            try:
                # Try to stop any ongoing motion
                if hasattr(self.robot, '_jogging') and self.robot._jogging:
                    self.robot.clear_jog_state()
                # Explicitly delete the underlying FrankyRobot if accessible
                if hasattr(self.robot, '_robot'):
                    del self.robot._robot
                if hasattr(self.robot, '_gripper') and self.robot._gripper is not None:
                    del self.robot._gripper
            except Exception as cleanup_err:
                print(f"Cleanup warning: {cleanup_err}")
            
            # Delete the controller and force garbage collection
            self.robot = None
            gc.collect()
            
            # Force reimport of franky - it establishes gRPC connection at import time
            print("Reloading franky module for fresh gRPC connection...")
            franky_modules = [key for key in sys.modules.keys() if key == 'franky' or key.startswith('franky.')]
            for mod_name in franky_modules:
                del sys.modules[mod_name]
            
            # Also reload the robot module that imports franky
            if 'franka_handeye.robot' in sys.modules:
                del sys.modules['franka_handeye.robot']
            
            # Force garbage collection again after module cleanup
            gc.collect()
            
            # Brief delay to allow OS to release network resources
            time.sleep(0.5)
            print("Cleanup complete.")
        
        try:
            print(f"Connecting to Robot at {self.host}...")
            # Import fresh RobotController after module cleanup
            from franka_handeye.robot import RobotController as FreshRobotController
            self.robot = FreshRobotController(self.host, dynamics_factor=0.05)
            print("Robot connected successfully!")
            return True
        except Exception as e:
            error_msg = str(e)
            # Extract the meaningful part of the error
            if "User stopped" in error_msg:
                self.robot_error = "Robot in User Stopped mode - release E-stop and reset"
            elif "command not possible" in error_msg:
                self.robot_error = "Robot not ready - check control mode"
            elif "Connection refused" in error_msg:
                self.robot_error = "Connection refused - check franky server"
            elif "timeout" in error_msg.lower():
                self.robot_error = "Connection timeout - check network"
            elif "franky is not installed" in error_msg:
                self.robot_error = "Franky server cannot be reached. Please check the franky server state"
            else:
                self.robot_error = f"Connection failed: {error_msg[:50]}"
            
            print(f"Robot connection failed: {self.robot_error}")
            self.robot = None
            return False
    
    def cleanup(self):
        """Clean up resources."""
        if self.camera:
            self.camera.stop()
            self.camera = None
        self._initialized = False


# Global state instance
state = AppState()


# =============================================================================
# UI State
# =============================================================================

ui_state = {
    'video_texture': None,
    'video_width': 800,
    'video_height': 450,
    'themes': {},
    'log_messages': [],
    'layout': {},  # Will store dynamic layout info
}


def log(message: str, level: str = "info"):
    """Add a message to the log."""
    timestamp = time.strftime("%H:%M:%S")
    prefix = {"info": "[INFO]", "success": "[OK]", "warning": "[WARN]", "error": "[ERR]"}.get(level, "[LOG]")
    formatted = f"[{timestamp}] {prefix} {message}"
    ui_state['log_messages'].append(formatted)
    # Keep only last 100 messages
    if len(ui_state['log_messages']) > 100:
        ui_state['log_messages'] = ui_state['log_messages'][-100:]
    print(formatted)


def reconnect_robot():
    """Attempt to reconnect to the robot."""
    log(f"Attempting to reconnect to robot at {state.host}...", "info")
    
    if state.connect_robot():
        log("Robot reconnected successfully!", "success")
    else:
        log(f"Reconnection failed: {state.robot_error}", "error")


# =============================================================================
# Video Processing
# =============================================================================

def update_camera_texture(frame: np.ndarray):
    """Update the camera texture with a new frame."""
    if frame is None or ui_state['video_texture'] is None:
        return
    
    # Resize to texture size (fixed buffer size)
    display_frame = cv2.resize(frame, (ui_state['video_width'], ui_state['video_height']))
    rgb_frame = cv2.cvtColor(display_frame, cv2.COLOR_BGR2RGB)
    rgb_frame = np.flipud(rgb_frame)
    normalized = rgb_frame.astype(np.float32) / 255.0
    flat = normalized.flatten()
    dpg.set_value(ui_state['video_texture'], flat)


def get_video_frame() -> np.ndarray | None:
    """Get current video frame and update detection state."""
    if not state.camera:
        return None
    
    # Try to initialize camera if needed
    if not state.camera.is_initialized and not state._camera_init_attempted:
        state._camera_init_attempted = True
        if state.camera.initialize():
            state.K, state.D = state.camera.get_intrinsics_matrix()
            log("Camera initialized", "success")
    
    frame = state.camera.get_frame()
    if frame is None:
        return None
    
    # Detection
    if state.detector and state.K is not None and state.D is not None:
        valid, rvec, tvec, corners = state.detector.detect(frame, state.K, state.D)
        if valid:
            cv2.drawFrameAxes(frame, state.K, state.D, rvec, tvec, 0.1)
            state.current_detection = (True, rvec, tvec)
        else:
            state.current_detection = (False, None, None)
    else:
        state.current_detection = (False, None, None)
    
    state.last_frame = frame.copy()
    return frame


# =============================================================================
# Robot Actions
# =============================================================================

def jog(axis: int, direction: int, btn_tag: str):
    """Start jogging motion."""
    if not state.robot:
        log("Cannot jog: Robot not connected", "error")
        return
    
    # Always stop any existing jog first
    stop_jog()
    
    try:
        state.robot.start_jog(axis, direction)
    except Exception as e:
        log(f"Jog error: {e}", "error")
        # Clear the jog state on error
        if state.robot:
            state.robot.clear_jog_state()


def stop_jog():
    """Stop jogging motion."""
    if state.robot:
        state.robot.stop_jog()


def go_home():
    """Move robot to home position."""
    if not state.robot:
        log("Cannot move: Robot not connected", "error")
        return
    
    try:
        state.robot.recover()
        state.robot.go_home(asynchronous=True)
        log("Moving to home position...", "info")
    except Exception as e:
        log(f"Home error: {e}", "error")



def _capture_impl(save_to_config=True) -> bool:
    """Internal implementation of capture logic."""
    if not state.robot or state.last_frame is None:
        log("Cannot capture: Robot not connected or no video", "error")
        return False
    
    try:
        robot_state = state.robot.get_state()
        q = robot_state['q']
        O_T_EE = robot_state['O_T_EE'].flatten().tolist()
        
        valid, rvec, tvec = state.current_detection
        
        pose_idx = state.captured_count
        pose_dir = state.output_dir / f"pose_{pose_idx:02d}"
        pose_dir.mkdir(exist_ok=True)
        
        cv2.imwrite(str(pose_dir / "image.png"), state.last_frame)
        
        data = {
            "joint_pose": q,
            "O_T_EE": O_T_EE,
            "camera_intrinsics": state.K.tolist() if state.K is not None else None,
            "dist_coeffs": state.D.tolist() if state.D is not None else None,
            "charuco_detected": valid,
        }
        
        if valid:
            data["T_cam_target_rvec"] = rvec.tolist() if hasattr(rvec, 'tolist') else rvec
            data["T_cam_target_tvec"] = tvec.tolist() if hasattr(tvec, 'tolist') else tvec
        
        with open(pose_dir / "data.json", 'w') as f:
            json.dump(data, f, indent=4, cls=NumpyEncoder)
        
        state.captured_poses.append(q)
        state.captured_count += 1
        
        if save_to_config:
            # Save joint poses config
            yaml_content = "joint_poses:\n"
            for pose in state.captured_poses:
                pose_str = "  - [" + ", ".join([f"{x:.4f}" for x in pose]) + "]\n"
                yaml_content += pose_str
            
            state.poses_config_path.parent.mkdir(exist_ok=True)
            with open(state.poses_config_path, 'w') as f:
                f.write(yaml_content)
        
        log(f"Captured pose {pose_idx} {'(ChArUco detected)' if valid else '(No ChArUco)'}", 
            "success" if valid else "warning")
        return True
        
    except Exception as e:
        log(f"Capture error: {e}", "error")
        return False


def capture_pose(sender=None, app_data=None, user_data=None):
    """Capture current pose and save data."""
    if state.is_auto_capturing:
        return
    _capture_impl(save_to_config=True)



def clear_captures():
    """Clear all captured data."""
    if state.output_dir.exists():
        shutil.rmtree(state.output_dir)
        state.output_dir.mkdir(parents=True, exist_ok=True)
    
    state.captured_count = 0
    state.captured_poses = []
    log("Cleared all captured data", "info")


def _auto_capture_thread(joint_poses):
    """Thread for auto-capture sequence."""
    state.is_auto_capturing = True
    log("Starting auto-capture sequence...", "info")
    
    try:
        # Clear existing captures first
        if state.output_dir.exists():
            shutil.rmtree(state.output_dir)
            state.output_dir.mkdir(parents=True, exist_ok=True)
        
        state.captured_count = 0
        state.captured_poses = []
        
        # Update UI safely
        dpg.set_value("capture_count", "0")
        dpg.set_value("capture_progress", 0.0)
        
        successful_captures = 0
        
        for i, pose in enumerate(joint_poses):
            if not state.is_auto_capturing: # Check for abort
                log("Auto-capture aborted by user", "warning")
                break
                
            log(f"Moving to pose {i+1}/{len(joint_poses)}...", "info")
            
            # Move robot
            try:
                state.robot.move_joints(pose)
                time.sleep(0.5) # Stabilization delay
            except Exception as e:
                log(f"Motion error at pose {i+1}: {e}", "error")
                continue
            
            # Capture
            if _capture_impl(save_to_config=False):
                successful_captures += 1
            
            # Update UI progress
            dpg.set_value("capture_count", str(state.captured_count))
            dpg.set_value("capture_progress", state.captured_count / len(joint_poses))
            
        if state.is_auto_capturing:
            log(f"Auto-capture complete: {successful_captures}/{len(joint_poses)} successful", 
                "success" if successful_captures == len(joint_poses) else "warning")
            
    except Exception as e:
        log(f"Auto-capture error: {e}", "error")
    finally:
        state.is_auto_capturing = False


def start_auto_capture(sender=None, app_data=None, user_data=None):
    """Start the auto-capture sequence."""
    if state.is_auto_capturing:
        log("Auto-capture is already running", "warning")
        return
        
    if not state.robot:
        log("Robot not connected", "error")
        return
        
    # Load poses
    if not state.poses_config_path.exists():
        log(f"Config file not found: {state.poses_config_path}", "error")
        return
        
    try:
        with open(state.poses_config_path, 'r') as f:
            config = yaml.safe_load(f)
            
        if not config or 'joint_poses' not in config:
            log("No joint_poses found in config", "error")
            return
            
        joint_poses = config['joint_poses']
        
        if len(joint_poses) < 12:
            log(f"Insufficient poses: {len(joint_poses)} (need 12+)", "error")
            return
            
        # Start thread
        threading.Thread(target=_auto_capture_thread, args=(joint_poses,), daemon=True).start()
        
    except Exception as e:
        log(f"Failed to load poses: {e}", "error")


def stop_auto_capture(sender=None, app_data=None, user_data=None):
    """Stop the auto-capture sequence."""
    if state.is_auto_capturing:
        state.is_auto_capturing = False
        log("Stopping auto-capture...", "warning")


def _show_calibration_plot_thread(T_g2b, T_cam_gripper, T_t2c):
    """Thread to show calibration plot."""
    plot_verification_preview(
        T_g2b, 
        T_cam_gripper, 
        T_t2c,
        title="Calibration Result (Last Pose)\nClose window to continue"
    )


def run_calibration():
    """Run the calibration computation."""
    try:
        R_g2b, t_g2b, R_t2c, t_t2c = load_captured_data(state.output_dir)
        
        if len(R_g2b) < 12:
            log(f"Need at least 12 valid poses for calibration (found {len(R_g2b)})", "error")
            return
        
        log(f"Running calibration with {len(R_g2b)} poses...", "info")
        
        R_cam2gripper, t_cam2gripper = compute_hand_eye_calibration(
            R_g2b, t_g2b, R_t2c, t_t2c
        )
        
        T_cam2gripper = np.eye(4)
        T_cam2gripper[:3, :3] = R_cam2gripper
        T_cam2gripper[:3, 3] = t_cam2gripper.flatten()
        
        mean_err, std_err = compute_consistency_metrics(
            R_g2b, t_g2b, R_t2c, t_t2c, T_cam2gripper
        )
        
        state.calibration_result = save_calibration_result(
            state.calibration_file, R_cam2gripper, t_cam2gripper, mean_err, std_err
        )
        state.T_cam_gripper = T_cam2gripper
        
        log(f"Calibration complete! Error: {mean_err*1000:.2f}mm +/- {std_err*1000:.2f}mm", "success")
        
        # Show 3D plot
        # Use last captured pose for visualization
        idx = -1
        T_g2b_last = np.eye(4)
        T_g2b_last[:3, :3] = R_g2b[idx]
        T_g2b_last[:3, 3] = t_g2b[idx].flatten()
        
        T_t2c_last = np.eye(4)
        T_t2c_last[:3, :3] = R_t2c[idx]
        T_t2c_last[:3, 3] = t_t2c[idx].flatten()
        
        threading.Thread(
            target=_show_calibration_plot_thread,
            args=(T_g2b_last, T_cam2gripper, T_t2c_last),
            daemon=True
        ).start()
        
    except Exception as e:
        log(f"Calibration error: {e}", "error")


import multiprocessing

def _run_plot_process(T_gripper_base_desired, T_cam_gripper, T_target_cam, queue, title=None):
    """
    Process function to run the matplotlib plot.
    Puts True in queue if proceeding, False if cancelled.
    """
    try:
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
        
        # Plot Gripper Frame (desired position)
        plot_frame(T_gripper_base_desired, "Gripper (target)")
        
        # Plot Camera Frame
        T_cam_base = T_gripper_base_desired @ T_cam_gripper
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
            T_gripper_base_desired[:3, 3], 
            T_cam_base[:3, 3], 
            T_target_base[:3, 3]
        ])
        
        center = np.mean(points, axis=0)
        radius = np.max(np.linalg.norm(points - center, axis=1)) + 0.1
        
        ax.set_xlim(center[0] - radius, center[0] + radius)
        ax.set_ylim(center[1] - radius, center[1] + radius)
        ax.set_zlim(center[2] - radius, center[2] + radius)
        
        if title:
            plt.title(title)
        else:
            plt.title("Verification Preview\nClose window to PROCEED | Press 'q' to CANCEL")
        
        # Track if user cancelled
        cancelled = [False]
        def on_key(event):
            if event.key == 'q':
                cancelled[0] = True
                plt.close()
        
        fig.canvas.mpl_connect('key_press_event', on_key)
        plt.show(block=True)
        
        queue.put(not cancelled[0])
        
        # Ensure figure is closed (though plt.show blocks until close usually)
        try:
            plt.close(fig)
        except:
            pass
            
    except Exception as e:
        print(f"Plotting error: {e}")
        queue.put(False)


def plot_verification_preview(T_gripper_base_desired, T_cam_gripper, T_target_cam, title=None):
    """
    Show a matplotlib 3D plot of the planned alignment.
    Runs in a separate PROCESS to avoid crashing the main GUI loop.
    """
    queue = multiprocessing.Queue()
    p = multiprocessing.Process(
        target=_run_plot_process, 
        args=(T_gripper_base_desired, T_cam_gripper, T_target_cam, queue, title)
    )
    p.start()
    p.join()  # Wait for process to finish (window closed)
    
    if not queue.empty():
        return queue.get()
    return False


def move_to_board_position(robot, T_gripper_base_current, T_cam_gripper, rvec, tvec, offset, target_point, position_name):
    """Move robot to a position relative to the charuco board."""
    T_gripper_base_desired = compute_alignment_pose(
        T_gripper_base_current, T_cam_gripper, rvec, tvec, offset, target_point
    )
    
    translation = T_gripper_base_desired[:3, 3].tolist()
    quaternion = R.from_matrix(T_gripper_base_desired[:3, :3]).as_quat().tolist()
    
    log(f"[ROBOT] Moving to {position_name}...", "info")
    robot.move_cartesian(translation, quaternion, asynchronous=False)
    log(f"[OK] Reached {position_name}", "success")


def run_homing_and_detection():
    """
    Helper to perform homing and detection sequence.
    Returns (valid, rvec, tvec)
    """
    # Step 1: Validate prerequisites
    if not state.robot:
        log("Cannot verify: Robot not connected", "error")
        return False, None, None
    
    # Step 2: Go to home pose
    log("[ROBOT] Moving to home pose...", "info")
    state.robot.go_home()
    log("[OK] Reached home pose", "success")
    
    # Step 3: Home and close gripper
    log("[HOME] Homing gripper...", "info")
    state.robot.home_gripper()
    log("[OK] Gripper homed", "success")
    
    log("[GRIP] Closing gripper...", "info")
    state.robot.close_gripper()
    log("[OK] Gripper closed", "success")
    
    # Step 4: Wait and detect charuco board from home position
    log("[CAM] Detecting charuco board from home position...", "info")
    time.sleep(1.0)  # Wait for robot to settle
    
    frame = state.camera.get_frame()
    if frame is None:
        log("Cannot verify: Failed to capture frame", "error")
        return False, None, None
    
    K, D = state.camera.get_intrinsics_matrix()
    valid, rvec, tvec, _ = state.detector.detect(frame, K, D)
    
    if not valid:
        log("[ERR] ChArUco board not detected! Make sure it's visible from home position.", "error")
        return False, None, None
    
    log(f"[OK] ChArUco detected at position: {tvec.flatten()}", "success")
    return True, rvec, tvec


def check_frames_visualizer_thread():
    """
    Runs the check frames sequence:
    1. Home robot & gripper
    2. Detect charuco
    3. Show plot
    """
    try:
        if state.T_cam_gripper is None:
            log("Cannot check frames: No calibration loaded", "error")
            return
        
        # Run homing and detection
        valid, rvec, tvec = run_homing_and_detection()
        if not valid:
            return
            
        # Get current robot state (should be home)
        T_gripper_base_current = state.robot.get_state()['O_T_EE']
        
        # Compute transforms for visualization
        R_target_cam, _ = cv2.Rodrigues(rvec)
        T_target_cam = np.eye(4)
        T_target_cam[:3, :3] = R_target_cam
        T_target_cam[:3, 3] = tvec.flatten()
        
        log("[PLOT] Showing frame plot...", "info")
        log("   Close window to finish", "info")
        
        # Show plot
        plot_verification_preview(
            T_gripper_base_current, 
            state.T_cam_gripper, 
            T_target_cam,
            title="Current Frame Check\nClose window to finish"
        )
        
        log("[OK] Frame check complete", "success")

    except Exception as e:
        log(f"Frame check error: {e}", "error")
        import traceback
        traceback.print_exc()


def verify_visit_corners_thread():
    """
    Runs the visit corners sequence:
    1. Checks for valid detection (from Check Frames step)
    2. Moves to center
    3. Tours corners
    4. Returns home
    """
    try:
        # Validate prerequisites
        if not state.robot:
            log("Cannot verify: Robot not connected", "error")
            return
        if state.T_cam_gripper is None:
            log("Cannot verify: No calibration loaded", "error")
            return
            
        offset = 0.06  # 6cm offset from board
        
        # Use current detection from state (updated by UI loop or previous check)
        valid, rvec, tvec = state.current_detection
        
        if not valid:
            log("[ERR] No valid ChArUco detection!", "error")
            log("   Please run 'CHECK CURRENT FRAMES' first or ensure board is visible.", "error")
            return

        # Get current robot state
        T_gripper_base_current = state.robot.get_state()['O_T_EE']
        center_point = state.detector.get_board_center()
        board_width, board_height = state.detector.board_dimensions
        
        log(f"[CALC] Computing path to board center...", "info")
        
        # Move to board center
        move_to_board_position(
            state.robot, T_gripper_base_current, state.T_cam_gripper, 
            rvec, tvec, offset, center_point, "board center"
        )
        log("[OK] Center alignment complete!", "success")
        
        # Tour corners
        log("[TARGET] Now visiting the 4 corners of the charuco board...", "info")
        corners = state.detector.get_board_corners()
        corners.append(corners[0])  # Return to first corner
        corner_names = ["Top-Left", "Top-Right", "Bottom-Right", "Bottom-Left", "Top-Left (return)"]
        
        # We need to update T_gripper_base_current after moving to center
        # Actually, move_to_board_position moves the robot, so get_state() would be updated
        # BUT compute_alignment_pose uses the detection relative to the STARTING pose
        # If we move, the detection (rvec, tvec) is relative to the Camera at the STARTING pose.
        # So we must use the ORIGINAL T_gripper_base_current (where detection happened)
        # combined with the ORIGINAL rvec/tvec.
        
        # Wait, if the robot moved, the detection in 'state.current_detection' might be from the NEW position?
        # state.current_detection is updated by update_ui -> get_video_frame loop.
        # If the robot is at Center, we might lose detection (too close?).
        # But we should use the detection we started with.
        
        # CRITICAL: state.current_detection is live. 
        # If we trust the user ran "Check Frames" (went Home) and we are still at Home, 
        # then state.current_detection corresponds to Home.
        
        # Let's grab the detection snapshot at the start of this function
        start_rvec = rvec
        start_tvec = tvec
        start_T_gripper_base = T_gripper_base_current # Robot is presumably at Home
        
        for i, (corner, corner_name) in enumerate(zip(corners, corner_names)):
            log(f"--- Corner {i+1}/5: {corner_name} ---", "info")
            move_to_board_position(
                state.robot, start_T_gripper_base, state.T_cam_gripper,
                start_rvec, start_tvec, offset, corner, f"{corner_name} corner"
            )
            time.sleep(0.3)
        
        log("[OK] Corner tour complete!", "success")
        
        # Return to home
        log("[HOME] Returning to home position...", "info")
        state.robot.go_home()
        state.robot.home_gripper()
        log("[OK] Returned to home position", "success")
        
        log("[DONE] Verification complete!", "success")
        
    except Exception as e:
        log(f"Visit error: {e}", "error")
        import traceback
        traceback.print_exc()


def check_frames_visualizer():
    """Start frame checker in a background thread."""
    thread = threading.Thread(target=check_frames_visualizer_thread, daemon=True)
    thread.start()


def start_visit_corners():
    """Start corner visit in a background thread."""
    thread = threading.Thread(target=verify_visit_corners_thread, daemon=True)
    thread.start()


# =============================================================================
# UI Construction
# =============================================================================

def create_jog_button(label: str, axis: int, sign: int, width: int = 60, height: int = 40):
    """Create a jog button - state will be checked in update loop."""
    btn_tag = f"jog_btn_{axis}_{sign}"
    dpg.add_button(label=label, tag=btn_tag, width=width, height=height)
    # Store metadata for the update loop
    if 'jog_button_metadata' not in ui_state:
        ui_state['jog_button_metadata'] = {}
    ui_state['jog_button_metadata'][btn_tag] = (axis, sign)


def on_viewport_resize():
    """Handle viewport resize - update layout dynamically."""
    vp_width = dpg.get_viewport_client_width()
    vp_height = dpg.get_viewport_client_height()
    
    # Minimum dimensions
    vp_width = max(vp_width, 1400)
    vp_height = max(vp_height, 850)
    
    # Calculate layout
    padding = 24
    gap = 24
    header_height = 60
    
    # Available content area
    content_width = vp_width - 2 * padding
    content_height = vp_height - header_height - padding
    
    # Two-column layout: 70% Camera, 30% Controls
    left_width = int(content_width * 0.70)
    right_width = content_width - left_width - gap
    
    # Video sizing - maintain 16:9 aspect ratio, fill available width
    video_w = left_width - 40
    video_h = int(video_w * 9 / 16)
    
    # Update child window sizes
    if dpg.does_item_exist("left_column"):
        dpg.configure_item("left_column", width=left_width)
    if dpg.does_item_exist("video_image"):
        dpg.configure_item("video_image", width=video_w, height=video_h)
    
    # Robot State and Log are side by side, each gets half width
    half_width = (left_width - 40) // 2
    if dpg.does_item_exist("robot_state_window"):
        dpg.configure_item("robot_state_window", width=half_width)
    if dpg.does_item_exist("log_window"):
        dpg.configure_item("log_window", width=half_width)
        if dpg.does_item_exist("log_text"):
            dpg.configure_item("log_text", wrap=half_width - 20)
    if dpg.does_item_exist("right_column"):
        dpg.configure_item("right_column", width=right_width)
    
    # Update capture tab buttons and progress bar - make them responsive
    btn_area_width = right_width - 40  # Account for padding
    
    if dpg.does_item_exist("capture_progress"):
        dpg.configure_item("capture_progress", width=btn_area_width)
    if dpg.does_item_exist("btn_auto_run"):
        dpg.configure_item("btn_auto_run", width=int(btn_area_width * 0.70))
    if dpg.does_item_exist("btn_stop"):
        dpg.configure_item("btn_stop", width=int(btn_area_width * 0.26))
    if dpg.does_item_exist("btn_capture"):
        dpg.configure_item("btn_capture", width=int((btn_area_width - 10) * 0.5))
    if dpg.does_item_exist("btn_clear"):
        dpg.configure_item("btn_clear", width=int((btn_area_width - 10) * 0.5))
        
    # Resize other full-width elements
    if dpg.does_item_exist("btn_go_home"):
        dpg.configure_item("btn_go_home", width=btn_area_width)
    if dpg.does_item_exist("btn_run_calib"):
        dpg.configure_item("btn_run_calib", width=btn_area_width)
    if dpg.does_item_exist("calib_result_window"):
        dpg.configure_item("calib_result_window", width=btn_area_width)
    if dpg.does_item_exist("verify_status_window"):
        dpg.configure_item("verify_status_window", width=btn_area_width)
    if dpg.does_item_exist("btn_check_frames"):
        dpg.configure_item("btn_check_frames", width=int((btn_area_width - 10) * 0.5))
    if dpg.does_item_exist("btn_visit_corners"):
        dpg.configure_item("btn_visit_corners", width=int((btn_area_width - 10) * 0.5))
        
        
    # Update text wrapping for description texts
    if dpg.does_item_exist("calib_desc_text"):
        dpg.configure_item("calib_desc_text", wrap=btn_area_width)
    if dpg.does_item_exist("verify_desc_text"):
        dpg.configure_item("verify_desc_text", wrap=btn_area_width)


def create_ui():
    """Create the complete DearPyGui interface."""
    dpg.create_context()
    
    # Start with a good default size for HD screens - sized to fit all content without scrolling
    dpg.create_viewport(title='Franka Hand-Eye Calibration', width=1600, height=920)
    dpg.set_viewport_min_width(1400)
    dpg.set_viewport_min_height(1040)
    
    # Setup theme
    ui_state['themes'] = setup_theme()
    
    # Register font
    with dpg.font_registry():
        # Try to use a nice font if available, otherwise use default
        default_font = dpg.add_font(
            "/usr/share/fonts/truetype/dejavu/DejaVuSansMono.ttf", 15,
        ) if os.path.exists("/usr/share/fonts/truetype/dejavu/DejaVuSansMono.ttf") else None
        if default_font:
            dpg.bind_font(default_font)
    
    # Register texture for video display
    with dpg.texture_registry(show=False):
        ui_state['video_texture'] = dpg.add_raw_texture(
            width=ui_state['video_width'],
            height=ui_state['video_height'],
            default_value=np.zeros((ui_state['video_height'], ui_state['video_width'], 3), dtype=np.float32).flatten(),
            format=dpg.mvFormat_Float_rgb
        )
    
    # Register resize handler
    with dpg.item_handler_registry(tag="viewport_resize_handler") as handler:
        dpg.add_item_resize_handler(callback=lambda s, a, u: on_viewport_resize())
    
    # Main window
    with dpg.window(label="Main", tag="primary_window", no_title_bar=True, no_move=True, no_scrollbar=True):
        
        # ===== HEADER =====
        with dpg.group(horizontal=True):
            dpg.add_text("FRANKA HAND-EYE CALIBRATION", color=Theme.ACCENT_PRIMARY)
            dpg.add_spacer(width=30)
            dpg.add_text("Robot:", color=Theme.TEXT_SECONDARY)
            dpg.add_text("Disconnected", tag="robot_status", color=Theme.DISCONNECTED)
            dpg.add_spacer(width=10)
            reconnect_btn = dpg.add_button(
                label="Reconnect", 
                tag="reconnect_btn",
                callback=reconnect_robot, 
                width=90, 
                height=24
            )
            dpg.add_spacer(width=30)
            dpg.add_text("Camera:", color=Theme.TEXT_SECONDARY)
            dpg.add_text("Disconnected", tag="camera_status", color=Theme.DISCONNECTED)
        
        # Robot error message (hidden by default)
        dpg.add_text("", tag="robot_error_msg", color=Theme.ACCENT_WARNING, show=False)
        
        dpg.add_spacer(height=8)
        dpg.add_separator()
        dpg.add_spacer(height=8)
        
        # ===== MAIN CONTENT =====
        with dpg.group(horizontal=True):
            
            # ===== LEFT COLUMN: Video & Detection =====
            with dpg.child_window(tag="left_column", width=950, height=-1, border=False):
                dpg.add_text("CAMERA FEED", color=Theme.TEXT_SECONDARY)
                dpg.add_spacer(height=4)
                
                # Video
                dpg.add_image(ui_state['video_texture'], tag="video_image", width=900, height=506)
                
                dpg.add_spacer(height=8)
                
                # Detection status
                with dpg.group(horizontal=True):
                    dpg.add_text("ChArUco:", color=Theme.TEXT_SECONDARY)
                    dpg.add_text("NOT DETECTED", tag="detection_status", color=Theme.NOT_DETECTING)
                
                dpg.add_spacer(height=12)
                
                # Robot State and Log side by side
                with dpg.group(horizontal=True):
                    # Robot State
                    with dpg.group():
                        dpg.add_text("ROBOT STATE", color=Theme.TEXT_SECONDARY)
                        dpg.add_spacer(height=4)
                        with dpg.child_window(tag="robot_state_window", width=440, height=170, border=True, no_scrollbar=True):
                            dpg.add_text("Position (xyz):", color=Theme.TEXT_MUTED)
                            dpg.add_text("---", tag="position_display")
                            dpg.add_spacer(height=6)
                            dpg.add_text("Joint angles:", color=Theme.TEXT_MUTED)
                            dpg.add_text("---", tag="joints_display")
                    
                    dpg.add_spacer(width=16)
                    
                    # Log window
                    with dpg.group():
                        dpg.add_text("LOG", color=Theme.TEXT_SECONDARY)
                        dpg.add_spacer(height=4)
                        with dpg.child_window(tag="log_window", width=440, height=170, border=True):
                            dpg.add_text("Application started.", tag="log_text", wrap=420)
            
            dpg.add_spacer(width=20)
            
            # ===== RIGHT COLUMN: Controls =====
            with dpg.child_window(tag="right_column", width=400, height=-1, border=False, no_scrollbar=True):
                
                # Tab bar for different modes
                with dpg.tab_bar(tag="main_tab_bar"):
                    
                    # ===== CAPTURE TAB =====
                    with dpg.tab(label="CAPTURE"):
                        dpg.add_spacer(height=8)
                    
                        # Capture status
                        dpg.add_text("CAPTURE STATUS", color=Theme.TEXT_SECONDARY)
                        dpg.add_spacer(height=6)
                        
                        with dpg.group(horizontal=True):
                            dpg.add_text("Captured:", color=Theme.TEXT_MUTED)
                            dpg.add_text("0", tag="capture_count")
                            dpg.add_text("/", color=Theme.TEXT_MUTED)
                            dpg.add_text(str(state.target_captures), tag="target_count")
                        
                        dpg.add_progress_bar(default_value=0.0, tag="capture_progress", width=-1)
                        
                        dpg.add_spacer(height=12)
                        
                        # Capture buttons
                        with dpg.group(horizontal=True, tag="capture_btn_row1"):
                            btn = dpg.add_button(label="AUTO RUN", tag="btn_auto_run", callback=start_auto_capture, width=250, height=44)
                            dpg.bind_item_theme(btn, ui_state['themes']['success'])
                            
                            dpg.add_spacer(width=10)
                            
                            btn = dpg.add_button(label="STOP", tag="btn_stop", callback=stop_auto_capture, width=100, height=44)
                            dpg.bind_item_theme(btn, ui_state['themes']['danger'])
                            
                        dpg.add_spacer(height=10)

                        with dpg.group(horizontal=True, tag="capture_btn_row2"):
                            btn = dpg.add_button(label="CAPTURE", tag="btn_capture", callback=capture_pose, width=175, height=44)
                            dpg.bind_item_theme(btn, ui_state['themes']['accent'])
                            
                            dpg.add_spacer(width=10)
                            
                            btn = dpg.add_button(label="CLEAR ALL", tag="btn_clear", callback=clear_captures, width=175, height=44)
                            dpg.bind_item_theme(btn, ui_state['themes']['danger'])
                        
                        dpg.add_spacer(height=16)
                        dpg.add_separator()
                        dpg.add_spacer(height=12)
                        
                        # Jog controls
                        dpg.add_text("JOG CONTROLS", color=Theme.TEXT_SECONDARY)
                        dpg.add_text("Hold buttons to move robot", color=Theme.TEXT_MUTED)
                        dpg.add_spacer(height=10)
                        
                        with dpg.group(horizontal=True):
                            # Translation Column
                            with dpg.group():
                                dpg.add_text("Translation", color=Theme.ACCENT_PRIMARY)
                                dpg.add_spacer(height=4)
                                # X
                                with dpg.group(horizontal=True):
                                    dpg.add_text("X:", color=Theme.TEXT_MUTED)
                                    dpg.add_spacer(width=4)
                                    create_jog_button("-", 0, -1, width=50, height=40)
                                    create_jog_button("+", 0, 1, width=50, height=40)
                                # Y
                                with dpg.group(horizontal=True):
                                    dpg.add_text("Y:", color=Theme.TEXT_MUTED)
                                    dpg.add_spacer(width=4)
                                    create_jog_button("-", 1, -1, width=50, height=40)
                                    create_jog_button("+", 1, 1, width=50, height=40)
                                # Z
                                with dpg.group(horizontal=True):
                                    dpg.add_text("Z:", color=Theme.TEXT_MUTED)
                                    dpg.add_spacer(width=4)
                                    create_jog_button("-", 2, -1, width=50, height=40)
                                    create_jog_button("+", 2, 1, width=50, height=40)
                            
                            dpg.add_spacer(width=30)
                            
                            # Rotation Column
                            with dpg.group():
                                dpg.add_text("Rotation", color=Theme.ACCENT_SECONDARY)
                                dpg.add_spacer(height=4)
                                # Rx
                                with dpg.group(horizontal=True):
                                    dpg.add_text("Rx:", color=Theme.TEXT_MUTED)
                                    dpg.add_spacer(width=4)
                                    create_jog_button("-", 3, -1, width=50, height=40)
                                    create_jog_button("+", 3, 1, width=50, height=40)
                                # Ry
                                with dpg.group(horizontal=True):
                                    dpg.add_text("Ry:", color=Theme.TEXT_MUTED)
                                    dpg.add_spacer(width=4)
                                    create_jog_button("-", 4, -1, width=50, height=40)
                                    create_jog_button("+", 4, 1, width=50, height=40)
                                # Rz
                                with dpg.group(horizontal=True):
                                    dpg.add_text("Rz:", color=Theme.TEXT_MUTED)
                                    dpg.add_spacer(width=4)
                                    create_jog_button("-", 5, -1, width=50, height=40)
                                    create_jog_button("+", 5, 1, width=50, height=40)
                        
                        dpg.add_spacer(height=12)
                        
                        # Home button
                        btn = dpg.add_button(label="GO HOME", tag="btn_go_home", callback=go_home, width=360, height=40)
                        dpg.bind_item_theme(btn, ui_state['themes']['success'])
                    
                    # ===== CALIBRATE TAB =====
                    with dpg.tab(label="CALIBRATE"):
                        dpg.add_spacer(height=8)
                        
                        dpg.add_text("CALIBRATION", color=Theme.TEXT_SECONDARY)
                        dpg.add_spacer(height=8)
                        
                        dpg.add_text(
                            "Compute hand-eye calibration.\n"
                            "Requires 12+ poses with ChArUco.",
                            tag="calib_desc_text", color=Theme.TEXT_MUTED, wrap=350
                        )
                        
                        dpg.add_spacer(height=16)
                        
                        btn = dpg.add_button(label="RUN CALIBRATION", tag="btn_run_calib", callback=run_calibration, width=360, height=44)
                        dpg.bind_item_theme(btn, ui_state['themes']['accent'])
                        
                        dpg.add_spacer(height=16)
                        dpg.add_separator()
                        dpg.add_spacer(height=12)
                        
                        # Calibration results display
                        dpg.add_text("CALIBRATION RESULT", color=Theme.TEXT_SECONDARY)
                        dpg.add_spacer(height=8)
                        
                        with dpg.child_window(tag="calib_result_window", width=360, height=-1, border=True, no_scrollbar=True):
                            dpg.add_text("Status:", color=Theme.TEXT_MUTED)
                            dpg.add_text("No calibration loaded", tag="calib_status")
                            dpg.add_spacer(height=10)
                            dpg.add_text("Translation (xyz):", color=Theme.TEXT_MUTED)
                            dpg.add_text("---", tag="calib_translation")
                            dpg.add_spacer(height=10)
                            dpg.add_text("Quaternion (xyzw):", color=Theme.TEXT_MUTED)
                            dpg.add_text("---", tag="calib_quaternion")
                            dpg.add_spacer(height=10)
                            dpg.add_text("Consistency Error:", color=Theme.TEXT_MUTED)
                            dpg.add_text("---", tag="calib_error")
                    
                    # ===== VERIFY TAB =====
                    with dpg.tab(label="VERIFY"):
                        dpg.add_spacer(height=8)
                        
                        dpg.add_text("VERIFICATION", color=Theme.TEXT_SECONDARY)
                        dpg.add_spacer(height=8)
                        
                        dpg.add_text(
                            "Verify calibration with ChArUco board.\n"
                            "- CHECK: Home - Detect - Plot\n"
                            "- VISIT: Tour board corners",
                            tag="verify_desc_text", color=Theme.TEXT_MUTED, wrap=350
                        )
                        
                        dpg.add_spacer(height=12)
                        
                        with dpg.group(horizontal=True, tag="verify_btn_row"):
                            btn_check = dpg.add_button(label="CHECK FRAMES", tag="btn_check_frames", callback=check_frames_visualizer, width=175, height=44)
                            dpg.bind_item_theme(btn_check, ui_state['themes']['accent'])
                            
                            dpg.add_spacer(width=10)
                            
                            btn_visit = dpg.add_button(label="VISIT CORNERS", tag="btn_visit_corners", callback=start_visit_corners, width=175, height=44)
                            dpg.bind_item_theme(btn_visit, ui_state['themes']['accent'])
                        
                        dpg.add_spacer(height=16)
                        dpg.add_separator()
                        dpg.add_spacer(height=12)
                        
                        # Verification status
                        dpg.add_text("VERIFICATION STATUS", color=Theme.TEXT_SECONDARY)
                        dpg.add_spacer(height=8)
                        
                        with dpg.child_window(tag="verify_status_window", width=360, height=-1, border=True, no_scrollbar=True):
                            dpg.add_text("Calibration:", color=Theme.TEXT_MUTED)
                            dpg.add_text("Not loaded", tag="verify_calib_status", color=Theme.ACCENT_WARNING)
                            dpg.add_spacer(height=8)
                            dpg.add_text("ChArUco:", color=Theme.TEXT_MUTED)
                            dpg.add_text("Not detected", tag="verify_detection_status", color=Theme.ACCENT_WARNING)
                            dpg.add_spacer(height=8)
                            dpg.add_text("Board Position:", color=Theme.TEXT_MUTED)
                            dpg.add_text("---", tag="verify_board_pos")
    
    # Bind resize handler to primary window
    dpg.bind_item_handler_registry("primary_window", "viewport_resize_handler")
    
    dpg.setup_dearpygui()
    dpg.show_viewport()
    dpg.set_primary_window("primary_window", True)
    
    # Initial layout update
    on_viewport_resize()


def update_ui():
    """Update UI elements each frame."""
    # Update video
    frame = get_video_frame()
    if frame is not None:
        update_camera_texture(frame)
    
    # Update detection status
    is_detected, rvec, tvec = state.current_detection
    if is_detected:
        dpg.set_value("detection_status", "DETECTED")
        dpg.configure_item("detection_status", color=Theme.DETECTING)
        dpg.set_value("verify_detection_status", "Detected")
        dpg.configure_item("verify_detection_status", color=Theme.ACCENT_SUCCESS)
        if tvec is not None:
            pos_str = f"[{tvec[0][0]:.3f}, {tvec[1][0]:.3f}, {tvec[2][0]:.3f}]"
            dpg.set_value("verify_board_pos", pos_str)
    else:
        dpg.set_value("detection_status", "NOT DETECTED")
        dpg.configure_item("detection_status", color=Theme.NOT_DETECTING)
        dpg.set_value("verify_detection_status", "Not detected")
        dpg.configure_item("verify_detection_status", color=Theme.ACCENT_WARNING)
        dpg.set_value("verify_board_pos", "---")
    
    # Update capture count
    dpg.set_value("capture_count", str(state.captured_count))
    dpg.set_value("capture_progress", state.captured_count / max(1, state.target_captures))
    
    # Update robot state and reconnect button visibility
    if state.robot:
        try:
            robot_state = state.robot.get_state()
            pos = robot_state['position']
            q = robot_state['q']
            
            dpg.set_value("position_display", f"[{pos[0]:.4f}, {pos[1]:.4f}, {pos[2]:.4f}]")
            q_str = ", ".join([f"{x:.2f}" for x in q])
            dpg.set_value("joints_display", f"[{q_str}]")
            
            dpg.set_value("robot_status", "Connected")
            dpg.configure_item("robot_status", color=Theme.CONNECTED)
            dpg.configure_item("reconnect_btn", show=False)
            dpg.configure_item("robot_error_msg", show=False)
        except Exception as e:
            dpg.set_value("robot_status", "Error")
            dpg.configure_item("robot_status", color=Theme.ACCENT_WARNING)
            dpg.configure_item("reconnect_btn", show=True)
            dpg.set_value("robot_error_msg", f"  [WARN] Communication error - try reconnecting")
            dpg.configure_item("robot_error_msg", show=True)
    else:
        dpg.set_value("robot_status", "Disconnected")
        dpg.configure_item("robot_status", color=Theme.DISCONNECTED)
        dpg.configure_item("reconnect_btn", show=True)
        dpg.set_value("position_display", "--- (robot disconnected)")
        dpg.set_value("joints_display", "--- (robot disconnected)")
        # Show error message if we have one
        if state.robot_error:
            dpg.set_value("robot_error_msg", f"  [WARN] {state.robot_error}")
            dpg.configure_item("robot_error_msg", show=True)
        else:
            dpg.configure_item("robot_error_msg", show=False)
    
    # Update camera status
    if state.camera and state.camera.is_initialized:
        dpg.set_value("camera_status", "Connected")
        dpg.configure_item("camera_status", color=Theme.CONNECTED)
    else:
        dpg.set_value("camera_status", "Disconnected")
        dpg.configure_item("camera_status", color=Theme.DISCONNECTED)
    
    # Update calibration display
    if state.calibration_result:
        dpg.set_value("calib_status", "Loaded")
        dpg.configure_item("calib_status", color=Theme.ACCENT_SUCCESS)
        
        xyz = state.calibration_result.get('xyz', [0, 0, 0])
        quat = state.calibration_result.get('quaternion_xyzw', [0, 0, 0, 1])
        mean_err = state.calibration_result.get('consistency_error_mean', 0)
        std_err = state.calibration_result.get('consistency_error_std', 0)
        
        dpg.set_value("calib_translation", f"[{xyz[0]:.4f}, {xyz[1]:.4f}, {xyz[2]:.4f}]")
        dpg.set_value("calib_quaternion", f"[{quat[0]:.4f}, {quat[1]:.4f}, {quat[2]:.4f}, {quat[3]:.4f}]")
        dpg.set_value("calib_error", f"{mean_err*1000:.2f}mm +/- {std_err*1000:.2f}mm")
        
        dpg.set_value("verify_calib_status", "Loaded")
        dpg.configure_item("verify_calib_status", color=Theme.ACCENT_SUCCESS)
    else:
        dpg.set_value("calib_status", "No calibration")
        dpg.configure_item("calib_status", color=Theme.ACCENT_WARNING)
        dpg.set_value("verify_calib_status", "Not loaded")
        dpg.configure_item("verify_calib_status", color=Theme.ACCENT_WARNING)
    
    # Update log display
    if ui_state['log_messages']:
        log_text = "\n".join(ui_state['log_messages'][-8:])
        dpg.set_value("log_text", log_text)
    
    # Handle jog buttons - simple and robust
    button_metadata = ui_state.get('jog_button_metadata', {})
    mouse_down = dpg.is_mouse_button_down(dpg.mvMouseButton_Left)
    
    # Find if any jog button is currently pressed (hovered + mouse down)
    active_button = None
    if mouse_down:
        for btn_tag, (axis, sign) in button_metadata.items():
            if dpg.does_item_exist(btn_tag) and dpg.is_item_hovered(btn_tag):
                active_button = (btn_tag, axis, sign)
                break
    
    current_jog = ui_state.get('current_jog_button', None)
    
    if active_button:
        btn_tag, axis, sign = active_button
        # A button is being held
        if current_jog != btn_tag:
            # New button or different button - start jogging
            if state.robot:
                # Stop existing jog if any
                if state.robot.is_jogging:
                    state.robot.stop_jog()
                    time.sleep(0.02)
                
                # Start new jog
                try:
                    state.robot.start_jog(axis, sign)
                    ui_state['current_jog_button'] = btn_tag
                except Exception as e:
                    log(f"Jog error: {e}", "error")
                    state.robot.clear_jog_state()
                    ui_state['current_jog_button'] = None
    else:
        # No button is being held
        if current_jog is not None:
            # Button was released - stop jogging
            if state.robot:
                state.robot.stop_jog()
            ui_state['current_jog_button'] = None


# =============================================================================
# Main Entry Point
# =============================================================================

def main():
    parser = argparse.ArgumentParser(
        description="Franka Hand-Eye Calibration Application"
    )
    parser.add_argument("--host", default="172.16.0.2", help="Robot FCI IP address")
    args, _ = parser.parse_known_args()
    
    print("=" * 60)
    print("  FRANKA HAND-EYE CALIBRATION APPLICATION")
    print("=" * 60)
    print(f"  Robot Host: {args.host}")
    print("=" * 60)
    
    # Initialize
    if state.initialize(args.host):
        log("System initialized successfully", "success")
    else:
        log("System initialization failed - some features may be unavailable", "warning")
    
    # Create UI
    create_ui()
    
    # Main loop
    while dpg.is_dearpygui_running():
        update_ui()
        dpg.render_dearpygui_frame()
    
    # Cleanup
    state.cleanup()
    dpg.destroy_context()
    print("\nApplication closed.")


if __name__ == "__main__":
    main()

