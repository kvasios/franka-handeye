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

import os
import sys
import time
import argparse
import json
import shutil
import threading
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
    
    def initialize(self, host: str) -> bool:
        """Initialize hardware connections."""
        if self._initialized:
            return True
        
        try:
            # Camera (lazy init)
            print("Initializing Camera...")
            self.camera = RealSenseCamera(lazy=True)
            
            # Robot
            print(f"Connecting to Robot at {host}...")
            self.robot = RobotController(host, dynamics_factor=0.05)
            
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
    'video_width': 640,
    'video_height': 360,
    'themes': {},
    'log_messages': [],
}


def log(message: str, level: str = "info"):
    """Add a message to the log."""
    timestamp = time.strftime("%H:%M:%S")
    prefix = {"info": "ℹ️", "success": "✓", "warning": "⚠️", "error": "✗"}.get(level, "•")
    formatted = f"[{timestamp}] {prefix} {message}"
    ui_state['log_messages'].append(formatted)
    # Keep only last 100 messages
    if len(ui_state['log_messages']) > 100:
        ui_state['log_messages'] = ui_state['log_messages'][-100:]
    print(formatted)


# =============================================================================
# Video Processing
# =============================================================================

def update_camera_texture(frame: np.ndarray):
    """Update the camera texture with a new frame."""
    if frame is None or ui_state['video_texture'] is None:
        return
    
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
    
    if state.robot.is_jogging:
        stop_jog()
    
    try:
        state.robot.start_jog(axis, direction)
    except Exception as e:
        log(f"Jog error: {e}", "error")


def stop_jog():
    """Stop jogging motion."""
    if state.robot and state.robot.is_jogging:
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


def capture_pose():
    """Capture current pose and save data."""
    if not state.robot or state.last_frame is None:
        log("Cannot capture: Robot not connected or no video", "error")
        return
    
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
        
    except Exception as e:
        log(f"Capture error: {e}", "error")


def clear_captures():
    """Clear all captured data."""
    if state.output_dir.exists():
        shutil.rmtree(state.output_dir)
        state.output_dir.mkdir(parents=True, exist_ok=True)
    
    state.captured_count = 0
    state.captured_poses = []
    log("Cleared all captured data", "info")


def run_calibration():
    """Run the calibration computation."""
    try:
        R_g2b, t_g2b, R_t2c, t_t2c = load_captured_data(state.output_dir)
        
        if len(R_g2b) < 3:
            log("Need at least 3 valid poses for calibration", "error")
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
        
        log(f"Calibration complete! Error: {mean_err*1000:.2f}mm ± {std_err*1000:.2f}mm", "success")
        
    except Exception as e:
        log(f"Calibration error: {e}", "error")


def verify_calibration():
    """Move to align with detected charuco board."""
    if not state.robot or not state.T_cam_gripper is not None:
        log("Cannot verify: Robot not connected or no calibration", "error")
        return
    
    valid, rvec, tvec = state.current_detection
    if not valid:
        log("Cannot verify: ChArUco board not detected", "error")
        return
    
    try:
        T_gripper_base = state.robot.get_state()['O_T_EE']
        center = state.detector.get_board_center() if state.detector else [0, 0, 0]
        
        T_desired = compute_alignment_pose(
            T_gripper_base, state.T_cam_gripper, rvec, tvec, 
            offset_distance=0.06, target_point_in_board=center
        )
        
        translation = T_desired[:3, 3].tolist()
        quaternion = R.from_matrix(T_desired[:3, :3]).as_quat().tolist()
        
        log("Moving to align with board center...", "info")
        state.robot.move_cartesian(translation, quaternion, asynchronous=True)
        
    except Exception as e:
        log(f"Verification error: {e}", "error")


# =============================================================================
# UI Construction
# =============================================================================

def create_jog_button(label: str, axis: int, sign: int, width: int = 60, height: int = 40):
    """Create a jog button with proper state tracking."""
    btn_tag = f"jog_btn_{axis}_{sign}"
    dpg.add_button(label=label, tag=btn_tag, width=width, height=height)
    ui_state['jog_buttons_pressed'] = ui_state.get('jog_buttons_pressed', {})
    ui_state['jog_buttons_pressed'][btn_tag] = False


def create_ui():
    """Create the complete DearPyGui interface."""
    dpg.create_context()
    dpg.create_viewport(title='Franka Hand-Eye Calibration', width=1400, height=900)
    
    # Setup theme
    ui_state['themes'] = setup_theme()
    
    # Register font
    with dpg.font_registry():
        # Try to use a nice font if available, otherwise use default
        default_font = dpg.add_font(
            "/usr/share/fonts/truetype/dejavu/DejaVuSansMono.ttf", 14,
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
    
    # Main window
    with dpg.window(label="Main", tag="primary_window", no_title_bar=True, no_move=True):
        
        # ===== HEADER =====
        with dpg.group(horizontal=True):
            dpg.add_text("FRANKA HAND-EYE CALIBRATION", color=Theme.ACCENT_PRIMARY)
            dpg.add_spacer(width=20)
            dpg.add_text("Robot:", color=Theme.TEXT_SECONDARY)
            dpg.add_text("Disconnected", tag="robot_status", color=Theme.DISCONNECTED)
            dpg.add_spacer(width=20)
            dpg.add_text("Camera:", color=Theme.TEXT_SECONDARY)
            dpg.add_text("Disconnected", tag="camera_status", color=Theme.DISCONNECTED)
        
        dpg.add_spacer(height=8)
        dpg.add_separator()
        dpg.add_spacer(height=8)
        
        # ===== MAIN CONTENT =====
        with dpg.group(horizontal=True):
            
            # ===== LEFT COLUMN: Video & Detection =====
            with dpg.child_window(width=680, height=680, border=False):
                dpg.add_text("CAMERA FEED", color=Theme.TEXT_SECONDARY)
                dpg.add_spacer(height=4)
                
                with dpg.child_window(width=660, height=380, border=True):
                    dpg.add_image(ui_state['video_texture'], width=640, height=360)
                
                dpg.add_spacer(height=8)
                
                # Detection status
                with dpg.group(horizontal=True):
                    dpg.add_text("ChArUco:", color=Theme.TEXT_SECONDARY)
                    dpg.add_text("NOT DETECTED", tag="detection_status", color=Theme.NOT_DETECTING)
                
                dpg.add_spacer(height=16)
                
                # Robot position display
                dpg.add_text("ROBOT STATE", color=Theme.TEXT_SECONDARY)
                dpg.add_spacer(height=4)
                
                with dpg.child_window(width=660, height=100, border=True):
                    dpg.add_text("Position (xyz):", color=Theme.TEXT_MUTED)
                    dpg.add_text("---", tag="position_display")
                    dpg.add_spacer(height=4)
                    dpg.add_text("Joint angles:", color=Theme.TEXT_MUTED)
                    dpg.add_text("---", tag="joints_display")
                
                dpg.add_spacer(height=16)
                
                # Log window
                dpg.add_text("LOG", color=Theme.TEXT_SECONDARY)
                dpg.add_spacer(height=4)
                
                with dpg.child_window(width=660, height=120, border=True, tag="log_window"):
                    dpg.add_text("Application started.", tag="log_text", wrap=640)
            
            dpg.add_spacer(width=16)
            
            # ===== RIGHT COLUMN: Controls =====
            with dpg.child_window(width=660, height=680, border=False):
                
                # Tab bar for different modes
                with dpg.tab_bar():
                    
                    # ===== CAPTURE TAB =====
                    with dpg.tab(label="  CAPTURE  "):
                        dpg.add_spacer(height=8)
                        
                        # Capture status
                        dpg.add_text("CAPTURE STATUS", color=Theme.TEXT_SECONDARY)
                        dpg.add_spacer(height=8)
                        
                        with dpg.group(horizontal=True):
                            dpg.add_text("Captured:", color=Theme.TEXT_MUTED)
                            dpg.add_text("0", tag="capture_count")
                            dpg.add_text("/", color=Theme.TEXT_MUTED)
                            dpg.add_text(str(state.target_captures), tag="target_count")
                        
                        dpg.add_progress_bar(default_value=0.0, tag="capture_progress", width=620)
                        
                        dpg.add_spacer(height=16)
                        
                        # Capture buttons
                        with dpg.group(horizontal=True):
                            btn = dpg.add_button(label="CAPTURE POSE", callback=capture_pose, width=300, height=50)
                            dpg.bind_item_theme(btn, ui_state['themes']['accent'])
                            
                            dpg.add_spacer(width=16)
                            
                            btn = dpg.add_button(label="CLEAR ALL", callback=clear_captures, width=300, height=50)
                            dpg.bind_item_theme(btn, ui_state['themes']['danger'])
                        
                        dpg.add_spacer(height=24)
                        dpg.add_separator()
                        dpg.add_spacer(height=16)
                        
                        # Jog controls
                        dpg.add_text("JOG CONTROLS", color=Theme.TEXT_SECONDARY)
                        dpg.add_text("Hold buttons to move robot", color=Theme.TEXT_MUTED)
                        dpg.add_spacer(height=12)
                        
                        # Translation controls
                        with dpg.group(horizontal=True):
                            with dpg.group():
                                dpg.add_text("Translation", color=Theme.ACCENT_PRIMARY)
                                dpg.add_spacer(height=4)
                                
                                with dpg.group(horizontal=True):
                                    dpg.add_text("X:", color=Theme.TEXT_MUTED, indent=4)
                                    create_jog_button("−", 0, -1)
                                    create_jog_button("+", 0, 1)
                                    dpg.add_spacer(width=20)
                                    dpg.add_text("Y:", color=Theme.TEXT_MUTED)
                                    create_jog_button("−", 1, -1)
                                    create_jog_button("+", 1, 1)
                                    dpg.add_spacer(width=20)
                                    dpg.add_text("Z:", color=Theme.TEXT_MUTED)
                                    create_jog_button("−", 2, -1)
                                    create_jog_button("+", 2, 1)
                        
                        dpg.add_spacer(height=12)
                        
                        # Rotation controls
                        with dpg.group(horizontal=True):
                            with dpg.group():
                                dpg.add_text("Rotation", color=Theme.ACCENT_SECONDARY)
                                dpg.add_spacer(height=4)
                                
                                with dpg.group(horizontal=True):
                                    dpg.add_text("Rx:", color=Theme.TEXT_MUTED)
                                    create_jog_button("−", 3, -1)
                                    create_jog_button("+", 3, 1)
                                    dpg.add_spacer(width=16)
                                    dpg.add_text("Ry:", color=Theme.TEXT_MUTED)
                                    create_jog_button("−", 4, -1)
                                    create_jog_button("+", 4, 1)
                                    dpg.add_spacer(width=16)
                                    dpg.add_text("Rz:", color=Theme.TEXT_MUTED)
                                    create_jog_button("−", 5, -1)
                                    create_jog_button("+", 5, 1)
                        
                        dpg.add_spacer(height=24)
                        
                        # Home button
                        btn = dpg.add_button(label="GO HOME", callback=go_home, width=620, height=40)
                        dpg.bind_item_theme(btn, ui_state['themes']['success'])
                    
                    # ===== CALIBRATE TAB =====
                    with dpg.tab(label="  CALIBRATE  "):
                        dpg.add_spacer(height=8)
                        
                        dpg.add_text("CALIBRATION", color=Theme.TEXT_SECONDARY)
                        dpg.add_spacer(height=8)
                        
                        dpg.add_text(
                            "Compute hand-eye calibration from captured poses.\n"
                            "Requires at least 3 poses with ChArUco detection.",
                            color=Theme.TEXT_MUTED, wrap=600
                        )
                        
                        dpg.add_spacer(height=16)
                        
                        btn = dpg.add_button(label="RUN CALIBRATION", callback=run_calibration, width=620, height=50)
                        dpg.bind_item_theme(btn, ui_state['themes']['accent'])
                        
                        dpg.add_spacer(height=24)
                        dpg.add_separator()
                        dpg.add_spacer(height=16)
                        
                        # Calibration results display
                        dpg.add_text("CALIBRATION RESULT", color=Theme.TEXT_SECONDARY)
                        dpg.add_spacer(height=8)
                        
                        with dpg.child_window(width=620, height=200, border=True):
                            dpg.add_text("Status:", color=Theme.TEXT_MUTED)
                            dpg.add_text("No calibration loaded", tag="calib_status")
                            dpg.add_spacer(height=8)
                            
                            dpg.add_text("Translation (xyz):", color=Theme.TEXT_MUTED)
                            dpg.add_text("---", tag="calib_translation")
                            dpg.add_spacer(height=4)
                            
                            dpg.add_text("Quaternion (xyzw):", color=Theme.TEXT_MUTED)
                            dpg.add_text("---", tag="calib_quaternion")
                            dpg.add_spacer(height=4)
                            
                            dpg.add_text("Consistency Error:", color=Theme.TEXT_MUTED)
                            dpg.add_text("---", tag="calib_error")
                    
                    # ===== VERIFY TAB =====
                    with dpg.tab(label="  VERIFY  "):
                        dpg.add_spacer(height=8)
                        
                        dpg.add_text("VERIFICATION", color=Theme.TEXT_SECONDARY)
                        dpg.add_spacer(height=8)
                        
                        dpg.add_text(
                            "Verify calibration by moving the robot to align\n"
                            "with the detected ChArUco board.\n\n"
                            "Requirements:\n"
                            "• Valid calibration loaded\n"
                            "• ChArUco board visible in camera",
                            color=Theme.TEXT_MUTED, wrap=600
                        )
                        
                        dpg.add_spacer(height=16)
                        
                        btn = dpg.add_button(label="ALIGN WITH BOARD", callback=verify_calibration, width=620, height=50)
                        dpg.bind_item_theme(btn, ui_state['themes']['accent'])
                        
                        dpg.add_spacer(height=24)
                        dpg.add_separator()
                        dpg.add_spacer(height=16)
                        
                        # Verification info
                        dpg.add_text("VERIFICATION STATUS", color=Theme.TEXT_SECONDARY)
                        dpg.add_spacer(height=8)
                        
                        with dpg.child_window(width=620, height=150, border=True):
                            dpg.add_text("Calibration:", color=Theme.TEXT_MUTED)
                            dpg.add_text("Not loaded", tag="verify_calib_status", color=Theme.ACCENT_WARNING)
                            dpg.add_spacer(height=8)
                            
                            dpg.add_text("ChArUco Detection:", color=Theme.TEXT_MUTED)
                            dpg.add_text("Not detected", tag="verify_detection_status", color=Theme.ACCENT_WARNING)
                            dpg.add_spacer(height=8)
                            
                            dpg.add_text("Board Position (camera frame):", color=Theme.TEXT_MUTED)
                            dpg.add_text("---", tag="verify_board_pos")
    
    dpg.setup_dearpygui()
    dpg.show_viewport()
    dpg.set_primary_window("primary_window", True)


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
    
    # Update robot state
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
        except:
            dpg.set_value("robot_status", "Error")
            dpg.configure_item("robot_status", color=Theme.DISCONNECTED)
    else:
        dpg.set_value("robot_status", "Disconnected")
        dpg.configure_item("robot_status", color=Theme.DISCONNECTED)
    
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
        dpg.set_value("calib_error", f"{mean_err*1000:.2f}mm ± {std_err*1000:.2f}mm")
        
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
    
    # Handle jog button states
    jog_buttons = ui_state.get('jog_buttons_pressed', {})
    any_pressed = False
    
    for btn_tag in jog_buttons.keys():
        if dpg.does_item_exist(btn_tag):
            is_hovered = dpg.is_item_hovered(btn_tag)
            is_mouse_down = dpg.is_mouse_button_down(dpg.mvMouseButton_Left)
            button_held = is_hovered and is_mouse_down
            
            if button_held:
                any_pressed = True
                if not jog_buttons[btn_tag]:
                    parts = btn_tag.split('_')
                    axis = int(parts[2])
                    sign = int(parts[3])
                    jog_buttons[btn_tag] = True
                    jog(axis, sign, btn_tag)
            else:
                if jog_buttons[btn_tag]:
                    jog_buttons[btn_tag] = False
    
    if state.robot and state.robot.is_jogging and not any_pressed:
        stop_jog()


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

