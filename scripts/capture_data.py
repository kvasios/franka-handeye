import os
import time
import argparse
import yaml
import numpy as np
import cv2
import pyrealsense2 as rs
import json
import threading
import tkinter as tk
from tkinter import ttk, messagebox
from PIL import Image, ImageTk
from pathlib import Path
import shutil

# Set default server IP
os.environ.setdefault("FRANKY_SERVER_IP", "192.168.122.100")

from scipy.spatial.transform import Rotation as R
from franky import (
    Robot, 
    JointMotion, 
    CartesianMotion, 
    Affine, 
    ReferenceType,
    CartesianVelocityMotion,
    Twist,
    Duration,
    Reaction,
    Measure,
    CartesianVelocityStopMotion,
    Condition
)

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

class ManualCaptureApp:
    def __init__(self, root, robot, camera, detector, output_dir, K, D):
        self.root = root
        self.root.title("Franka Hand-Eye Capture (Step Jogging)")
        self.robot = robot
        self.camera = camera
        self.detector = detector
        self.output_dir = output_dir
        self.K = K
        self.D = D
        
        self.captured_count = 0
        self.captured_poses = []
        self.target_captures = 12
        
        self.jogging_active = False
        self.jog_axis = -1 # 0-5
        self.jog_direction = 0 # 1 or -1
        self.step_size_linear = 0.02 # 2cm per step
        self.step_size_angular = 0.1 # rad per step
        
        # Clean output directory for new session
        if output_dir.exists():
            shutil.rmtree(output_dir)
        output_dir.mkdir(parents=True)

        # UI Layout
        self.main_frame = ttk.Frame(root)
        self.main_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)

        # Video Feed
        self.video_label = ttk.Label(self.main_frame)
        self.video_label.grid(row=0, column=0, rowspan=10, padx=5, pady=5)

        # Controls Frame
        self.controls_frame = ttk.LabelFrame(self.main_frame, text="Cartesian Jogging")
        self.controls_frame.grid(row=0, column=1, sticky="ns", padx=5)

        # Instructions
        ttk.Label(self.controls_frame, text="Click & Hold to Move (Step)").pack(pady=5)

        # Jogging Buttons Layout
        # Translation
        trans_frame = ttk.LabelFrame(self.controls_frame, text="Translation")
        trans_frame.pack(pady=5, fill=tk.X)
        
        self.create_jog_btn(trans_frame, "X+", 0, 1) 
        self.create_jog_btn(trans_frame, "X-", 0, -1)
        self.create_jog_btn(trans_frame, "Y+", 1, 1)
        self.create_jog_btn(trans_frame, "Y-", 1, -1)
        self.create_jog_btn(trans_frame, "Z+", 2, 1)
        self.create_jog_btn(trans_frame, "Z-", 2, -1)

        # Rotation
        rot_frame = ttk.LabelFrame(self.controls_frame, text="Rotation")
        rot_frame.pack(pady=5, fill=tk.X)

        self.create_jog_btn(rot_frame, "RX+", 3, 1)
        self.create_jog_btn(rot_frame, "RX-", 3, -1)
        self.create_jog_btn(rot_frame, "RY+", 4, 1)
        self.create_jog_btn(rot_frame, "RY-", 4, -1)
        self.create_jog_btn(rot_frame, "RZ+", 5, 1)
        self.create_jog_btn(rot_frame, "RZ-", 5, -1)

        # Capture Frame
        self.capture_frame = ttk.LabelFrame(self.main_frame, text="Capture")
        self.capture_frame.grid(row=1, column=1, sticky="ew", padx=5)
        
        self.lbl_count = ttk.Label(self.capture_frame, text=f"Captured: 0/{self.target_captures}")
        self.lbl_count.pack(pady=5)
        
        self.btn_capture = ttk.Button(self.capture_frame, text="Capture Pose", command=self.capture_pose)
        self.btn_capture.pack(pady=5, fill=tk.X)

        # Robot State Display
        self.info_frame = ttk.LabelFrame(self.main_frame, text="Robot State")
        self.info_frame.grid(row=2, column=1, sticky="ew", padx=5)

        self.lbl_cartesian = ttk.Label(self.info_frame, text="Cartesian: ", wraplength=200)
        self.lbl_cartesian.pack(pady=2, fill=tk.X)
        
        self.lbl_joints = ttk.Label(self.info_frame, text="Joints: ", wraplength=200)
        self.lbl_joints.pack(pady=2, fill=tk.X)

        # Home Button
        self.btn_home = ttk.Button(self.main_frame, text="Go Home", command=self.go_home)
        self.btn_home.grid(row=3, column=1, sticky="ew", padx=5, pady=5)
        
        # Start Loops
        self.update_video()
        self.update_robot_state()
        
        self.root.protocol("WM_DELETE_WINDOW", self.on_close)

    def update_robot_state(self):
        try:
            state = self.robot.state
            
            # Format Cartesian Pose (Translation + Quat or Matrix)
            # O_T_EE is an Affine object. We can access .translation directly.
            # translation is a 3x1 array/list
            O_T_EE_affine = state.O_T_EE
            trans_remote = O_T_EE_affine.translation
            trans = [trans_remote[i] for i in range(3)]
            
            self.lbl_cartesian.configure(text=f"Pos: [{trans[0]:.3f}, {trans[1]:.3f}, {trans[2]:.3f}]")
            
            # Format Joints
            # Manual copy from RPyC
            q_remote = state.q
            q = [q_remote[i] for i in range(len(q_remote))]
            q_str = ", ".join([f"{x:.2f}" for x in q])
            self.lbl_joints.configure(text=f"Joints: [{q_str}]")
            
        except Exception as e:
            pass
            
        self.root.after(200, self.update_robot_state)

    def go_home(self):
        print("Going Home...")
        try:
            self.robot.recover_from_errors()
            self.robot.move(JointMotion([0.0, 0.0, 0.0, -2.2, 0.0, 2.2, 0.7]))
        except Exception as e:
            print(f"Home error: {e}")

    def create_jog_btn(self, parent, text, axis_idx, direction):
        btn = ttk.Button(parent, text=text)
        btn.pack(fill=tk.X, pady=2)
        btn.bind('<ButtonPress-1>', lambda e: self.start_jog(axis_idx, direction))
        btn.bind('<ButtonRelease-1>', lambda e: self.stop_jog())
        return btn

    def start_jog(self, axis, direction):
        if self.jogging_active:
            return
            
        self.jogging_active = True
        
        # Recovery attempt if previous move failed
        try:
            self.robot.recover_from_errors()
        except:
            pass
        
        # Define Twist
        linear = [0.0, 0.0, 0.0]
        angular = [0.0, 0.0, 0.0]
        
        # Reduced velocity constants to prevent discontinuity reflexes
        v_lin = 0.02  # Reduced from 0.05
        v_ang = 0.1   # Reduced from 0.25
        
        if axis < 3:
            linear[axis] = v_lin * direction
        else:
            angular[axis-3] = v_ang * direction
            
        twist = Twist(linear, angular)
        
        # Create Motion (run for up to 10 seconds)
        # Use RelativeDynamicsFactor to control acceleration/jerk more precisely if needed,
        # but CartesianVelocityMotion doesn't accept it directly in constructor in all versions.
        # It usually relies on robot.relative_dynamics_factor.
        # We set robot.relative_dynamics_factor = 0.05 globally which is safe.
        
        motion = CartesianVelocityMotion(twist, duration=Duration(10000))
        
        # Add safety reaction (Force Norm)
        # Manually constructing norm since FORCE_XYZ_NORM is missing
        force_norm_cond = (Measure.FORCE_X * Measure.FORCE_X + 
                           Measure.FORCE_Y * Measure.FORCE_Y + 
                           Measure.FORCE_Z * Measure.FORCE_Z) > 100.0 # 10^2 = 100
                           
        reaction_force = Reaction(
            force_norm_cond, 
            CartesianVelocityStopMotion(relative_dynamics_factor=0.05)
        )
        motion.add_reaction(reaction_force)
        
        try:
            self.robot.recover_from_errors()
            self.robot.move(motion, asynchronous=True)
        except Exception as e:
            print(f"Start jog error: {e}")
            self.jogging_active = False

    def stop_jog(self):
        if not self.jogging_active:
            return
            
        try:
            # Stop the robot immediately
            self.robot.stop()
            self.robot.recover_from_errors()
        except Exception as e:
            # Ignor preemption errors during stop
            pass
        finally:
            self.jogging_active = False

    # Threading logic removed as we use asynchronous moves
    def start_jog_thread(self):
        pass

    def _jog_loop(self):
        pass

    def update_video(self):
        frame = self.camera.get_frame()
        if frame is not None:
            # Detection
            valid, rvec, tvec, corners = self.detector.detect(frame, self.K, self.D)
            if valid:
                cv2.drawFrameAxes(frame, self.K, self.D, rvec, tvec, 0.1)
                self.current_detection = (True, rvec, tvec)
            else:
                self.current_detection = (False, None, None)

            # Convert to Tkinter
            img = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
            img = img.resize((640, 360)) # Resize for GUI
            imgtk = ImageTk.PhotoImage(image=img)
            self.video_label.imgtk = imgtk
            self.video_label.configure(image=imgtk)
            
            self.last_frame = frame # Store for saving

        self.root.after(33, self.update_video)

    def capture_pose(self):
        if not hasattr(self, 'last_frame') or self.last_frame is None:
            return
            
        state = self.robot.state
        # Convert to standard python types to avoid JSON serialization errors with numpy scalars
        # Iterate manually over RPyC netref arrays because direct numpy conversion relies on pickling which is disabled
        q_remote = state.q
        q = [float(q_remote[i]) for i in range(len(q_remote))]
        
        # O_T_EE is an Affine object, not a list/array. We must access its .matrix property.
        # .matrix returns a numpy array (likely 4x4), but accessing it via RPyC might still return a netref 
        # that we can't pickle-convert. We have to iterate over it manually.
        # Assuming .matrix returns a 4x4 array/list-like structure.
        O_T_EE_affine = state.O_T_EE
        O_T_EE_matrix_remote = O_T_EE_affine.matrix
        
        # Since it's likely a 2D array (4x4), we need nested iteration or flattening
        # Explicitly convert each element to float to avoid numpy.float64 serialization issues
        O_T_EE = []
        for r in range(4):
            row_remote = O_T_EE_matrix_remote[r]
            row = [float(row_remote[c]) for c in range(4)]
            O_T_EE.extend(row) # Flattening to 16-element list as expected by downstream code
        
        valid, rvec, tvec = self.current_detection
        
        # Save
        pose_idx = self.captured_count
        pose_dir = self.output_dir / f"pose_{pose_idx:02d}"
        pose_dir.mkdir(exist_ok=True)
        
        cv2.imwrite(str(pose_dir / "image.png"), self.last_frame)
        
        data = {
            "joint_pose": q,
            "O_T_EE": O_T_EE,
            "camera_intrinsics": self.K,
            "dist_coeffs": self.D,
            "charuco_detected": valid,
        }
        
        if valid:
            data["T_cam_target_rvec"] = rvec.tolist() if hasattr(rvec, 'tolist') else rvec
            data["T_cam_target_tvec"] = tvec.tolist() if hasattr(tvec, 'tolist') else tvec
            
        with open(pose_dir / "data.json", 'w') as f:
            json.dump(data, f, indent=4, cls=NumpyEncoder)
            
        self.captured_poses.append(q)
        self.captured_count += 1
        self.lbl_count.configure(text=f"Captured: {self.captured_count}/{self.target_captures}")
        print(f"Captured pose {pose_idx}")
        
        # Auto-save joint config on every capture (or at end)
        self.save_joint_poses()

        if self.captured_count >= self.target_captures:
            messagebox.showinfo("Done", "Captured 12 Poses! Exiting...")
            self.on_close()

    def save_joint_poses(self):
        path = "config/joint_poses.yaml"
        # Format each pose on a single line for readability
        yaml_content = "joint_poses:\n"
        for pose in self.captured_poses:
            pose_str = "  - [" + ", ".join([f"{x:.4f}" for x in pose]) + "]\n"
            yaml_content += pose_str
        
        with open(path, 'w') as f:
            f.write(yaml_content)

    def on_close(self):
        self.stop_thread = True
        self.camera.stop()
        self.root.destroy()
        os._exit(0) # Force exit threads

def run_automatic(robot, camera, detector, joint_poses, output_dir, K, D):
    print(f"Starting Automatic Capture of {len(joint_poses)} poses...")
    
    # Clean output directory
    if output_dir.exists():
        shutil.rmtree(output_dir)
    output_dir.mkdir(parents=True)
    
    for i, pose in enumerate(joint_poses):
        print(f"\n--- Pose {i+1}/{len(joint_poses)} ---")
        try:
            robot.move(JointMotion(pose))
            time.sleep(0.5) # Stabilization
            
            frame = camera.get_frame()
            if frame is None: continue
            
            state = robot.state
            O_T_EE = np.array(state.O_T_EE).tolist()
            
            valid, rvec, tvec, _ = detector.detect(frame, K, D)
            
            if valid:
                print("Charuco detected.")
                cv2.drawFrameAxes(frame, K, D, rvec, tvec, 0.1)
            else:
                print("WARNING: Charuco NOT detected.")
                
            pose_dir = output_dir / f"pose_{i:02d}"
            pose_dir.mkdir(exist_ok=True)
            cv2.imwrite(str(pose_dir / "image.png"), frame)
            
            data = {
                "joint_pose": pose,
                "O_T_EE": O_T_EE,
                "camera_intrinsics": K,
                "dist_coeffs": D,
                "charuco_detected": valid,
            }
            if valid:
                data["T_cam_target_rvec"] = rvec.tolist()
                data["T_cam_target_tvec"] = tvec.tolist()
                
            with open(pose_dir / "data.json", 'w') as f:
                json.dump(data, f, indent=4, cls=NumpyEncoder)
                
        except Exception as e:
            print(f"Error at pose {i}: {e}")

def main():
    parser = argparse.ArgumentParser(description="Franka Hand-Eye Calibration Capture")
    parser.add_argument("--host", default="172.16.0.2", help="FCI IP")
    parser.add_argument("--output", default="data/captured-data", help="Output dir")
    parser.add_argument("--manual", action="store_true", help="Force manual mode")
    args = parser.parse_args()

    output_dir = Path(args.output)
    # (Directory cleanup handled inside functions now)

    # Initialize Hardware
    print("Initializing Camera...")
    camera = RealSenseCamera()
    K, D = camera.get_intrinsics_matrix()
    
    print(f"Connecting to Robot at {args.host}...")
    try:
        robot = Robot(args.host)
        robot.recover_from_errors()
        robot.relative_dynamics_factor = 0.05 # Set global dynamics factor as requested
    except Exception as e:
        print(f"Failed to connect to robot: {e}")
        camera.stop()
        return

    detector = CharucoDetector("charuco/calibration_board_parameters.yaml")

    # Check Config
    poses_path = Path("config/joint_poses.yaml")
    joint_poses = []
    if poses_path.exists():
        with open(poses_path, 'r') as f:
            data = yaml.safe_load(f)
            if data and 'joint_poses' in data and data['joint_poses']:
                joint_poses = data['joint_poses']
    
    # Determine Mode
    mode_manual = args.manual or not joint_poses
    
    if mode_manual:
        print("Starting Manual GUI Mode (Cartesian Jogging)...")
        root = tk.Tk()
        app = ManualCaptureApp(root, robot, camera, detector, output_dir, K, D)
        root.mainloop()
    else:
        run_automatic(robot, camera, detector, joint_poses, output_dir, K, D)
        camera.stop()

if __name__ == "__main__":
    main()
