# Franka Hand-Eye Calibration

A pure Python tool for hand-eye calibration of the Franka robot using a RealSense camera and Charuco board.

## Installation

### 1. Server Setup (Set & Forget)

The robot control requires a real-time environment. We recommend using **`servobox`** on your local workstation, but you can also run `franky-remote` manually or use a local `franky` installation if you have a Real-Time kernel setup.

Open a dedicated terminal (Terminal 1) and run the server. **This process must remain running in the background.**

```bash
# Option A: Using Servobox (Recommended)
servobox pkg-install franky-remote-gen1  # (or franky-remote-fr3)
servobox run franky-remote-gen1

# Option B: Manual / Custom Setup
# Ensure your franky/franky-remote server is running and accessible.
```

### 2. Client Setup

In a new terminal (Terminal 2), clone the repository and install the environment. We recommend `micromamba`, but `conda` works too.

```bash
# Clone this repository
git clone https://github.com/kvasios/franka-handeye.git
cd franka-handeye

# Create and activate environment (micromamba or conda)
micromamba create -n franka-handeye python=3.10
micromamba activate franka-handeye

# Install dependencies
pip install -r requirements.txt
pip install -e .
```

## Configuration

### Charuco Board

1. Print the Charuco board provided in `config/charuco_board_5x7.png`
2. Measure the actual physical dimensions of the squares and markers
3. Update `config/calibration_board_parameters.yaml` with the measured values:
   - `board_size`: Number of squares [width, height]
   - `square_length`: Size of each square in meters
   - `marker_length`: Size of each ArUco marker in meters

### Joint Poses

The file `config/joint_poses.yaml` must contain exactly **12 joint configurations** (7 values each) that the robot will visit during calibration. These poses should provide good coverage of the workspace and diverse viewing angles of the calibration board.

**Requirements:**
- Exactly 12 poses
- Each pose must have 7 joint angles (in radians)
- Poses should be collision-free and reachable
- The calibration board must be visible from all poses

## Usage

### Step 1: Data Capture

The capture script automatically moves the robot through all 12 joint poses defined in `config/joint_poses.yaml`, captures images with the RealSense camera, detects the Charuco board, and records the robot state.

```bash
# Set the Franka Server IP (Real-Time Machine IP)
export FRANKY_SERVER_IP=192.168.1.X 

# Run the capture script
python scripts/capture_data.py --host <ROBOT_FCI_IP>
```

**Arguments:**
- `--host`: Robot FCI IP address (default: `172.16.0.2`)
- `--output`: Output directory (default: `data/captured-data`)

The script will:
1. Validate that exactly 12 poses are configured
2. Connect to the robot and camera
3. Move through each pose sequentially
4. Capture an image and detect the Charuco board at each pose
5. Save all data to the output directory

**Output:** The script creates a `pose_XX` folder for each capture containing:
- `image.png`: Captured camera image
- `data.json`: Robot state, camera intrinsics, and Charuco detection results

### Step 2: Compute Calibration

After successfully capturing data from all 12 poses, run the calibration script to compute the hand-eye transformation (camera-to-gripper transform).

```bash
python scripts/compute_calibration.py
```

**Optional arguments:**
- `--data`: Input directory with captured data (default: `data/captured-data`)
- `--plot`: Show 3D visualization of the calibrated frames

The script will:
1. Load all captured poses
2. Run hand-eye calibration using OpenCV's Daniilidis method
3. Compute consistency metrics across all poses
4. Save the calibration result to `data/hand-eye-calibration-output/calibration_result.json`

**Output:** The calibration result contains:
- `T_cam_gripper`: 4x4 homogeneous transformation matrix
- `xyz`: Translation vector [x, y, z]
- `quaternion_xyzw`: Orientation as quaternion [x, y, z, w]
- `consistency_error_mean`: Mean reprojection error
- `consistency_error_std`: Standard deviation of reprojection error

**⚠️ Important:** Before using the calibration, visually inspect the results with:

```bash
python scripts/compute_calibration.py --plot
```

### Step 3: Verify Calibration (Optional but Recommended)

To validate the calibration, you can run a verification script that uses the computed transformation to align the robot with the charuco board.

```bash
python scripts/verify_calibration.py --host <ROBOT_FCI_IP>
```

**Optional arguments:**
- `--host`: Robot FCI IP address (default: `172.16.0.2`)
- `--offset`: Distance from board in meters (default: `0.1`)
- `--calibration`: Path to calibration result (default: `data/hand-eye-calibration-output/calibration_result.json`)

This script will:
1. Detect the charuco board from the current robot position
2. Use the calibration to compute where the robot should move
3. Move the camera to be positioned `offset` meters from the board, aligned with its center
4. Prompt for confirmation before executing the motion

**Safety:** The script includes two confirmation prompts and moves the robot slowly. Always ensure the workspace is clear and have the emergency stop ready.

## Workflow Summary

1. **Setup**: Ensure server is running and configuration files are correct
2. **Capture**: `python scripts/capture_data.py --host <ROBOT_IP>`
3. **Calibrate**: `python scripts/compute_calibration.py [--plot]`
4. **Verify**: `python scripts/verify_calibration.py --host <ROBOT_IP>` (optional)
5. **Result**: Use the transformation from `data/hand-eye-calibration-output/calibration_result.json`

