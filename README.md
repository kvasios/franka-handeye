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

1.  **Charuco Board**:
    -   Ensure you have a Charuco board printed.
    -   Measure the actual physical dimensions of the squares and markers.
    -   Update `charuco/calibration_board_parameters.yaml` with the measured values.

2.  **Joint Poses**:
    -   `config/joint_poses.yaml` contains the joint configurations the robot will visit during calibration.

## Usage

### 1. Data Capture

This script moves the robot to the defined poses, captures images from the RealSense camera, detects the Charuco board, and records the robot state.

```bash
# Set the Franka Server IP (Real-Time Machine IP)
export FRANKY_SERVER_IP=192.168.1.X 

# Run the capture script
python scripts/capture_data.py --host <ROBOT_FCI_IP>
```
*Note: Replace `<ROBOT_FCI_IP>` with the robot's IP address (e.g., 172.16.0.2).*

### 2. Compute Calibration

After capturing data, run the calibration script to compute the hand-eye transformation.

```bash
python scripts/compute_calibration.py
```

