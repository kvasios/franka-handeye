#!/usr/bin/env python3
"""
Main orchestration script for Franka Hand-Eye Calibration.

This script runs the complete calibration workflow:
1. Data capture (capture_data.py)
2. Calibration computation (compute_calibration.py)
3. Verification (verify_calibration.py)
"""

import os
import sys
import subprocess
import argparse
from pathlib import Path


def run_command(cmd, description):
    """Run a command and handle errors."""
    print("\n" + "=" * 70)
    print(f"STEP: {description}")
    print("=" * 70)
    print(f"Running: {' '.join(cmd)}\n")
    
    result = subprocess.run(cmd)
    
    if result.returncode != 0:
        print(f"\n❌ ERROR: {description} failed with exit code {result.returncode}")
        return False
    
    print(f"\n✅ {description} completed successfully!")
    return True


def main():
    parser = argparse.ArgumentParser(
        description="Run complete Franka Hand-Eye Calibration workflow",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
This script automates the entire calibration process:
  1. Captures data from robot at 12 predefined poses
  2. Computes hand-eye calibration using captured data (with visualization)
  3. Verifies calibration by visiting board center and corners

Requirements:
  - franky-remote server must be running
  - Robot must be operational and accessible
  - RealSense camera must be connected
  - Charuco board must be visible from robot poses
  - config/joint_poses.yaml must contain exactly 12 poses

Note: For more control, run individual scripts in scripts/ directory.
        """
    )
    
    parser.add_argument("--host", default="172.16.0.2", 
                        help="Robot FCI IP address (default: 172.16.0.2)")
    parser.add_argument("--verify-offset", type=float, default=0.06,
                        help="Verification offset from board in meters (default: 0.06)")
    
    args = parser.parse_args()
    
    print("=" * 70)
    print("Franka Hand-Eye Calibration - Complete Workflow")
    print("=" * 70)
    print("\nThis script will run the complete calibration pipeline:")
    print("  1. Data Capture")
    print("  2. Calibration Computation (with 3D visualization)")
    print("  3. Calibration Verification")
    print("\n" + "=" * 70)
    
    response = input("\n➤ Press ENTER to begin or Ctrl+C to abort: ")
    
    # Step 1: Data Capture
    if not run_command(
        ["python", "scripts/capture_data.py", "--host", args.host],
        "Data Capture"
    ):
        return 1
    
    # Step 2: Calibration Computation (always with plot)
    if not run_command(
        ["python", "scripts/compute_calibration.py", "--plot"],
        "Calibration Computation"
    ):
        return 1
    
    # Step 3: Verification
    if not run_command(
        ["python", "scripts/verify_calibration.py", 
         "--host", args.host,
         "--offset", str(args.verify_offset)],
        "Calibration Verification"
    ):
        return 1
    
    # Summary
    print("\n" + "=" * 70)
    print("🎉 COMPLETE WORKFLOW FINISHED SUCCESSFULLY!")
    print("=" * 70)
    print("\n✅ Calibration pipeline completed:")
    print("  ✓ Data captured from 12 robot poses")
    print("  ✓ Hand-eye calibration computed and visualized")
    print("  ✓ Result saved to: data/hand-eye-calibration-output/calibration_result.json")
    print("  ✓ Calibration verified by visiting board center and corners")
    
    print("\nYou can now use the calibration result in your applications!")
    print("\n" + "=" * 70)
    
    return 0


if __name__ == "__main__":
    try:
        exit(main())
    except KeyboardInterrupt:
        print("\n\n❌ Workflow interrupted by user")
        exit(1)

