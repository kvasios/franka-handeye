"""
Franka Hand-Eye Calibration Package Setup
"""

from setuptools import setup, find_packages
from pathlib import Path

# Read README for long description
readme_path = Path(__file__).parent / "README.md"
long_description = readme_path.read_text() if readme_path.exists() else ""

setup(
    name="franka-handeye",
    version="0.2.0",
    author="kvasios",
    description="Hand-eye calibration toolkit for Franka robot with RealSense camera",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/kvasios/franka-handeye",
    packages=find_packages(),
    python_requires=">=3.10",
    install_requires=[
        "numpy",
        "opencv-contrib-python",
        "pyrealsense2",
        "PyYAML",
        "scipy",
        "matplotlib",
        "Pillow",
        "dearpygui",
    ],
    extras_require={
        "robot": [
            "franky-remote @ git+https://github.com/kvasios/franky-remote.git",
        ],
    },
    entry_points={
        "console_scripts": [
            "franka-handeye=franka_handeye_app:main",
            "franka-capture=scripts.capture_data:main",
            "franka-calibrate=scripts.compute_calibration:main",
            "franka-verify=scripts.verify_calibration:main",
        ],
    },
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: MIT License",
        "Operating System :: POSIX :: Linux",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Programming Language :: Python :: 3.12",
        "Topic :: Scientific/Engineering :: Image Processing",
        "Topic :: Scientific/Engineering :: Robotics",
    ],
    keywords="robotics, calibration, hand-eye, franka, realsense, charuco",
)
