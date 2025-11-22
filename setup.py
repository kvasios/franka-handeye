from setuptools import setup, find_packages

setup(
    name="franka_handeye",
    version="0.1.0",
    packages=find_packages(),
    install_requires=[
        "numpy",
        "opencv-contrib-python",
        "pyrealsense2",
        "PyYAML",
        "scipy",
        "matplotlib",
        "franky-remote", 
    ],
)
