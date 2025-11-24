"""
ChArUco board detection for hand-eye calibration.
"""

import os
import yaml
import numpy as np
import cv2
from pathlib import Path


class CharucoDetector:
    """
    ChArUco board detector for pose estimation.
    
    Parameters
    ----------
    params_path : str or Path
        Path to YAML file containing board parameters:
        - board_size: [cols, rows]
        - square_length: float (meters)
        - marker_length: float (meters)
    """
    
    def __init__(self, params_path: str | Path):
        params_path = Path(params_path)
        
        if not params_path.exists():
            raise FileNotFoundError(f"Board parameters file not found: {params_path}")
        
        with open(params_path, 'r') as f:
            params = yaml.safe_load(f)
        
        self.board_size = tuple(params['board_size'])
        self.square_length = params['square_length']
        self.marker_length = params['marker_length']
        
        self.board = cv2.aruco.CharucoBoard(
            self.board_size,
            self.square_length,
            self.marker_length,
            cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_6X6_250)
        )
        self.dictionary = self.board.getDictionary()
        self.params = cv2.aruco.DetectorParameters()
    
    @property
    def board_dimensions(self) -> tuple[float, float]:
        """
        Get physical dimensions of the board in meters.
        
        Returns
        -------
        tuple
            (width, height) in meters.
        """
        width = self.board_size[0] * self.square_length
        height = self.board_size[1] * self.square_length
        return width, height
    
    def detect(
        self, 
        image: np.ndarray, 
        K: np.ndarray, 
        D: np.ndarray
    ) -> tuple[bool, np.ndarray | None, np.ndarray | None, np.ndarray | None]:
        """
        Detect ChArUco board and estimate its pose.
        
        Parameters
        ----------
        image : np.ndarray
            BGR image to detect board in.
        K : np.ndarray
            3x3 camera intrinsics matrix.
        D : np.ndarray
            Distortion coefficients.
        
        Returns
        -------
        tuple
            (valid, rvec, tvec, charuco_corners) where:
            - valid: bool indicating if detection succeeded
            - rvec: rotation vector (Rodrigues)
            - tvec: translation vector
            - charuco_corners: detected corner points
        """
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        corners, ids, _ = cv2.aruco.detectMarkers(
            gray, self.dictionary, parameters=self.params
        )
        
        if ids is not None and len(ids) > 0:
            _, charuco_corners, charuco_ids = cv2.aruco.interpolateCornersCharuco(
                corners, ids, gray, self.board
            )
            if charuco_corners is not None and len(charuco_corners) > 3:
                valid, rvec, tvec = cv2.aruco.estimatePoseCharucoBoard(
                    charuco_corners, charuco_ids, self.board, K, D, None, None
                )
                return valid, rvec, tvec, charuco_corners
        
        return False, None, None, None
    
    def draw_detection(
        self, 
        image: np.ndarray, 
        K: np.ndarray, 
        D: np.ndarray, 
        rvec: np.ndarray, 
        tvec: np.ndarray,
        axis_length: float = 0.1
    ) -> np.ndarray:
        """
        Draw coordinate axes on detected board.
        
        Parameters
        ----------
        image : np.ndarray
            Image to draw on (will be modified in place).
        K : np.ndarray
            Camera intrinsics matrix.
        D : np.ndarray
            Distortion coefficients.
        rvec : np.ndarray
            Rotation vector from detection.
        tvec : np.ndarray
            Translation vector from detection.
        axis_length : float
            Length of coordinate axes to draw (meters).
        
        Returns
        -------
        np.ndarray
            Image with axes drawn.
        """
        cv2.drawFrameAxes(image, K, D, rvec, tvec, axis_length)
        return image
    
    def get_board_corners(self) -> list[list[float]]:
        """
        Get the 4 corners of the board in board frame coordinates.
        
        OpenCV charuco frame: origin at top-left, X right, Y down, Z out.
        
        Returns
        -------
        list
            List of 4 corner positions [x, y, z] in board frame.
        """
        width, height = self.board_dimensions
        return [
            [0, 0, 0],              # Top-left
            [width, 0, 0],          # Top-right
            [width, height, 0],     # Bottom-right
            [0, height, 0]          # Bottom-left
        ]
    
    def get_board_center(self) -> list[float]:
        """
        Get the center point of the board in board frame coordinates.
        
        Returns
        -------
        list
            [x, y, z] center position in board frame.
        """
        width, height = self.board_dimensions
        return [width / 2, height / 2, 0]

