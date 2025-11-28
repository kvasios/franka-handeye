"""
Franka robot interface for hand-eye calibration.
"""

import os
import numpy as np
from typing import Callable

# Set default server IP before importing franky
os.environ.setdefault("FRANKY_SERVER_IP", "192.168.122.100")

try:
    from franky import (
        Robot as FrankyRobot,
        JointMotion,
        CartesianMotion,
        CartesianVelocityMotion,
        CartesianVelocityStopMotion,
        Twist,
        Duration,
        Reaction,
        Measure,
        Affine,
        ReferenceType,
        RelativeDynamicsFactor,
        Gripper as FrankyGripper,
    )
    FRANKY_AVAILABLE = True
except ImportError:
    FRANKY_AVAILABLE = False


class RobotController:
    """
    High-level Franka robot controller for hand-eye calibration.
    
    Parameters
    ----------
    host : str
        Robot FCI IP address.
    dynamics_factor : float
        Global relative dynamics factor (0.0 to 1.0). Lower = slower/safer.
    cartesian_velocity : float
        Velocity factor for Cartesian motions (0.0 to 1.0).
    cartesian_acceleration : float
        Acceleration factor for Cartesian motions (0.0 to 1.0).
    cartesian_jerk : float
        Jerk factor for Cartesian motions (0.0 to 1.0). Lower = smoother.
    """
    
    # Default home position (joint angles in radians)
    HOME_POSE = [0.0, 0.0, 0.0, -2.2, 0.0, 2.2, 0.7]
    
    def __init__(
        self, 
        host: str, 
        dynamics_factor: float = 0.1,
        cartesian_velocity: float = 0.8,
        cartesian_acceleration: float = 0.5,
        cartesian_jerk: float = 0.1
    ):
        # Try to import franky if it wasn't available at module load time
        global FRANKY_AVAILABLE, FrankyRobot, JointMotion, CartesianMotion, CartesianVelocityMotion, CartesianVelocityStopMotion, Twist, Duration, Reaction, Measure, Affine, ReferenceType, RelativeDynamicsFactor, FrankyGripper
        
        if not FRANKY_AVAILABLE:
            try:
                from franky import (
                    Robot as FrankyRobot,
                    JointMotion,
                    CartesianMotion,
                    CartesianVelocityMotion,
                    CartesianVelocityStopMotion,
                    Twist,
                    Duration,
                    Reaction,
                    Measure,
                    Affine,
                    ReferenceType,
                    RelativeDynamicsFactor,
                    Gripper as FrankyGripper,
                )
                FRANKY_AVAILABLE = True
            except ImportError:
                pass

        if not FRANKY_AVAILABLE:
            raise RuntimeError("franky is not installed")
            
        self.host = host
        self._robot = FrankyRobot(host)
        self._robot.recover_from_errors()
        self._robot.relative_dynamics_factor = dynamics_factor
        self._gripper = None
        self._jogging = False
        
        # Cartesian motion dynamics (conservative defaults to avoid discontinuity errors)
        self.cartesian_velocity = cartesian_velocity
        self.cartesian_acceleration = cartesian_acceleration
        self.cartesian_jerk = cartesian_jerk
    
    @property
    def robot(self) -> "FrankyRobot":
        """Access underlying franky Robot object."""
        return self._robot
    
    @property
    def gripper(self) -> "FrankyGripper":
        """Lazy-initialized gripper."""
        if self._gripper is None:
            self._gripper = FrankyGripper(self.host)
        return self._gripper
    
    @property
    def dynamics_factor(self) -> float:
        """Get current dynamics factor."""
        return self._robot.relative_dynamics_factor
    
    @dynamics_factor.setter
    def dynamics_factor(self, value: float):
        """Set dynamics factor (0.0 to 1.0)."""
        self._robot.relative_dynamics_factor = max(0.01, min(1.0, value))
    
    def recover(self):
        """
        Recover from errors.
        
        Attempts to call recover_from_errors().
        If that fails (e.g. connection lost), it re-initializes the robot connection.
        """
        try:
            self._robot.recover_from_errors()
        except Exception as e:
            print(f"Recovery failed ({e}), attempting full reconnection...")
            try:
                # Store current config
                current_dynamics = self._robot.relative_dynamics_factor
                
                # Re-instantiate robot
                self._robot = FrankyRobot(self.host)
                self._robot.recover_from_errors()
                
                # Restore config
                self._robot.relative_dynamics_factor = current_dynamics
                
                # Re-initialize gripper if it was used
                if self._gripper is not None:
                    self._gripper = FrankyGripper(self.host)
                    
                print("Reconnection successful")
            except Exception as e2:
                print(f"Reconnection failed: {e2}")
                raise e2
    
    def get_state(self) -> dict:
        """
        Get current robot state.
        
        Returns
        -------
        dict
            Dictionary with 'q' (joint angles), 'O_T_EE' (4x4 EE pose matrix),
            'position' (xyz), 'forces' (external forces).
        """
        state = self._robot.state
        
        # Extract joint positions
        q_remote = state.q
        q = [float(q_remote[i]) for i in range(len(q_remote))]
        
        # Extract end-effector pose as 4x4 matrix
        O_T_EE_affine = state.O_T_EE
        O_T_EE_matrix_remote = O_T_EE_affine.matrix
        O_T_EE = np.zeros((4, 4))
        for r in range(4):
            row_remote = O_T_EE_matrix_remote[r]
            for c in range(4):
                O_T_EE[r, c] = float(row_remote[c])
        
        # Extract position
        trans = O_T_EE_affine.translation
        position = [float(trans[i]) for i in range(3)]
        
        return {
            'q': q,
            'O_T_EE': O_T_EE,
            'position': position,
        }
    
    def get_ee_pose_flat(self) -> list[float]:
        """
        Get end-effector pose as flattened 16-element list (row-major).
        
        Returns
        -------
        list
            16-element list representing 4x4 transformation matrix.
        """
        state = self.get_state()
        return state['O_T_EE'].flatten().tolist()
    
    def move_joints(self, joint_positions: list[float], asynchronous: bool = False):
        """
        Move to joint positions.
        
        Parameters
        ----------
        joint_positions : list
            7 joint angles in radians.
        asynchronous : bool
            If True, return immediately without waiting for motion to complete.
        """
        self._robot.move(JointMotion(joint_positions), asynchronous=asynchronous)
    
    def move_cartesian(
        self, 
        position: list[float], 
        quaternion: list[float],
        relative_dynamics_factor: "RelativeDynamicsFactor | float | None" = None,
        asynchronous: bool = False
    ):
        """
        Move to Cartesian pose.
        
        Parameters
        ----------
        position : list
            [x, y, z] position in meters.
        quaternion : list
            [x, y, z, w] quaternion orientation.
        relative_dynamics_factor : RelativeDynamicsFactor, float, or None
            Dynamics factor for this motion. Uses instance defaults if None.
        asynchronous : bool
            If True, return immediately.
        """
        target = Affine(position, quaternion)
        if relative_dynamics_factor is not None:
            dynamics = relative_dynamics_factor
        else:
            dynamics = RelativeDynamicsFactor(
                velocity=self.cartesian_velocity,
                acceleration=self.cartesian_acceleration,
                jerk=self.cartesian_jerk
            )
        motion = CartesianMotion(target, ReferenceType.Absolute, dynamics)
        self._robot.move(motion, asynchronous=asynchronous)
    
    def go_home(self, asynchronous: bool = False):
        """Move to home position."""
        self.move_joints(self.HOME_POSE, asynchronous=asynchronous)
    
    def start_jog(
        self, 
        axis: int, 
        direction: int,
        linear_speed: float = 0.02,
        angular_speed: float = 0.1,
        force_threshold: float = 10.0
    ):
        """
        Start jogging motion along an axis.
        
        Parameters
        ----------
        axis : int
            Axis index (0-2 for translation X/Y/Z, 3-5 for rotation RX/RY/RZ).
        direction : int
            Direction (+1 or -1).
        linear_speed : float
            Linear velocity in m/s.
        angular_speed : float
            Angular velocity in rad/s.
        force_threshold : float
            Force threshold for safety stop (N).
        """
        if self._jogging:
            self.stop_jog()
        
        self.recover()
        
        linear = [0.0, 0.0, 0.0]
        angular = [0.0, 0.0, 0.0]
        
        if axis < 3:
            linear[axis] = linear_speed * direction
        else:
            angular[axis - 3] = angular_speed * direction
        
        twist = Twist(linear, angular)
        motion = CartesianVelocityMotion(twist, duration=Duration(10000))
        
        # Safety reactions for force limits
        force_x_exceeded = Measure.FORCE_X > force_threshold
        force_x_neg_exceeded = Measure.FORCE_X < -force_threshold
        force_y_exceeded = Measure.FORCE_Y > force_threshold
        force_y_neg_exceeded = Measure.FORCE_Y < -force_threshold
        force_z_exceeded = Measure.FORCE_Z > force_threshold
        force_z_neg_exceeded = Measure.FORCE_Z < -force_threshold
        
        force_exceeded = (
            force_x_exceeded | force_x_neg_exceeded |
            force_y_exceeded | force_y_neg_exceeded |
            force_z_exceeded | force_z_neg_exceeded
        )
        
        reaction = Reaction(
            force_exceeded, 
            CartesianVelocityStopMotion(relative_dynamics_factor=0.05)
        )
        motion.add_reaction(reaction)
        
        self._jogging = True
        self._robot.move(motion, asynchronous=True)
    
    def stop_jog(self):
        """Stop jogging motion."""
        # Always try to stop, regardless of _jogging state
        # (the async motion might have finished/errored but flag is still True)
        try:
            stop_motion = CartesianVelocityStopMotion(relative_dynamics_factor=0.9)
            self._robot.move(stop_motion)
            self.recover()
        except Exception:
            pass
        finally:
            self._jogging = False
    
    def clear_jog_state(self):
        """Clear jogging state without sending stop command."""
        self._jogging = False
    
    @property
    def is_jogging(self) -> bool:
        """Check if currently jogging."""
        return self._jogging
    
    def home_gripper(self):
        """Home the gripper."""
        self.gripper.homing()
    
    def close_gripper(self, speed: float = 0.1):
        """Close the gripper."""
        self.gripper.move(0, speed)
    
    def open_gripper(self, width: float = 0.08, speed: float = 0.1):
        """Open the gripper to specified width."""
        self.gripper.move(width, speed)

