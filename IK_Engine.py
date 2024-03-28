import math

import matplotlib.pyplot as plt
import numpy as np
from mpl_toolkits.mplot3d import Axes3D

DEBUG = False

# NOTE: In the coordinate system, x is left and right, y is up and down, z is forward and backward (depth)

class Point:
    """
    A point in 3D space
    """
    def __init__(self, x: float, y: float, z: float):
        self.x = x
        self.y = y
        self.z = z


class Arm:
    def __init__(self, origin: Point):
        self.origin = np.array([origin.x, origin.y, origin.z]) 

        # Servo angles
        self.servo1_angle: float = math.pi / 2  # servo1 (yaw) angle in radians
        self.servo2_angle: float = math.pi / 2  # servo2 (pitch) angle in radians:

        # Arm dimensions
        self.base_segment_height: float = 20  # Length of the first vertical segment, cm
        self.offset: float = 12  # Offset from the center of the robot, cm
        self.actuator_min_len: float = 65  # minimum extension length of the arm, cm
        self.actuator_ext_len: float = 0  # Actuator extension length, cm
        self.actuator_total_len: float = (
            self.actuator_min_len + self.actuator_ext_len
        )  # Total actuator length, cm

        # Vectors representing the arm segments
        self.base_segment: np.ndarray = np.array(
            [0, self.base_segment_height, 0]
        )  # Fixed base segment
        self.segment1: np.ndarray = np.array(
            [self.offset, 0, 0]
        )  # First segment, the offset
        self.segment2: np.ndarray = np.array(
            [0, 0, self.actuator_min_len]
        )  # Second segment, the actuator fixed length part
        self.segment3: np.ndarray = np.array(
            [0, 0, 0]
        )  # Third segment, the actuator extension part

        # coordinate of arm end effector
        self.endpoint: Point = Point(
            self.offset, self.base_segment_height, self.actuator_min_len
        )

    def __str__(self):
        return f"""
        origin: {self.origin}, 
        yaw angle: {math.degrees(self.servo1_angle)}, 
        pitch angle: {math.degrees(self.servo2_angle)}, 
        arm_length: {self.actuator_min_len}, 
        actuator_ext: {self.actuator_ext_len},
        endpoint: {(self.endpoint.x, self.endpoint.y, self.endpoint.z)}
        """

    def reset_position(self):
        # Reset the arm to the initial position
        self.endpoint = Point(
            self.offset, self.base_segment_height, self.actuator_min_len
        )

        # Reset the servo angles using IK Engine
        self.servo1_angle = ArmIK.calc_servo1_angle(
            self.endpoint.x, self.endpoint.z, self.offset
        )
        self.servo2_angle = ArmIK.calc_servo2_angle(
            self.endpoint.x, self.endpoint.y, self.endpoint.z, self.offset
        )
        self.actuator_ext_len = ArmIK.calc_actuator_extension(
            self.endpoint.x,
            self.endpoint.y,
            self.endpoint.z,
            self.offset,
            self.actuator_min_len,
        )

    @staticmethod
    def rotate_vector(vector: np.ndarray, axis: np.ndarray, theta: float) -> np.ndarray:
        """
        Return the rotation matrix associated with counterclockwise rotation about
        the given axis by theta radians.
        """
        global DEBUG
        if DEBUG:
            print(f"vector: {vector}, axis: {axis}, theta: {math.degrees(theta)}")

        axis = axis / math.sqrt(np.dot(axis, axis))
        a = math.cos(theta / 2.0)
        b, c, d = -axis * math.sin(theta / 2.0)
        aa, bb, cc, dd = a * a, b * b, c * c, d * d
        bc, ad, ac, ab, bd, cd = b * c, a * d, a * c, a * b, b * d, c * d
        rotation_matrix = np.array(
            [
                [aa + bb - cc - dd, 2 * (bc + ad), 2 * (bd - ac)],
                [2 * (bc - ad), aa + cc - bb - dd, 2 * (cd + ab)],
                [2 * (bd + ac), 2 * (cd - ab), aa + dd - bb - cc],
            ]
        )
        return np.dot(rotation_matrix, vector)

class ArmIK:
    """
    Inverse Kinematics for the arm
    """

    @staticmethod
    def calc_servo1_angle(x: float, z: float, offset: float) -> float:
        """
        Calculate the servo1 angle (yaw) in radians
        x: left and right
        z: depth (forward and backward)
        offset: offset from the center of the robot
        """
        beta = math.atan2(z, x)
        alpha = math.asin(offset / math.sqrt(x**2 + z**2))
        return beta + alpha

    @staticmethod
    def calc_servo2_angle(x: float, y: float, z: float, d: float) -> float:
        """
        Calculate the servo2 angle (pitch) in radians
        x: left and right
        y: up and down
        z: depth (forward and backward)
        """
        return math.atan2(y, math.sqrt(x**2 + z**2 - d**2))

    @staticmethod
    def calc_actuator_extension(
        x: float, y: float, z: float, d: float, min_length: float
    ) -> float:
        """
        Calculate the actuator extension length
        x: left and right
        y: up and down
        z: depth (forward and backward)
        d: distance from the shoulder to the wrist
        """
        return math.sqrt(x**2 + y**2 + z**2 - d**2) - min_length


class ArmFK:
    """
    Forward Kinematics for the arm
    """

    def update_segment1_vector(self, servo1_angle: float, segment1: np.ndarray):
        """
        Rotate the segment1 vector based on the servo1 angle.
        Returns rotated vector.
        """
        # Rotate the segment1 vector around the y-axis (vertical) by the servo1 angle
        return Arm.rotate_vector(segment1, np.array([0, 1, 0]), servo1_angle)

    def update_segment2_vector(
        self, servo1_angle: float, servo2_angle: float, segment2: np.ndarray
    ):
        """
        Rotate the segment2 vector based on the servo2 angle and servo1 angle.
        """
        # Rotate the segment2 vector around the x-axis (horizontal) by the servo2 angle
        # The order is important, first rotate around the x-axis, then around the y-axis
        segment2 = Arm.rotate_vector(segment2, np.array([1, 0, 0]), servo2_angle)
        # Rotate the segment2 vector around the y-axis (vertical) by the servo1 angle
        segment2 = Arm.rotate_vector(segment2, np.array([0, 1, 0]), servo1_angle)
        return segment2

    def update_segment3_vector(
        self,
        actuator_ext_len: float,
        segment3: np.ndarray,
        servo1_angle: float,
        servo2_angle: float,
    ):
        """
        Update the segment3 vector based on the actuator extension length.
        """
        segment3[2] = actuator_ext_len
        # Rotate the segment3 vector around the x-axis (horizontal) by the servo2 angle
        # The order is important, first rotate around the x-axis, then around the y-axis
        segment3 = Arm.rotate_vector(segment3, np.array([1, 0, 0]), servo2_angle)
        # Rotate the segment3 vector around the y-axis (vertical) by the servo1 angle
        segment3 = Arm.rotate_vector(segment3, np.array([0, 1, 0]), servo1_angle)
        return segment3

    def draw_arm(
        self,
        ax: plt.Axes,
        origin: np.ndarray,
        v1: np.ndarray,
        v2: np.ndarray,
        v3: np.ndarray,
        v4: np.ndarray,
    ):
        """
        Draw the arm one vector at a time, each one a different colour
        """
        endpoint1 = origin + v1
        endpoint2 = endpoint1 + v2
        endpoint3 = endpoint2 + v3
        endpoint4 = endpoint3 + v4  # endpoint4 is the end effector

        # Draw the arm
        self.plot_segment(ax, origin, endpoint1, "r")
        self.plot_segment(ax, endpoint1, endpoint2, "g")
        self.plot_segment(ax, endpoint2, endpoint3, "b")
        self.plot_segment(ax, endpoint3, endpoint4, "k")

    @staticmethod
    def plot_segment(ax: plt.Axes, start: np.ndarray, end: np.ndarray, color: str)
        """
        Plot a segment of the arm
        """
        ax.plot([start[0], end[0]], [start[1], end[1]], [start[2], end[2]], color=color)
