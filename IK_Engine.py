import math

import matplotlib.pyplot as plt
import numpy as np

DEBUG = True
SCALER = 4


class Arm:
    def __init__(self):
        # Servo angles
        self.servo1_angle: float = math.pi / 2  # servo1 (yaw) angle in radians
        self.servo2_angle: float = 0  # servo2 (pitch) angle in radians:

        # Arm dimensions
        self.base_segment_height: float = 20  # Length of vertical segment, cm
        self.offset: float = 12  # Offset from the center of the robot, cm
        self.actuator_min_len: float = 65  # minimum extension length of the arm, cm
        self.actuator_ext_len: float = 0  # Actuator extension length, cm
        self._scale_values()  # Scale values based on SCALER

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

        self.end_effector_point = (
            self.segment1 + self.segment2 + self.segment3
        )  # End effector point
        self.reset_point = (
            self.end_effector_point
        )  # Reset point for the arm. Used to reset the arm to the initial position. DO NOT CHANGE.
        self.target_point = (
            self.end_effector_point
        )  # Target point for the arm. Used by IK Engine. Starts at the end effector point.
        self.origin = np.array([0, 0, 0])

    def __str__(self):
        return f"""
        yaw angle: {math.degrees(self.servo1_angle)}, 
        pitch angle: {math.degrees(self.servo2_angle)}, 
        actuator_ext: {self.actuator_ext_len},
        endpoint: {(self.end_effector_point[0], self.end_effector_point[1], self.end_effector_point[2])}
        """

    def _scale_values(self):
        self.base_segment_height = self.base_segment_height * SCALER
        self.offset = self.offset * SCALER
        self.actuator_min_len = self.actuator_min_len * SCALER
        self.actuator_ext_len = self.actuator_ext_len * SCALER

    def fully_define_position(self):
        """
        Fully define the position of the arm based on the target point.
        First calculates the servo angles and actuator extension length using IK Engine,
        and updates these attributes in the Arm object.
        Then calculates and updates the vectors representing the arm segments using FK Engine.

        """
        if DEBUG:
            print(
                "Point for IK Engine: ",
                self.target_point[0],
                self.target_point[1],
                self.target_point[2],
            )

        servo1_angle:float = ArmIK.calc_servo1_angle(
            self.target_point[0], self.target_point[2], self.offset
        )
        servo2_angle:float = ArmIK.calc_servo2_angle(
            self.target_point[0],
            self.target_point[1],
            self.target_point[2],
            self.offset,
        )
        self.actuator_ext_len = ArmIK.calc_actuator_extension(
            self.target_point[0],
            self.target_point[1],
            self.target_point[2],
            self.offset,
            self.actuator_min_len,
        )
        # Delta angles
        d_angle1 = self.servo1_angle - servo1_angle
        d_angle2 = self.servo2_angle - servo2_angle
        
        if DEBUG:
            #print delta angles
            print(f"d_angle1: {math.degrees(d_angle1)}, d_angle2: {math.degrees(d_angle2)}")

        self.segment1 = ArmFK.update_segment1_vector(d_angle1, self.segment1)
        self.segment2 = ArmFK.update_segment2_vector(
            d_angle1, d_angle2, self.segment2
        )
        self.segment3 = ArmFK.update_segment3_vector(
            self.actuator_ext_len,
            self.segment3,
            d_angle1,
            d_angle2,
        )

        # Update angles
        self.servo1_angle = servo1_angle
        self.servo2_angle = servo2_angle
        if DEBUG:
            print(
                f"""
                servo1: {math.degrees(self.servo1_angle)},
                servo2: {math.degrees(self.servo2_angle)},
                actuator: {self.actuator_ext_len}
                ---
                segment1: {self.segment1},
                segment2: {self.segment2},
                segment3: {self.segment3}
                """
            )

    def reset_position(self):
        # Reset the arm to the initial position
        self.end_effector_point = self.reset_point
        # Reset the servo angles using IK Engine
        self.servo1_angle = ArmIK.calc_servo1_angle(
            self.end_effector_point[0], self.end_effector_point[2], self.offset
        )
        self.servo2_angle = ArmIK.calc_servo2_angle(
            self.end_effector_point[0],
            self.end_effector_point[1],
            self.end_effector_point[2],
            self.offset,
        )
        self.actuator_ext_len = ArmIK.calc_actuator_extension(
            self.end_effector_point[0],
            self.end_effector_point[1],
            self.end_effector_point[2],
            self.offset,
            self.actuator_min_len,
        )


def rotate_vector(vector: np.ndarray, axis: np.ndarray, theta: float) -> np.ndarray:
    """
    Return the rotation matrix associated with counterclockwise rotation about
    the given axis by theta radians.
    """
    global DEBUG
#    if DEBUG:
#        print(f"vector: {vector}, axis: {axis}, theta: {math.degrees(theta)}")

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
    result = np.dot(rotation_matrix, vector)
    if DEBUG:
        print(f"({result[0]}, {result[1]}, {result[2]})")
    return result


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
    def calc_servo2_angle(x: float, y: float, z: float, offset: float) -> float:
        """
        Calculate the servo2 angle (pitch) in radians
        x: left and right
        y: up and down
        z: depth (forward and backward)
        """
        return math.atan2(y, math.sqrt(x**2 + z**2 - offset**2))

    @staticmethod
    def calc_actuator_extension(
        x: float, y: float, z: float, offset: float, min_length: float
    ) -> float:
        """
        Calculate the actuator extension length
        x: left and right
        y: up and down
        z: depth (forward and backward)
        d: distance from the shoulder to the wrist
        """
        value = math.sqrt(x**2 + y**2 + z**2 - offset**2) - min_length
        if value < 0:
            return 0
        return value


class ArmFK:
    """
    Forward Kinematics for the arm
    """

    @staticmethod
    def update_segment1_vector(servo1_angle: float, segment1: np.ndarray) -> np.ndarray:
        """
        Rotate the segment1 vector based on the servo1 angle.
        Returns rotated vector.
        """
        # Rotate the segment1 vector around the y-axis (vertical) by the servo1 angle
        return rotate_vector(segment1, np.array([0, 1, 0]), servo1_angle)

    @staticmethod
    def update_segment2_vector(
        servo1_angle: float, servo2_angle: float, segment2: np.ndarray
    ) -> np.ndarray:
        """
        Rotate the segment2 vector based on the servo2 angle and servo1 angle.
        """
        # Rotate the segment2 vector around the x-axis (horizontal) by the servo2 angle
        # The order is important, first rotate around the x-axis, then around the y-axis
        segment2 = rotate_vector(segment2, np.array([1, 0, 0]), servo2_angle)
        # Rotate the segment2 vector around the y-axis (vertical) by the servo1 angle
        segment2 = rotate_vector(segment2, np.array([0, 1, 0]), servo1_angle)
        return segment2

    @staticmethod
    def update_segment3_vector(
        actuator_ext_len: float,
        segment3: np.ndarray,
        servo1_angle: float,
        servo2_angle: float,
    ) -> np.ndarray:
        """
        Update the segment3 vector based on the actuator extension length.
        """
        segment3[2] = actuator_ext_len
        # Rotate the segment3 vector around the x-axis (horizontal) by the servo2 angle
        # The order is important, first rotate around the x-axis, then around the y-axis
        segment3 = rotate_vector(segment3, np.array([1, 0, 0]), servo2_angle)
        # Rotate the segment3 vector around the y-axis (vertical) by the servo1 angle
        segment3 = rotate_vector(segment3, np.array([0, 1, 0]), servo1_angle)
        return segment3

    @staticmethod
    def plot_segment(ax: plt.Axes, start: np.ndarray, end: np.ndarray, color: str, linewidth: int = 1):
        """
        Plot a segment of the arm
        """
        # NOTE: The y and z axes are swapped in the plot
        ax.plot([start[0], end[0]], [start[2], end[2]], [start[1], end[1]], color=color, linewidth=linewidth)
        #ax.plot([start[0], end[0]], [start[1], end[1]], [start[2], end[2]], color=color)    # Original. no swap.

    @staticmethod
    def draw_arm(
        ax: plt.Axes,
        origin: np.ndarray,
        v1: np.ndarray,
        v2: np.ndarray,
        v3: np.ndarray,
        target: np.ndarray,
    ):
        """
        Draw the arm one vector at a time, each one a different colour
        """
        endpoint1 = origin - v1
        endpoint2 = origin + v2
        endpoint3 = endpoint2 + v3
        # Print 
        print("target point (draw): ", target[0], target[1], target[2])

        # Draw the arm
        ArmFK.plot_segment(ax, origin, endpoint1, "r", 4)
        ArmFK.plot_segment(ax, origin, endpoint2, "g", 4)
        ArmFK.plot_segment(ax, endpoint2, endpoint3, "b", 4)
        ArmFK.plot_segment(ax, endpoint3, target, "k", 2)


if __name__ == "__main__":
    # convert to degrees
    v1 = np.array([20, 0, 0])
    rotation = --180
    v2 = rotate_vector(v1, np.array([0, 1, 0]), math.radians(rotation))
    print(v1)
    print("({}, {}, {})".format(v2[0], v2[1], v2[2]))

