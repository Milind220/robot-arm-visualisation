import matplotlib.pyplot as plt
from matplotlib.figure import Figure
from matplotlib.animation import FuncAnimation
from mpl_toolkits.mplot3d import Axes3D
from pynput import keyboard

from IK_Engine import Arm, ArmFK


class RobotArmController:
    def __init__(self):
        # Initialize the robot arm's state
        self.robot = Arm()
        self.increment = 10  # Increment for movements and rotations
        print("target point: ", self.robot.target_point[0], self.robot.target_point[1], self.robot.target_point[2]) 

    def move(self, dx=0, dy=0, dz=0):
        """Moves the arm's target position by the given deltas"""
        self.robot.target_point[0] += dx
        self.robot.target_point[1] += dy
        self.robot.target_point[2] += dz
        self.update_arm()

    #def rotate(self, dyaw=0, dpitch=0, droll=0):
    #    """Rotates the arm's target orientation by the given deltas"""

    #def extend_actuator(self, dlength=0):
    #    """Extends or retracts the arm's actuator by the given delta"""

    def update_arm(self):
        """Updates the arm's configuration based on current target position"""
        self.robot.fully_define_position()

    def reset_position(self):
        """Resets the arm's target position and orientation"""
        self.robot.reset_position()


class ArmVisualizer:
    def __init__(self, controller: RobotArmController):
        # Initialize the visualizer
        self.controller = controller
        self.fig: Figure = plt.figure()
        self.ax: Axes3D = self.fig.add_subplot(111,projection="3d")
        self.configure_plot()

    def configure_plot(self):
        WINDOW_SIZE: int = 500 
        start_height: int = 170
        self.ax.set_xlim3d(-WINDOW_SIZE / 2, WINDOW_SIZE / 2)
        self.ax.set_ylim3d(-WINDOW_SIZE / 10, 9*WINDOW_SIZE / 10)
        self.ax.set_zlim3d(-start_height, WINDOW_SIZE - start_height)
        self.ax.set_xlabel("x (cm)")
        self.ax.set_ylabel("z (cm)")
        self.ax.set_zlabel("y (cm)")

    def update_plot(self):
        self.ax.clear()
        self.configure_plot()
        ArmFK.draw_arm(
            self.ax,
            self.controller.robot.origin,
            self.controller.robot.base_segment,
            self.controller.robot.segment1,
            self.controller.robot.segment2,
            self.controller.robot.target_point
        )
   
    def on_press(self, key):
        try:
            if key.char == 'x':
                self.selected_axis = 'x'
            elif key.char == 'y':
                self.selected_axis = 'y'
            elif key.char == 'z':
                self.selected_axis = 'z'
        except AttributeError:
            if key == keyboard.Key.up:
                if self.selected_axis == 'x':
                    self.controller.move(dx=self.controller.increment)
                elif self.selected_axis == 'y':
                    self.controller.move(dy=self.controller.increment)
                elif self.selected_axis == 'z':
                    self.controller.move(dz=self.controller.increment)
            elif key == keyboard.Key.down:
                if self.selected_axis == 'x':
                    self.controller.move(dx=-self.controller.increment)
                elif self.selected_axis == 'y':
                    self.controller.move(dy=-self.controller.increment)
                elif self.selected_axis == 'z':
                    self.controller.move(dz=-self.controller.increment)

    def run(self):
        """Starts the animation loop."""
        ani = FuncAnimation(self.fig, lambda frame: self.update_plot(), interval=50)
        plt.show()


def main():
    controller = RobotArmController()
    visualizer = ArmVisualizer(controller)

    # setup keyboard listener
    listener = keyboard.Listener(on_press=visualizer.on_press)
    listener.start()

    visualizer.run()

if __name__ == "__main__":
    main()
