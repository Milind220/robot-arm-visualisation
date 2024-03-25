# 3-DOF Robot Arm visualization tool
The arm in question has two rotational degrees of freedom, and one translational degree of freedom.

## Visualization Usage
To try the visualization tool out, you need to run the [animate.py](https://github.com/Milind220/robot-arm-visualisation/edit/main/animate.py) file with the proper libraries installed. (matplotlib 3.0.3 supported).

You can then use keyboard controls: use x, y, z, a, p, e to select (x axis, y axis, z axis, yaw, pitch, extension), and then the up and down buttons to increment the selected position. If you click '1' on your keyboard, it will reset the position. 

Right now, if you try to go to an impossible pose that would result in collisions, the body will do some weird things. If that happens, just click "1" on your keyboard to reset the position.

based on the fantastic [vis-tool](https://github.com/adham-elarabawy/open-quadruped/tree/master/vis-tool) script for open-quadruped 
