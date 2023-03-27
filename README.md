Iterative LQR algorithm on a quadrotor

The goal of this project is to control a 2D quadrotor to perform acrobatic moves. It has been carried out in 4 parts of increasing complexity. The final part is the implementation of the iterative LQR (iLQR) algorithm.

Part-1

Computing u1* and  u2* such that the robot stays at a stationary position, where u1* and u2* are forces produced by the rotors.

Part-2

After the u* is calculated to keep the quad-rotor stationary, in this part I design a controller that ensures the stationary position of the quad-rotor even when pushed by random disturbance (e.g. due to wind). Therefore, designing a LQR controller to keep the quad-rotor at a predefined position.

Part-3

In this part my goal is to make the quad-rotor follow a desired trajectory.

Part-4

In this final part of the project my goal is to make the quad-rotor do a flip. This is achieved by using the iterative LQR algorithm.